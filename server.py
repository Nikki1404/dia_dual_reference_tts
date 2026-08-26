#!/usr/bin/env python3

import io
import json
import os
import random
import re
import struct
import time
import uuid
from math import gcd

import numpy as np
import soundfile as sf
import torch

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from scipy.signal import resample_poly
from transformers import AutoProcessor, DiaForConditionalGeneration


# =============================================================================
# CONFIG
# =============================================================================

MODEL_ID = os.getenv(
    "MODEL_ID",
    "nari-labs/Dia-1.6B-0626",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = (
    torch.float16
    if DEVICE == "cuda"
    else torch.float32
)

SAMPLE_RATE = 44100

DEFAULT_PRODUCTION_MAX_NEW_TOKENS = 3072
DEFAULT_SEED_MAX_NEW_TOKENS = 1024

REFERENCE_SILENCE_MS = 180


# Production: less random
PRODUCTION_GUIDANCE_SCALE = 3.0
PRODUCTION_TEMPERATURE = 1.0
PRODUCTION_TOP_P = 0.95
PRODUCTION_TOP_K = 50


# Seed auditioning
SEED_GUIDANCE_SCALE = 3.0
SEED_TEMPERATURE = 1.8
SEED_TOP_P = 0.90
SEED_TOP_K = 45


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="Dia Speaker-Wise Dual Reference TTS",
    version="3.0.0",
)

processor = None
model = None
model_load_ms = None


# =============================================================================
# SEED REQUEST
# =============================================================================

class SeedTTSRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
    )

    seed: int = Field(
        ...,
        ge=0,
        le=2147483647,
    )

    max_new_tokens: int = Field(
        default=DEFAULT_SEED_MAX_NEW_TOKENS,
        ge=256,
        le=4096,
    )


# =============================================================================
# CUDA / SEED
# =============================================================================

def sync_cuda():

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def set_seed(seed: int):

    random.seed(seed)

    np.random.seed(
        seed % (2**32 - 1)
    )

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)


# =============================================================================
# TEXT
# =============================================================================

def normalize_tags(text: str):

    text = re.sub(
        r"\[\s*s1\s*\]",
        "[S1]",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\[\s*s2\s*\]",
        "[S2]",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def strip_leading_tag(
    text: str,
    tag: str,
):

    text = normalize_tags(text)

    text = re.sub(
        rf"^\s*\[{tag}\]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


# =============================================================================
# SPEAKER-WISE PARSER
#
# One complete [S1] or [S2] turn = one chunk.
#
# Works even without newline:
#
# [S1] hello.[S2] hi.[S1] next...
# =============================================================================

def parse_turns(text: str):

    text = normalize_tags(text)

    matches = list(
        re.finditer(
            r"\[(S1|S2)\]\s*",
            text,
            flags=re.IGNORECASE,
        )
    )

    if not matches:

        raise ValueError(
            "Transcript must contain [S1] or [S2]."
        )

    turns = []

    for index, match in enumerate(matches):

        speaker = match.group(1).upper()

        start = match.end()

        if index + 1 < len(matches):

            end = matches[
                index + 1
            ].start()

        else:

            end = len(text)

        speech = text[
            start:end
        ].strip()

        if speech:

            turns.append(
                (
                    speaker,
                    speech,
                )
            )

    if not turns:

        raise ValueError(
            "No speech found after speaker tags."
        )

    return turns


# =============================================================================
# TOKEN PROTECTION
#
# Helps:
#
# Hello.
# Yes.
# English please.
# My member ID is 2043.
#
# and reduces early EOS for final words.
# =============================================================================

def generation_limits(
    speech: str,
    requested_max: int,
):

    words = len(
        speech.split()
    )

    words = max(
        words,
        1,
    )


    if words == 1:

        min_tokens = 128

        local_max = min(
            requested_max,
            512,
        )


    elif words <= 3:

        min_tokens = 160

        local_max = min(
            requested_max,
            640,
        )


    elif words <= 6:

        min_tokens = 192

        local_max = min(
            requested_max,
            768,
        )


    elif words <= 12:

        min_tokens = 256

        local_max = min(
            requested_max,
            1280,
        )


    elif words <= 20:

        min_tokens = 384

        local_max = min(
            requested_max,
            1792,
        )


    elif words <= 30:

        min_tokens = 512

        local_max = min(
            requested_max,
            2304,
        )


    else:

        estimated_seconds = (
            words / 2.8
        )

        estimated_tokens = int(
            estimated_seconds
            * 128
            * 0.65
        )

        min_tokens = max(
            512,
            estimated_tokens,
        )

        min_tokens = min(
            min_tokens,
            1536,
        )

        local_max = (
            requested_max
        )


    if min_tokens >= local_max:

        min_tokens = max(
            1,
            local_max - 32,
        )


    return (
        min_tokens,
        local_max,
    )


def minimum_reasonable_duration(
    speech: str,
):

    words = max(
        1,
        len(
            speech.split()
        ),
    )

    if words == 1:
        return 0.30

    if words <= 3:
        return 0.50

    return max(
        0.65,
        words / 5.2,
    )


# =============================================================================
# AUDIO
# =============================================================================

def read_reference_wav(
    wav_bytes: bytes,
):

    audio, sr = sf.read(
        io.BytesIO(
            wav_bytes
        ),
        dtype="float32",
    )

    if audio.ndim == 2:

        audio = np.mean(
            audio,
            axis=1,
        )


    audio = np.asarray(
        audio,
        dtype=np.float32,
    )


    if len(audio) == 0:

        raise ValueError(
            "Reference WAV is empty."
        )


    if sr != SAMPLE_RATE:

        divisor = gcd(
            sr,
            SAMPLE_RATE,
        )

        audio = resample_poly(
            audio,
            SAMPLE_RATE // divisor,
            sr // divisor,
        ).astype(
            np.float32
        )


    return audio


def trim_reference_silence(
    audio,
    threshold_ratio=0.02,
    padding_ms=40,
):

    if len(audio) == 0:

        return audio


    peak = float(
        np.max(
            np.abs(audio)
        )
    )


    if peak <= 0:

        return audio


    active = np.flatnonzero(
        np.abs(audio)
        >= peak * threshold_ratio
    )


    if len(active) == 0:

        return audio


    padding = int(
        SAMPLE_RATE
        * padding_ms
        / 1000
    )


    start = max(
        0,
        int(active[0])
        - padding,
    )


    end = min(
        len(audio),
        int(active[-1])
        + padding
        + 1,
    )


    return audio[
        start:end
    ]


def combine_references(
    first_audio,
    second_audio,
):

    first_audio = (
        trim_reference_silence(
            first_audio
        )
    )

    second_audio = (
        trim_reference_silence(
            second_audio
        )
    )


    silence = np.zeros(
        int(
            SAMPLE_RATE
            * REFERENCE_SILENCE_MS
            / 1000
        ),
        dtype=np.float32,
    )


    return np.concatenate(
        [
            first_audio,
            silence,
            second_audio,
        ]
    ).astype(
        np.float32
    )


# =============================================================================
# DECODE
# =============================================================================

def decode_conditioned(
    outputs,
    prompt_len,
):

    decoded = processor.batch_decode(
        outputs,
        audio_prompt_len=prompt_len,
    )


    audio = (
        decoded[0]
        if isinstance(
            decoded,
            (list, tuple),
        )
        else decoded
    )


    if torch.is_tensor(audio):

        audio = (
            audio
            .detach()
            .float()
            .cpu()
            .numpy()
        )


    audio = np.squeeze(
        np.asarray(
            audio,
            dtype=np.float32,
        )
    )


    if audio.ndim != 1:

        raise RuntimeError(
            f"Unexpected audio shape: {audio.shape}"
        )


    if len(audio) == 0:

        raise RuntimeError(
            "Generated audio is empty."
        )


    return audio


def decode_plain(outputs):

    decoded = processor.batch_decode(
        outputs
    )


    audio = (
        decoded[0]
        if isinstance(
            decoded,
            (list, tuple),
        )
        else decoded
    )


    if torch.is_tensor(audio):

        audio = (
            audio
            .detach()
            .float()
            .cpu()
            .numpy()
        )


    return np.squeeze(
        np.asarray(
            audio,
            dtype=np.float32,
        )
    )


# =============================================================================
# STREAM
# =============================================================================

def make_frame(
    metadata,
    pcm=b"",
):

    metadata_bytes = json.dumps(
        metadata
    ).encode(
        "utf-8"
    )


    return (
        struct.pack(
            ">I",
            len(metadata_bytes),
        )
        +
        metadata_bytes
        +
        struct.pack(
            ">Q",
            len(pcm),
        )
        +
        pcm
    )


# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
def load_model():

    global processor
    global model
    global model_load_ms


    print("")
    print("=" * 80)
    print("DIA TTS STARTUP")
    print("=" * 80)

    print(
        f"Model             : {MODEL_ID}"
    )

    print(
        f"Device            : {DEVICE}"
    )

    print(
        f"PyTorch           : {torch.__version__}"
    )

    print(
        f"PyTorch CUDA      : {torch.version.cuda}"
    )

    print(
        f"CUDA available    : {torch.cuda.is_available()}"
    )


    if torch.cuda.is_available():

        print(
            f"GPU               : "
            f"{torch.cuda.get_device_name(0)}"
        )


    print(
        "Chunk mode        : complete speaker turn"
    )

    print(
        "Short-text guard  : enabled"
    )

    print(
        "Retry guard       : enabled"
    )

    print("=" * 80)


    start = (
        time.perf_counter_ns()
    )


    processor = (
        AutoProcessor
        .from_pretrained(
            MODEL_ID
        )
    )


    model = (
        DiaForConditionalGeneration
        .from_pretrained(
            MODEL_ID
        )
    )


    model = model.to(
        device=DEVICE,
        dtype=DTYPE,
    )


    model.eval()


    sync_cuda()


    model_load_ms = (
        time.perf_counter_ns()
        - start
    ) / 1_000_000


    print(
        f"Model loaded      : "
        f"{model_load_ms:.2f} ms"
    )

    print("=" * 80)


# =============================================================================
# HEALTH
# =============================================================================

@app.get("/health")
def health():

    return {

        "status":
            "ok",

        "model":
            MODEL_ID,

        "device":
            DEVICE,

        "cuda_available":
            torch.cuda.is_available(),

        "chunk_mode":
            "complete_speaker_turn",

        "short_text_guard":
            True,

        "retry_guard":
            True,

        "endpoints": {

            "/tts":
                "production",

            "/tts_seed":
                "seed audition",
        },
    }


# =============================================================================
# GENERATE ONE TURN
# =============================================================================

def generate_one_turn(
    conditioned_text,
    reference_audio,
    speech,
    requested_max,
):

    (
        min_tokens,
        local_max,
    ) = generation_limits(
        speech,
        requested_max,
    )


    min_duration = (
        minimum_reasonable_duration(
            speech
        )
    )


    preprocess_start = (
        time.perf_counter_ns()
    )


    inputs = processor(
        text=[
            conditioned_text
        ],
        audio=
            reference_audio,
        padding=True,
        return_tensors="pt",
    ).to(
        model.device
    )


    prompt_len = (
        processor
        .get_audio_prompt_len(
            inputs[
                "decoder_attention_mask"
            ]
        )
    )


    sync_cuda()


    preprocess_ms = (
        time.perf_counter_ns()
        - preprocess_start
    ) / 1_000_000


    inference_start = (
        time.perf_counter_ns()
    )


    with torch.inference_mode():

        outputs = model.generate(
            **inputs,

            min_new_tokens=
                min_tokens,

            max_new_tokens=
                local_max,

            guidance_scale=
                PRODUCTION_GUIDANCE_SCALE,

            temperature=
                PRODUCTION_TEMPERATURE,

            top_p=
                PRODUCTION_TOP_P,

            top_k=
                PRODUCTION_TOP_K,
        )


    sync_cuda()


    inference_ms = (
        time.perf_counter_ns()
        - inference_start
    ) / 1_000_000


    decode_start = (
        time.perf_counter_ns()
    )


    audio = decode_conditioned(
        outputs,
        prompt_len,
    )


    decode_ms = (
        time.perf_counter_ns()
        - decode_start
    ) / 1_000_000


    duration = (
        len(audio)
        / SAMPLE_RATE
    )


    retried = False


    # =========================================================================
    # RETRY IF CLEARLY TRUNCATED
    # =========================================================================

    if duration < min_duration:

        retried = True


        print(
            "[WARN] Short output detected. "
            "Retrying turn once."
        )


        stronger_min = min(
            local_max - 16,

            max(
                min_tokens + 96,

                int(
                    min_tokens
                    * 1.30
                ),
            ),
        )


        retry_start = (
            time.perf_counter_ns()
        )


        with torch.inference_mode():

            retry_outputs = model.generate(
                **inputs,

                min_new_tokens=
                    stronger_min,

                max_new_tokens=
                    local_max,

                guidance_scale=
                    PRODUCTION_GUIDANCE_SCALE,

                temperature=
                    0.8,

                top_p=
                    0.95,

                top_k=
                    50,
            )


        sync_cuda()


        retry_inference_ms = (
            time.perf_counter_ns()
            - retry_start
        ) / 1_000_000


        retry_decode_start = (
            time.perf_counter_ns()
        )


        retry_audio = decode_conditioned(
            retry_outputs,
            prompt_len,
        )


        retry_decode_ms = (
            time.perf_counter_ns()
            - retry_decode_start
        ) / 1_000_000


        retry_duration = (
            len(retry_audio)
            / SAMPLE_RATE
        )


        if retry_duration > duration:

            audio = retry_audio

            duration = (
                retry_duration
            )


        inference_ms += (
            retry_inference_ms
        )

        decode_ms += (
            retry_decode_ms
        )


    return (
        audio,
        preprocess_ms,
        inference_ms,
        decode_ms,
        min_tokens,
        local_max,
        retried,
    )


# =============================================================================
# PRODUCTION GENERATOR
# =============================================================================

def production_generator(
    text,
    agent_text,
    customer_text,
    agent_audio,
    customer_audio,
    max_new_tokens,
    request_id,
):

    start = (
        time.perf_counter_ns()
    )


    turns = parse_turns(
        text
    )


    # =========================================================================
    # AGENT REFERENCE
    #
    # internal S1 = agent
    # =========================================================================

    agent_ref_audio = (
        combine_references(
            agent_audio,
            customer_audio,
        )
    )


    agent_ref_text = (
        f"[S1] {agent_text} "
        f"[S2] {customer_text}"
    )


    # =========================================================================
    # CUSTOMER REFERENCE
    #
    # internal S1 = customer
    # =========================================================================

    customer_ref_audio = (
        combine_references(
            customer_audio,
            agent_audio,
        )
    )


    customer_ref_text = (
        f"[S1] {customer_text} "
        f"[S2] {agent_text}"
    )


    print("")
    print("=" * 80)
    print("PRODUCTION REQUEST")
    print("=" * 80)

    print(
        f"Turns             : {len(turns)}"
    )

    print(
        f"Max new tokens    : {max_new_tokens}"
    )

    print("=" * 80)


    total_preprocess = 0.0
    total_inference = 0.0
    total_decode = 0.0
    total_samples = 0


    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()


    for index, (
        speaker,
        speech,
    ) in enumerate(
        turns,
        start=1,
    ):


        if speaker == "S1":

            voice = "agent"

            ref_audio = (
                agent_ref_audio
            )

            ref_text = (
                agent_ref_text
            )


        else:

            voice = "customer"

            ref_audio = (
                customer_ref_audio
            )

            ref_text = (
                customer_ref_text
            )


        # Target always internal S1
        target = (
            f"[S1] {speech}"
        )


        conditioned = (
            f"{ref_text} "
            f"{target}"
        )


        print("")
        print("-" * 80)

        print(
            f"Chunk {index}/{len(turns)}"
        )

        print(
            f"Speaker           : "
            f"{speaker} ({voice})"
        )

        print(
            f"Text              : "
            f"{speech}"
        )


        (
            audio,
            preprocess_ms,
            inference_ms,
            decode_ms,
            min_tokens,
            local_max,
            retried,
        ) = generate_one_turn(
            conditioned,
            ref_audio,
            speech,
            max_new_tokens,
        )


        total_preprocess += (
            preprocess_ms
        )

        total_inference += (
            inference_ms
        )

        total_decode += (
            decode_ms
        )

        total_samples += (
            len(audio)
        )


        duration = (
            len(audio)
            / SAMPLE_RATE
        )


        pcm = (
            np.clip(
                audio,
                -1.0,
                1.0,
            )
            * 32767
        ).astype(
            "<i2"
        )


        yield make_frame(
            {

                "type":
                    "audio",

                "chunk_index":
                    index,

                "chunk_count":
                    len(turns),

                "speaker":
                    speaker,

                "voice":
                    voice,

                "chunk_text":
                    speech,

                "chunk_words":
                    len(
                        speech.split()
                    ),

                "sample_rate":
                    SAMPLE_RATE,

                "audio_duration_s":
                    duration,

                "inference_ms":
                    inference_ms,

                "min_new_tokens":
                    min_tokens,

                "local_max_new_tokens":
                    local_max,

                "retried":
                    retried,
            },

            pcm.tobytes(),
        )


    server_total = (
        time.perf_counter_ns()
        - start
    ) / 1_000_000


    audio_duration = (
        total_samples
        / SAMPLE_RATE
    )


    yield make_frame(
        {

            "type":
                "end",

            "chunk_count":
                len(turns),

            "preprocess_ms":
                total_preprocess,

            "inference_ms":
                total_inference,

            "decode_ms":
                total_decode,

            "server_total_ms":
                server_total,

            "audio_duration_s":
                audio_duration,

            "generation_rtf":
                (
                    (
                        total_inference
                        / 1000
                    )
                    / audio_duration

                    if audio_duration
                    else 0
                ),

            "total_rtf":
                (
                    (
                        server_total
                        / 1000
                    )
                    / audio_duration

                    if audio_duration
                    else 0
                ),

            "gpu_peak_mb":
                (
                    torch.cuda
                    .max_memory_allocated()
                    / 1024**2

                    if torch.cuda.is_available()

                    else 0
                ),
        }
    )


# =============================================================================
# /tts
# =============================================================================

@app.post("/tts")
def tts(

    text: str = Form(...),

    agent_reference_text:
        str = Form(...),

    customer_reference_text:
        str = Form(...),

    max_new_tokens:
        int = Form(
            DEFAULT_PRODUCTION_MAX_NEW_TOKENS
        ),

    agent_audio:
        UploadFile = File(...),

    customer_audio:
        UploadFile = File(...),
):


    try:

        agent_bytes = (
            agent_audio.file.read()
        )

        customer_bytes = (
            customer_audio.file.read()
        )


        agent_array = (
            read_reference_wav(
                agent_bytes
            )
        )

        customer_array = (
            read_reference_wav(
                customer_bytes
            )
        )


        agent_text = (
            strip_leading_tag(
                agent_reference_text,
                "S1",
            )
        )


        customer_text = (
            strip_leading_tag(
                customer_reference_text,
                "S2",
            )
        )


        request_id = str(
            uuid.uuid4()
        )


        return StreamingResponse(

            production_generator(

                normalize_tags(text),

                agent_text,

                customer_text,

                agent_array,

                customer_array,

                max_new_tokens,

                request_id,
            ),

            media_type=
                "application/x-dia-speaker-stream",
        )


    except Exception as exc:

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


        raise HTTPException(

            status_code=500,

            detail=(
                f"TTS failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


# =============================================================================
# /tts_seed
# =============================================================================

@app.post("/tts_seed")
def tts_seed(
    req: SeedTTSRequest,
):

    set_seed(
        req.seed
    )


    try:

        inputs = processor(

            text=[
                normalize_tags(
                    req.text
                )
            ],

            padding=True,

            return_tensors="pt",

        ).to(
            model.device
        )


        with torch.inference_mode():

            outputs = model.generate(

                **inputs,

                max_new_tokens=
                    req.max_new_tokens,

                guidance_scale=
                    SEED_GUIDANCE_SCALE,

                temperature=
                    SEED_TEMPERATURE,

                top_p=
                    SEED_TOP_P,

                top_k=
                    SEED_TOP_K,
            )


        audio = decode_plain(
            outputs
        )


        buffer = io.BytesIO()


        sf.write(
            buffer,
            audio,
            SAMPLE_RATE,
            format="WAV",
            subtype="PCM_16",
        )


        return Response(

            content=
                buffer.getvalue(),

            media_type=
                "audio/wav",
        )


    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                f"Seed TTS failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
