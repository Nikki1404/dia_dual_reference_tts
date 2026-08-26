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
from typing import Iterator

import numpy as np
import soundfile as sf
import torch

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from fastapi.responses import (
    Response,
    StreamingResponse,
)

from pydantic import (
    BaseModel,
    Field,
)

from scipy.signal import resample_poly

from transformers import (
    AutoProcessor,
    DiaForConditionalGeneration,
)


# =============================================================================
# CONFIG
# =============================================================================

MODEL_ID = os.getenv(
    "MODEL_ID",
    "nari-labs/Dia-1.6B-0626",
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

DTYPE = (
    torch.float16
    if DEVICE == "cuda"
    else torch.float32
)

SAMPLE_RATE = 44100


# =============================================================================
# PRODUCTION CONFIG
#
# IMPORTANT:
#
# Production chunking is now SPEAKER-TURN based.
#
# Example:
#
# [S1] complete agent turn
# [S2] complete customer turn
# [S1] complete agent turn
#
# becomes exactly 3 chunks.
#
# There is NO sentence splitting.
# There is NO word-count splitting.
# =============================================================================

DEFAULT_PRODUCTION_MAX_NEW_TOKENS = int(
    os.getenv(
        "DEFAULT_PRODUCTION_MAX_NEW_TOKENS",
        "3072",
    )
)

DEFAULT_SEED_MAX_NEW_TOKENS = int(
    os.getenv(
        "DEFAULT_SEED_MAX_NEW_TOKENS",
        "1024",
    )
)

REFERENCE_SILENCE_MS = int(
    os.getenv(
        "REFERENCE_SILENCE_MS",
        "200",
    )
)


# =============================================================================
# PRODUCTION GENERATION SETTINGS
# =============================================================================

PRODUCTION_GUIDANCE_SCALE = float(
    os.getenv(
        "PRODUCTION_GUIDANCE_SCALE",
        "3.0",
    )
)

PRODUCTION_TEMPERATURE = float(
    os.getenv(
        "PRODUCTION_TEMPERATURE",
        "1.0",
    )
)

PRODUCTION_TOP_P = float(
    os.getenv(
        "PRODUCTION_TOP_P",
        "0.95",
    )
)

PRODUCTION_TOP_K = int(
    os.getenv(
        "PRODUCTION_TOP_K",
        "50",
    )
)


# =============================================================================
# SEED GENERATION SETTINGS
# =============================================================================

SEED_GUIDANCE_SCALE = float(
    os.getenv(
        "SEED_GUIDANCE_SCALE",
        "3.0",
    )
)

SEED_TEMPERATURE = float(
    os.getenv(
        "SEED_TEMPERATURE",
        "1.8",
    )
)

SEED_TOP_P = float(
    os.getenv(
        "SEED_TOP_P",
        "0.90",
    )
)

SEED_TOP_K = int(
    os.getenv(
        "SEED_TOP_K",
        "45",
    )
)


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="Dia Final Dual-Reference TTS",
    version="1.1.0",
)

processor = None
model = None
model_load_ms = None


# =============================================================================
# SEED REQUEST
# =============================================================================

class SeedTTSRequest(
    BaseModel
):

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
        default=
            DEFAULT_SEED_MAX_NEW_TOKENS,
        ge=256,
        le=4096,
    )


# =============================================================================
# CUDA / RANDOM
# =============================================================================

def sync_cuda():

    if torch.cuda.is_available():

        torch.cuda.synchronize()


def set_seed(
    seed: int,
):

    random.seed(
        seed
    )

    np.random.seed(
        seed % (2**32 - 1)
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )


# =============================================================================
# TEXT HELPERS
# =============================================================================

def normalize_tags(
    text: str,
) -> str:

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
) -> str:

    text = normalize_tags(
        text
    )

    text = re.sub(
        rf"^\s*\[{tag}\]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


# =============================================================================
# PARSE SPEAKER TURNS
#
# This works with:
#
# [S1] hello
# [S2] hi
#
# and also:
#
# [S1] hello[S2] hi[S1] next...
#
# Newline is NOT required.
# =============================================================================

def parse_turns(
    text: str,
):

    text = normalize_tags(
        text
    )

    matches = list(
        re.finditer(
            r"\[(S1|S2)\]\s*",
            text,
            flags=re.IGNORECASE,
        )
    )

    if not matches:

        raise ValueError(
            "Transcript must contain "
            "[S1] and/or [S2] tags."
        )

    turns = []

    for index, match in enumerate(
        matches
    ):

        speaker = (
            match.group(1).upper()
        )

        start = match.end()

        if (
            index + 1
            < len(matches)
        ):

            end = (
                matches[
                    index + 1
                ].start()
            )

        else:

            end = len(text)

        speech = (
            text[
                start:end
            ].strip()
        )

        if speech:

            turns.append(
                (
                    speaker,
                    speech,
                )
            )

    if not turns:

        raise ValueError(
            "No speech found after "
            "[S1]/[S2] tags."
        )

    return turns


# =============================================================================
# PRODUCTION CHUNKING
#
# CRITICAL CHANGE:
#
# One COMPLETE speaker turn = one chunk.
#
# No sentence splitting.
# No word-count splitting.
#
# Example:
#
# [S1] Hello. Thank you for calling. Would you prefer English or Spanish?
# [S2] English please.
# [S1] Thank you. How can I help you today?
#
# ->
#
# chunk 1 = complete S1 turn
# chunk 2 = complete S2 turn
# chunk 3 = complete S1 turn
# =============================================================================

def build_chunks(
    text: str,
):

    return parse_turns(
        text
    )


# =============================================================================
# REFERENCE AUDIO
# =============================================================================

def read_reference_wav(
    wav_bytes: bytes,
) -> np.ndarray:

    audio, sample_rate = sf.read(
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

    if (
        sample_rate
        != SAMPLE_RATE
    ):

        divisor = gcd(
            sample_rate,
            SAMPLE_RATE,
        )

        audio = resample_poly(
            audio,
            SAMPLE_RATE // divisor,
            sample_rate // divisor,
        ).astype(
            np.float32
        )

    return audio


def trim_outer_silence(
    audio: np.ndarray,
    threshold_ratio: float = 0.02,
    padding_ms: int = 40,
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


def combine_two_references(
    first_audio: np.ndarray,
    second_audio: np.ndarray,
) -> np.ndarray:

    first_audio = (
        trim_outer_silence(
            first_audio
        )
    )

    second_audio = (
        trim_outer_silence(
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

    combined = np.concatenate(
        [
            first_audio,
            silence,
            second_audio,
        ]
    )

    return combined.astype(
        np.float32
    )


# =============================================================================
# AUDIO DECODING
# =============================================================================

def decode_conditioned_audio(
    outputs,
    prompt_len,
):

    decoded = (
        processor.batch_decode(
            outputs,
            audio_prompt_len=
                prompt_len,
        )
    )

    audio = (
        decoded[0]
        if isinstance(
            decoded,
            (
                list,
                tuple,
            ),
        )
        else decoded
    )

    if torch.is_tensor(
        audio
    ):

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
            f"Unexpected audio shape: "
            f"{audio.shape}"
        )

    if len(audio) == 0:

        raise RuntimeError(
            "Dia returned empty audio."
        )

    if not np.all(
        np.isfinite(
            audio
        )
    ):

        raise RuntimeError(
            "Generated audio contains "
            "NaN/Inf."
        )

    return audio


def decode_plain_audio(
    outputs,
):

    decoded = (
        processor.batch_decode(
            outputs
        )
    )

    audio = (
        decoded[0]
        if isinstance(
            decoded,
            (
                list,
                tuple,
            ),
        )
        else decoded
    )

    if torch.is_tensor(
        audio
    ):

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
            f"Unexpected audio shape: "
            f"{audio.shape}"
        )

    if len(audio) == 0:

        raise RuntimeError(
            "Dia returned empty audio."
        )

    return audio


# =============================================================================
# STREAM FRAME
# =============================================================================

def make_frame(
    metadata: dict,
    pcm_bytes: bytes = b"",
) -> bytes:

    metadata_bytes = (
        json.dumps(
            metadata,
            ensure_ascii=False,
        ).encode(
            "utf-8"
        )
    )

    return (
        struct.pack(
            ">I",
            len(
                metadata_bytes
            ),
        )
        +
        metadata_bytes
        +
        struct.pack(
            ">Q",
            len(
                pcm_bytes
            ),
        )
        +
        pcm_bytes
    )


# =============================================================================
# STARTUP
# =============================================================================

@app.on_event(
    "startup"
)
def load_model():

    global processor
    global model
    global model_load_ms

    print("")
    print("=" * 80)
    print("DIA FINAL TTS STARTUP")
    print("=" * 80)

    print(
        f"Model             : "
        f"{MODEL_ID}"
    )

    print(
        f"Device            : "
        f"{DEVICE}"
    )

    print(
        f"PyTorch           : "
        f"{torch.__version__}"
    )

    print(
        f"PyTorch CUDA      : "
        f"{torch.version.cuda}"
    )

    print(
        f"CUDA available    : "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU               : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"DTYPE             : "
        f"{DTYPE}"
    )

    print(
        "Chunk mode        : "
        "complete speaker turn"
    )

    print("=" * 80)

    started = (
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
        - started
    ) / 1_000_000

    print(
        f"Model loaded      : "
        f"{model_load_ms:.2f} ms"
    )

    if torch.cuda.is_available():

        print(
            f"GPU allocated     : "
            f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
        )

    print("=" * 80)


# =============================================================================
# HEALTH
# =============================================================================

@app.get(
    "/health"
)
def health():

    result = {

        "status":
            "ok",

        "model":
            MODEL_ID,

        "device":
            DEVICE,

        "dtype":
            str(
                DTYPE
            ),

        "cuda_available":
            torch.cuda.is_available(),

        "sample_rate":
            SAMPLE_RATE,

        "chunk_mode":
            "complete_speaker_turn",

        "production_max_new_tokens":
            DEFAULT_PRODUCTION_MAX_NEW_TOKENS,

        "seed_max_new_tokens":
            DEFAULT_SEED_MAX_NEW_TOKENS,

        "endpoints": {

            "/tts":
                "dual-reference production TTS",

            "/tts_seed":
                "native Dia seed auditioning",
        },
    }

    if torch.cuda.is_available():

        result["gpu"] = (
            torch.cuda.get_device_name(0)
        )

    return result


# =============================================================================
# PRODUCTION GENERATOR
# =============================================================================

def generate_production_chunks(
    full_text: str,
    agent_reference_text: str,
    customer_reference_text: str,
    agent_audio: np.ndarray,
    customer_audio: np.ndarray,
    max_new_tokens: int,
    request_id: str,
) -> Iterator[bytes]:

    server_start = (
        time.perf_counter_ns()
    )

    # =========================================================================
    # One complete [S1]/[S2] turn per chunk.
    # =========================================================================

    chunks = build_chunks(
        full_text
    )

    # =========================================================================
    # Two prompt orders.
    #
    # For agent generation:
    #   internal S1 = agent
    #
    # For customer generation:
    #   internal S1 = customer
    #
    # This lets us keep external:
    #
    #   S1 = agent
    #   S2 = customer
    #
    # while making the target speaker internal S1.
    # =========================================================================

    agent_first_audio = (
        combine_two_references(
            agent_audio,
            customer_audio,
        )
    )

    customer_first_audio = (
        combine_two_references(
            customer_audio,
            agent_audio,
        )
    )

    agent_prompt = (
        f"[S1] "
        f"{agent_reference_text} "
        f"[S2] "
        f"{customer_reference_text}"
    )

    customer_prompt = (
        f"[S1] "
        f"{customer_reference_text} "
        f"[S2] "
        f"{agent_reference_text}"
    )

    print("")
    print("=" * 80)
    print("PRODUCTION TTS REQUEST")
    print("=" * 80)

    print(
        f"Request ID        : "
        f"{request_id}"
    )

    print(
        f"Chunks            : "
        f"{len(chunks)}"
    )

    print(
        "Chunk mode        : "
        "complete speaker turn"
    )

    print(
        f"Max new tokens    : "
        f"{max_new_tokens}"
    )

    print("=" * 80)

    total_preprocess_ms = 0.0
    total_inference_ms = 0.0
    total_decode_ms = 0.0
    total_samples = 0

    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()

    # =========================================================================
    # GENERATE EACH COMPLETE SPEAKER TURN
    # =========================================================================

    for (
        chunk_index,
        (
            speaker,
            speech,
        ),
    ) in enumerate(
        chunks,
        start=1,
    ):

        if speaker == "S1":

            reference_audio = (
                agent_first_audio
            )

            reference_text = (
                agent_prompt
            )

            voice_name = (
                "agent"
            )

        else:

            reference_audio = (
                customer_first_audio
            )

            reference_text = (
                customer_prompt
            )

            voice_name = (
                "customer"
            )

        # =========================================================================
        # Target speaker is internally S1.
        # =========================================================================

        target_text = (
            f"[S1] "
            f"{speech}"
        )

        conditioned_text = (
            f"{reference_text} "
            f"{target_text}"
        )

        chunk_words = len(
            speech.split()
        )

        print("")
        print("-" * 80)

        print(
            f"Chunk "
            f"{chunk_index}/"
            f"{len(chunks)}"
        )

        print(
            f"External speaker  : "
            f"{speaker} "
            f"({voice_name})"
        )

        print(
            f"Words             : "
            f"{chunk_words}"
        )

        print(
            f"Text              : "
            f"{speech}"
        )

        # =====================================================================
        # PREPROCESS
        # =====================================================================

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

        total_preprocess_ms += (
            preprocess_ms
        )

        # =====================================================================
        # INFERENCE
        # =====================================================================

        inference_start = (
            time.perf_counter_ns()
        )

        with torch.inference_mode():

            outputs = model.generate(
                **inputs,

                max_new_tokens=
                    max_new_tokens,

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

        total_inference_ms += (
            inference_ms
        )

        # =====================================================================
        # DECODE
        # =====================================================================

        decode_start = (
            time.perf_counter_ns()
        )

        audio = (
            decode_conditioned_audio(
                outputs,
                prompt_len,
            )
        )

        decode_ms = (
            time.perf_counter_ns()
            - decode_start
        ) / 1_000_000

        total_decode_ms += (
            decode_ms
        )

        total_samples += (
            len(audio)
        )

        duration = (
            len(audio)
            / SAMPLE_RATE
        )

        # =====================================================================
        # PCM16
        # =====================================================================

        pcm = (
            np.clip(
                audio,
                -1.0,
                1.0,
            )
            * 32767.0
        ).astype(
            "<i2"
        )

        print(
            f"Inference         : "
            f"{inference_ms:.2f} ms"
        )

        print(
            f"Audio duration    : "
            f"{duration:.2f}s"
        )

        # =====================================================================
        # STREAM FRAME
        # =====================================================================

        frame = make_frame(
            {
                "type":
                    "audio",

                "request_id":
                    request_id,

                "chunk_index":
                    chunk_index,

                "chunk_count":
                    len(chunks),

                "speaker":
                    speaker,

                "voice":
                    voice_name,

                "chunk_words":
                    chunk_words,

                "chunk_text":
                    speech,

                "sample_rate":
                    SAMPLE_RATE,

                "channels":
                    1,

                "audio_duration_s":
                    duration,

                "preprocess_ms":
                    preprocess_ms,

                "inference_ms":
                    inference_ms,

                "decode_ms":
                    decode_ms,
            },

            pcm.tobytes(),
        )

        del inputs
        del outputs
        del audio
        del pcm

        yield frame


    # =========================================================================
    # FINAL METRICS
    # =========================================================================

    server_total_ms = (
        time.perf_counter_ns()
        - server_start
    ) / 1_000_000

    audio_duration_s = (
        total_samples
        / SAMPLE_RATE
    )

    generation_rtf = (
        (
            total_inference_ms
            / 1000.0
        )
        / audio_duration_s

        if audio_duration_s > 0

        else 0.0
    )

    total_rtf = (
        (
            server_total_ms
            / 1000.0
        )
        / audio_duration_s

        if audio_duration_s > 0

        else 0.0
    )

    yield make_frame(
        {
            "type":
                "end",

            "request_id":
                request_id,

            "chunk_count":
                len(chunks),

            "preprocess_ms":
                total_preprocess_ms,

            "inference_ms":
                total_inference_ms,

            "decode_ms":
                total_decode_ms,

            "server_total_ms":
                server_total_ms,

            "audio_duration_s":
                audio_duration_s,

            "generation_rtf":
                generation_rtf,

            "total_rtf":
                total_rtf,

            "gpu_peak_mb":
                (
                    torch.cuda
                    .max_memory_allocated()
                    / 1024**2

                    if torch.cuda.is_available()

                    else 0.0
                ),
        }
    )


# =============================================================================
# PRODUCTION /tts
# =============================================================================

@app.post(
    "/tts"
)
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

    if (
        model is None
        or processor is None
    ):

        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded yet."
            ),
        )

    if not (
        256
        <= max_new_tokens
        <= 4096
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "max_new_tokens must be "
                "between 256 and 4096."
            ),
        )

    request_id = str(
        uuid.uuid4()
    )

    try:

        normalized_text = (
            normalize_tags(
                text
            )
        )

        # Validate transcript.
        parse_turns(
            normalized_text
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

        if not agent_text:

            raise ValueError(
                "Agent reference text "
                "is empty."
            )

        if not customer_text:

            raise ValueError(
                "Customer reference text "
                "is empty."
            )

        agent_bytes = (
            agent_audio.file.read()
        )

        customer_bytes = (
            customer_audio.file.read()
        )

        if not agent_bytes:

            raise ValueError(
                "Agent reference WAV "
                "is empty."
            )

        if not customer_bytes:

            raise ValueError(
                "Customer reference WAV "
                "is empty."
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

        print("")
        print(
            f"[reference] Agent file       : "
            f"{agent_audio.filename}"
        )

        print(
            f"[reference] Customer file    : "
            f"{customer_audio.filename}"
        )

        print(
            f"[reference] Agent duration   : "
            f"{len(agent_array) / SAMPLE_RATE:.2f}s"
        )

        print(
            f"[reference] Customer duration: "
            f"{len(customer_array) / SAMPLE_RATE:.2f}s"
        )

        return StreamingResponse(
            generate_production_chunks(
                full_text=
                    normalized_text,

                agent_reference_text=
                    agent_text,

                customer_reference_text=
                    customer_text,

                agent_audio=
                    agent_array,

                customer_audio=
                    customer_array,

                max_new_tokens=
                    max_new_tokens,

                request_id=
                    request_id,
            ),

            media_type=(
                "application/"
                "x-dia-production-stream"
            ),

            headers={
                "X-Request-ID":
                    request_id,

                "X-Sample-Rate":
                    str(
                        SAMPLE_RATE
                    ),

                "X-Chunk-Mode":
                    "complete-speaker-turn",
            },
        )

    except Exception as exc:

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        print(
            f"[ERROR] "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Production TTS failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        ) from exc


# =============================================================================
# SEED AUDITION /tts_seed
# =============================================================================

@app.post(
    "/tts_seed"
)
def tts_seed(
    req: SeedTTSRequest,
):

    if (
        model is None
        or processor is None
    ):

        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded yet."
            ),
        )

    request_id = str(
        uuid.uuid4()
    )

    server_start = (
        time.perf_counter_ns()
    )

    try:

        text = normalize_tags(
            req.text
        )

        set_seed(
            req.seed
        )

        print("")
        print("=" * 80)
        print("SEED AUDITION")
        print("=" * 80)

        print(
            f"Seed              : "
            f"{req.seed}"
        )

        print(
            f"Text              : "
            f"{text}"
        )

        print("=" * 80)

        # =====================================================================
        # PREPROCESS
        # =====================================================================

        preprocess_start = (
            time.perf_counter_ns()
        )

        inputs = processor(
            text=[
                text
            ],
            padding=True,
            return_tensors="pt",
        ).to(
            model.device
        )

        sync_cuda()

        preprocess_ms = (
            time.perf_counter_ns()
            - preprocess_start
        ) / 1_000_000

        # =====================================================================
        # INFERENCE
        # =====================================================================

        inference_start = (
            time.perf_counter_ns()
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

        sync_cuda()

        inference_ms = (
            time.perf_counter_ns()
            - inference_start
        ) / 1_000_000

        # =====================================================================
        # DECODE
        # =====================================================================

        decode_start = (
            time.perf_counter_ns()
        )

        audio = (
            decode_plain_audio(
                outputs
            )
        )

        decode_ms = (
            time.perf_counter_ns()
            - decode_start
        ) / 1_000_000

        duration = (
            len(audio)
            / SAMPLE_RATE
        )

        # =====================================================================
        # WAV
        # =====================================================================

        encode_start = (
            time.perf_counter_ns()
        )

        buffer = io.BytesIO()

        sf.write(
            buffer,
            audio,
            SAMPLE_RATE,
            format="WAV",
            subtype="PCM_16",
        )

        wav_bytes = (
            buffer.getvalue()
        )

        encode_ms = (
            time.perf_counter_ns()
            - encode_start
        ) / 1_000_000

        server_total_ms = (
            time.perf_counter_ns()
            - server_start
        ) / 1_000_000

        print(
            f"Inference         : "
            f"{inference_ms:.2f} ms"
        )

        print(
            f"Audio duration    : "
            f"{duration:.2f}s"
        )

        print("=" * 80)

        return Response(
            content=
                wav_bytes,

            media_type=
                "audio/wav",

            headers={
                "X-Request-ID":
                    request_id,

                "X-Seed":
                    str(
                        req.seed
                    ),

                "X-Preprocess-Time-MS":
                    f"{preprocess_ms:.2f}",

                "X-Inference-Time-MS":
                    f"{inference_ms:.2f}",

                "X-Decode-Time-MS":
                    f"{decode_ms:.2f}",

                "X-Encoding-Time-MS":
                    f"{encode_ms:.2f}",

                "X-Server-Total-MS":
                    f"{server_total_ms:.2f}",

                "X-Audio-Duration-S":
                    f"{duration:.3f}",

                "X-Sample-Rate":
                    str(
                        SAMPLE_RATE
                    ),
            },
        )

    except Exception as exc:

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        raise HTTPException(
            status_code=500,
            detail=(
                "Seed TTS failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        ) from exc
