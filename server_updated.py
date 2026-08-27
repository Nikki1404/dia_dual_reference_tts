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
        "180",
    )
)


MAX_WORDS_PER_CHUNK = int(
    os.getenv(
        "MAX_WORDS_PER_CHUNK",
        "12",
    )
)


# =============================================================================
# PRODUCTION GENERATION
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
        "0.8",
    )
)

PRODUCTION_TOP_P = float(
    os.getenv(
        "PRODUCTION_TOP_P",
        "0.90",
    )
)

PRODUCTION_TOP_K = int(
    os.getenv(
        "PRODUCTION_TOP_K",
        "50",
    )
)


# =============================================================================
# SEED MODE GENERATION
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
    title="Dia Speaker-Wise Dual Reference TTS",
    version="4.0.0",
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
# CUDA / RNG
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
# TEXT HELPERS
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
# One complete speaker turn = one generation chunk.
#
# [S1] = agent
# [S2] = customer
#
# Works with:
#
# [S1] hello.
# [S2] hi.
#
# and:
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
            "Transcript must contain [S1] and/or [S2] tags."
        )

    turns = []

    for index, match in enumerate(matches):

        speaker = (
            match.group(1)
            .upper()
        )

        start = (
            match.end()
        )

        if index + 1 < len(matches):

            end = (
                matches[
                    index + 1
                ]
                .start()
            )

        else:

            end = len(text)

        speech = (
            text[
                start:end
            ]
            .strip()
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
            "No speech found after speaker tags."
        )

    return turns


# =============================================================================
# TOKEN PROTECTION
#
# Prevents very short text from immediately ending.
#
# Examples:
#
# Hello.
# Yes.
# English please.
# My member ID is 2043.
# =============================================================================

def generation_limits(
    speech: str,
    requested_max: int,
):

    word_count = max(
        1,
        len(
            speech.split()
        ),
    )

    if word_count == 1:

        min_tokens = 128

        local_max = min(
            requested_max,
            512,
        )

    elif word_count <= 3:

        min_tokens = 160

        local_max = min(
            requested_max,
            640,
        )

    elif word_count <= 6:

        min_tokens = 192

        local_max = min(
            requested_max,
            768,
        )

    elif word_count <= 12:

        min_tokens = 256

        local_max = min(
            requested_max,
            1280,
        )

    elif word_count <= 20:

        min_tokens = 384

        local_max = min(
            requested_max,
            1792,
        )

    elif word_count <= 30:

        min_tokens = 512

        local_max = min(
            requested_max,
            2304,
        )

    else:

        estimated_seconds = (
            word_count
            / 2.8
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


# =============================================================================
# OBVIOUSLY-TOO-SHORT DETECTION
#
# This does NOT verify transcript accuracy.
# It only catches outputs that are clearly too short.
# =============================================================================

def minimum_reasonable_duration(
    speech: str,
):

    word_count = max(
        1,
        len(
            speech.split()
        ),
    )

    if word_count == 1:

        return 0.30

    if word_count <= 3:

        return 0.50

    return max(
        0.65,
        word_count / 3.6,
    )


# =============================================================================
# REFERENCE AUDIO
# =============================================================================

def read_reference_wav(
    wav_bytes: bytes,
):

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

    if sample_rate != SAMPLE_RATE:

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
        >= peak
        * threshold_ratio
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
            f"Unexpected audio shape: "
            f"{audio.shape}"
        )

    if len(audio) == 0:

        raise RuntimeError(
            "Generated audio is empty."
        )

    if not np.all(
        np.isfinite(audio)
    ):

        raise RuntimeError(
            "Generated audio contains NaN/Inf."
        )

    return audio


def decode_plain(
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
            f"Unexpected audio shape: "
            f"{audio.shape}"
        )

    return audio


# =============================================================================
# STREAM FRAME
# =============================================================================

def make_frame(
    metadata,
    pcm=b"",
):

    metadata_bytes = (
        json.dumps(
            metadata
        )
        .encode(
            "utf-8"
        )
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
    print("DIA SPEAKER-WISE TTS STARTUP")
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
        "Chunk mode        : "
        "complete speaker turn"
    )

    print(
        "Short-text guard  : enabled"
    )

    print(
        "Early-EOS retry   : enabled"
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

        "early_eos_retry":
            True,

        "endpoints": {

            "/tts":
                "production",

            "/tts_seed":
                "seed audition",
        },
    }


# =============================================================================
# GENERATED-AUDIO SILENCE TRIM
#
# Removes Dia's leading/trailing silence from each generated chunk so that
# chunks butt together seamlessly with no stacked pause.
# =============================================================================

def trim_generated_silence(
    audio,
    threshold_ratio=0.015,
    pad_ms=25,
):

    if len(audio) == 0:
        return audio

    peak = float(np.max(np.abs(audio)))

    if peak <= 0:
        return audio

    active = np.flatnonzero(
        np.abs(audio) >= peak * threshold_ratio
    )

    if len(active) == 0:
        return audio

    pad = int(SAMPLE_RATE * pad_ms / 1000)

    start = max(0, int(active[0]) - pad)
    end = min(len(audio), int(active[-1]) + pad + 1)

    return audio[start:end]


# =============================================================================
# TERMINAL PUNCTUATION
#
# A clean sentence-final cue makes Dia stop deliberately instead of rushing
# and dropping the last few words before an early EOS.
# =============================================================================

def ensure_terminal_punctuation(text):

    text = text.strip().rstrip(",;:")

    if text and text[-1] not in ".?!":
        text = text + "."

    return text


# =============================================================================
# SUB-CHUNKING
#
# A single speaker turn is split so no one generation is long enough for Dia
# to skip words. Splits at sentence boundaries first, then clause boundaries,
# then a hard word cap; small pieces are greedily packed back up to max_words.
# =============================================================================

def split_turn(speech, max_words):

    words = speech.split()

    if len(words) <= max_words:
        return [speech.strip()]

    units = []

    for sentence in re.split(r"(?<=[.!?])\s+", speech.strip()):

        s = sentence.strip()

        if not s:
            continue

        if len(s.split()) <= max_words:
            units.append(s)
            continue

        for clause in re.split(r"(?<=[,;:])\s+", s):

            c = clause.strip()

            if not c:
                continue

            cw = c.split()

            if len(cw) <= max_words:
                units.append(c)
            else:
                for i in range(0, len(cw), max_words):
                    units.append(" ".join(cw[i:i + max_words]))

    chunks, buf, bw = [], [], 0

    for u in units:

        uw = len(u.split())

        if buf and bw + uw > max_words:
            chunks.append(" ".join(buf))
            buf, bw = [], 0

        buf.append(u)
        bw += uw

    if buf:
        chunks.append(" ".join(buf))

    return chunks


# =============================================================================
# GENERATE ONE SPEAKER TURN
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

    audio = trim_generated_silence(
        audio
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
    # RETRY ON CLEARLY TOO-SHORT OUTPUT
    # =========================================================================

    if duration < min_duration:

        retried = True

        print(
            "[WARN] Output shorter than expected. "
            "Retrying once."
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

            retry_outputs = (
                model.generate(
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
            )

        sync_cuda()

        retry_inference_ms = (
            time.perf_counter_ns()
            - retry_start
        ) / 1_000_000

        retry_decode_start = (
            time.perf_counter_ns()
        )

        retry_audio = (
            decode_conditioned(
                retry_outputs,
                prompt_len,
            )
        )

        retry_audio = (
            trim_generated_silence(
                retry_audio
            )
        )

        retry_decode_ms = (
            time.perf_counter_ns()
            - retry_decode_start
        ) / 1_000_000

        retry_duration = (
            len(retry_audio)
            / SAMPLE_RATE
        )

        print(
            f"Retry duration    : "
            f"{retry_duration:.2f}s"
        )

        if retry_duration > duration:

            audio = (
                retry_audio
            )

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


    # Expand each speaker turn into bounded sub-chunks.
    expanded = []

    for turn_i, (speaker, speech) in enumerate(
        turns,
        start=1,
    ):

        parts = split_turn(
            speech,
            MAX_WORDS_PER_CHUNK,
        )

        for j, part in enumerate(parts):

            expanded.append(
                {
                    "speaker": speaker,
                    "speech": ensure_terminal_punctuation(part),
                    "turn_index": turn_i,
                    "is_continuation": j > 0,
                }
            )

    print(
        f"Sub-chunks        : "
        f"{len(expanded)}"
    )


    # =========================================================================
    # AGENT
    #
    # Internal S1 = agent
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
    # CUSTOMER
    #
    # Internal S1 = customer
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
        f"Request ID        : "
        f"{request_id}"
    )

    print(
        f"Turns             : "
        f"{len(turns)}"
    )

    print(
        f"Max new tokens    : "
        f"{max_new_tokens}"
    )

    print("=" * 80)


    total_preprocess = 0.0
    total_inference = 0.0
    total_decode = 0.0
    total_samples = 0


    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()


    for index, item in enumerate(
        expanded,
        start=1,
    ):

        speaker = item["speaker"]
        speech = item["speech"]
        turn_index = item["turn_index"]
        is_continuation = item["is_continuation"]

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


        # Target is always internal S1.
        target_text = (
            f"[S1] {speech}"
        )

        conditioned_text = (
            f"{ref_text} "
            f"{target_text}"
        )


        print("")
        print("-" * 80)

        print(
            f"Chunk "
            f"{index}/{len(expanded)}"
        )

        print(
            f"Speaker           : "
            f"{speaker} ({voice})"
        )

        print(
            f"Words             : "
            f"{len(speech.split())}"
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
            conditioned_text=
                conditioned_text,

            reference_audio=
                ref_audio,

            speech=
                speech,

            requested_max=
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


        print(
            f"Min new tokens    : "
            f"{min_tokens}"
        )

        print(
            f"Local max tokens  : "
            f"{local_max}"
        )

        print(
            f"Audio duration    : "
            f"{duration:.2f}s"
        )

        print(
            f"Retry used        : "
            f"{retried}"
        )


        yield make_frame(
            {

                "type":
                    "audio",

                "request_id":
                    request_id,

                "chunk_index":
                    index,

                "chunk_count":
                    len(expanded),

                "speaker":
                    speaker,

                "turn_index":
                    turn_index,

                "is_continuation":
                    is_continuation,

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

            "request_id":
                request_id,

            "chunk_count":
                len(expanded),

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

    if (
        model is None
        or processor is None
    ):

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet.",
        )

    if not (
        256
        <= max_new_tokens
        <= 4096
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "max_new_tokens must be between "
                "256 and 4096."
            ),
        )


    try:

        text = normalize_tags(
            text
        )

        # Validate.
        parse_turns(
            text
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


        agent_bytes = (
            agent_audio
            .file
            .read()
        )


        customer_bytes = (
            customer_audio
            .file
            .read()
        )


        if not agent_bytes:

            raise ValueError(
                "Agent reference WAV is empty."
            )


        if not customer_bytes:

            raise ValueError(
                "Customer reference WAV is empty."
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


        request_id = str(
            uuid.uuid4()
        )


        return StreamingResponse(

            production_generator(

                text=
                    text,

                agent_text=
                    agent_text,

                customer_text=
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

            media_type=
                "application/x-dia-speaker-stream",

            headers={

                "X-Request-ID":
                    request_id,

                "X-Sample-Rate":
                    str(SAMPLE_RATE),

                "X-Chunk-Mode":
                    "complete-speaker-turn",
            },
        )


    except Exception as exc:

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


        raise HTTPException(
            status_code=500,
            detail=(
                "Production TTS failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        ) from exc


# =============================================================================
# /tts_seed
# =============================================================================

@app.post("/tts_seed")
def tts_seed(
    req: SeedTTSRequest,
):

    if (
        model is None
        or processor is None
    ):

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet.",
        )


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
                "Seed TTS failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        ) from exc
