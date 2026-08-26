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


# =============================================================================
# CHUNK CONFIG
#
# IMPORTANT:
#
# Chunk boundaries are based on TOTAL WORD COUNT,
# NOT speaker boundaries.
#
# S1/S2 tags are preserved inside every chunk.
# =============================================================================

DEFAULT_CHUNK_WORDS = int(
    os.getenv(
        "DEFAULT_CHUNK_WORDS",
        "30",
    )
)

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
    title="Dia Chunked Dual-Reference TTS",
    version="2.0.0",
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
# CUDA / RANDOM
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
# TEXT NORMALIZATION
# =============================================================================

def normalize_tags(text: str) -> str:

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

    text = normalize_tags(text)

    text = re.sub(
        rf"^\s*\[{tag}\]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


# =============================================================================
# PARSE ORIGINAL TRANSCRIPT
#
# External meaning:
#
# [S1] = agent
# [S2] = customer
#
# Works with newlines OR compact text:
#
# [S1] hello[S2] hi[S1] next...
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
            "Transcript must contain "
            "[S1] and/or [S2] tags."
        )

    turns = []

    for index, match in enumerate(matches):

        speaker = match.group(1).upper()

        start = match.end()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(text)

        speech = text[start:end].strip()

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
# WORD-BASED CHUNK BUILDER
#
# THIS IS THE IMPORTANT CHANGE.
#
# Example with chunk_words=20:
#
# INPUT:
#
# [S1] Hello thank you for calling...
# [S2] Hello I need help...
# [S1] I can help...
#
# OUTPUT:
#
# chunk 1:
# [S1] ... [S2] ...
#
# chunk 2:
# [S2] ... [S1] ...
#
#
# Speaker switch DOES NOT end the chunk.
#
# If a speaker turn crosses the word limit,
# that turn continues in next chunk and its speaker tag
# is automatically re-added.
# =============================================================================

def build_word_chunks(
    text: str,
    chunk_words: int,
):

    if chunk_words < 1:

        raise ValueError(
            "chunk_words must be >= 1"
        )

    turns = parse_turns(text)

    chunks = []

    current_parts = []

    current_word_count = 0

    for speaker, speech in turns:

        words = speech.split()

        position = 0

        while position < len(words):

            remaining_capacity = (
                chunk_words
                - current_word_count
            )

            # Current chunk full.
            if remaining_capacity <= 0:

                chunks.append(
                    " ".join(
                        current_parts
                    ).strip()
                )

                current_parts = []

                current_word_count = 0

                remaining_capacity = (
                    chunk_words
                )

            take_count = min(
                remaining_capacity,
                len(words) - position,
            )

            selected_words = words[
                position:
                position + take_count
            ]

            selected_text = " ".join(
                selected_words
            )

            # Every segment explicitly carries the correct speaker.
            current_parts.append(
                f"[{speaker}] {selected_text}"
            )

            current_word_count += (
                take_count
            )

            position += (
                take_count
            )

            # Exactly hit limit.
            if (
                current_word_count
                >= chunk_words
            ):

                chunks.append(
                    " ".join(
                        current_parts
                    ).strip()
                )

                current_parts = []

                current_word_count = 0


    if current_parts:

        chunks.append(
            " ".join(
                current_parts
            ).strip()
        )


    return chunks


# =============================================================================
# TOKEN ESTIMATION
#
# Prevents early EOS / missing final words.
# =============================================================================

def get_generation_token_limits(
    chunk_text: str,
    requested_max_new_tokens: int,
):

    # Do not count [S1]/[S2] as words.
    clean_text = re.sub(
        r"\[(S1|S2)\]",
        "",
        chunk_text,
        flags=re.IGNORECASE,
    )

    word_count = max(
        1,
        len(
            clean_text.split()
        ),
    )


    # Very short chunk.
    if word_count <= 2:

        min_new_tokens = 160

        local_max_new_tokens = min(
            requested_max_new_tokens,
            640,
        )


    elif word_count <= 5:

        min_new_tokens = 192

        local_max_new_tokens = min(
            requested_max_new_tokens,
            768,
        )


    elif word_count <= 10:

        min_new_tokens = 256

        local_max_new_tokens = min(
            requested_max_new_tokens,
            1024,
        )


    elif word_count <= 20:

        min_new_tokens = 384

        local_max_new_tokens = min(
            requested_max_new_tokens,
            1536,
        )


    elif word_count <= 30:

        min_new_tokens = 512

        local_max_new_tokens = min(
            requested_max_new_tokens,
            2048,
        )


    else:

        estimated_seconds = (
            word_count
            / 2.8
        )

        estimated_tokens = int(
            estimated_seconds
            * 128
            * 0.70
        )

        min_new_tokens = max(
            512,
            estimated_tokens,
        )

        min_new_tokens = min(
            min_new_tokens,
            1536,
        )

        local_max_new_tokens = (
            requested_max_new_tokens
        )


    if (
        min_new_tokens
        >= local_max_new_tokens
    ):

        min_new_tokens = max(
            1,
            local_max_new_tokens
            - 32,
        )


    return (
        min_new_tokens,
        local_max_new_tokens,
    )


# =============================================================================
# MIN REASONABLE AUDIO DURATION
#
# Used only to detect obviously truncated generation.
# =============================================================================

def get_min_reasonable_duration(
    chunk_text: str,
):

    clean_text = re.sub(
        r"\[(S1|S2)\]",
        "",
        chunk_text,
        flags=re.IGNORECASE,
    )

    word_count = max(
        1,
        len(
            clean_text.split()
        ),
    )

    # Very conservative minimum.
    return max(
        0.50,
        word_count / 5.0,
    )


# =============================================================================
# AUDIO HELPERS
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


def combine_two_references(
    agent_audio: np.ndarray,
    customer_audio: np.ndarray,
):

    agent_audio = (
        trim_outer_silence(
            agent_audio
        )
    )

    customer_audio = (
        trim_outer_silence(
            customer_audio
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
            agent_audio,
            silence,
            customer_audio,
        ]
    )


    return combined.astype(
        np.float32
    )


# =============================================================================
# DECODE
# =============================================================================

def decode_conditioned_audio(
    outputs,
    prompt_len,
):

    decoded = processor.batch_decode(

        outputs,

        audio_prompt_len=
            prompt_len,
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
            "Generated audio is empty."
        )


    if not np.all(
        np.isfinite(audio)
    ):

        raise RuntimeError(
            "Generated audio contains NaN/Inf."
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
            "Generated audio is empty."
        )


    return audio


# =============================================================================
# STREAM FRAME
# =============================================================================

def make_frame(
    metadata: dict,
    pcm_bytes: bytes = b"",
):

    metadata_bytes = (

        json.dumps(
            metadata,
            ensure_ascii=False,
        )

        .encode(
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
    print("DIA CHUNKED TTS STARTUP")
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
        f"Default chunk size: "
        f"{DEFAULT_CHUNK_WORDS} words"
    )

    print(
        "Chunk mode        : "
        "word-count based"
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

@app.get(
    "/health"
)
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
            "word_count",

        "default_chunk_words":
            DEFAULT_CHUNK_WORDS,

        "production_max_new_tokens":
            DEFAULT_PRODUCTION_MAX_NEW_TOKENS,

        "seed_max_new_tokens":
            DEFAULT_SEED_MAX_NEW_TOKENS,

        "endpoints": {

            "/tts":
                "word-chunk dual-reference TTS",

            "/tts_seed":
                "seed audition",
        },
    }


# =============================================================================
# GENERATE ONE CHUNK
# =============================================================================

def generate_one_chunk(

    conditioned_text,

    reference_audio,

    chunk_text,

    requested_max_new_tokens,
):


    (
        min_new_tokens,
        local_max_new_tokens,

    ) = get_generation_token_limits(

        chunk_text,

        requested_max_new_tokens,
    )


    min_duration = (
        get_min_reasonable_duration(
            chunk_text
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

        return_tensors=
            "pt",

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


    # =========================================================================
    # FIRST GENERATION
    # =========================================================================

    inference_start = (
        time.perf_counter_ns()
    )


    with torch.inference_mode():

        outputs = model.generate(

            **inputs,

            min_new_tokens=
                min_new_tokens,

            max_new_tokens=
                local_max_new_tokens,

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


    audio = decode_conditioned_audio(

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
    # EARLY-EOS RETRY
    # =========================================================================

    if duration < min_duration:

        retried = True


        print(
            "[WARN] Audio seems too short. "
            "Retrying chunk once."
        )


        stronger_min = min(

            local_max_new_tokens
            - 16,

            max(

                min_new_tokens + 96,

                int(
                    min_new_tokens
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
                    local_max_new_tokens,

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


        retry_audio = (
            decode_conditioned_audio(

                retry_outputs,

                prompt_len,
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


        if retry_duration > duration:

            audio = retry_audio

            duration = retry_duration


        inference_ms += (
            retry_inference_ms
        )


        decode_ms += (
            retry_decode_ms
        )


        del retry_outputs
        del retry_audio


    del outputs
    del inputs


    return (

        audio,

        preprocess_ms,

        inference_ms,

        decode_ms,

        min_new_tokens,

        local_max_new_tokens,

        retried,
    )


# =============================================================================
# PRODUCTION GENERATOR
#
# BOTH S1/S2 CAN EXIST INSIDE SAME CHUNK.
#
# Reference mapping:
#
# S1 = agent
# S2 = customer
#
# =============================================================================

def generate_production_chunks(

    full_text: str,

    agent_reference_text: str,

    customer_reference_text: str,

    agent_audio: np.ndarray,

    customer_audio: np.ndarray,

    chunk_words: int,

    max_new_tokens: int,

    request_id: str,

) -> Iterator[bytes]:


    server_start = (
        time.perf_counter_ns()
    )


    # =========================================================================
    # Build word-based chunks.
    # =========================================================================

    chunks = build_word_chunks(

        full_text,

        chunk_words,
    )


    # =========================================================================
    # ONE fixed two-speaker reference for every chunk.
    #
    # S1 = agent
    # S2 = customer
    # =========================================================================

    reference_audio = (
        combine_two_references(

            agent_audio,

            customer_audio,
        )
    )


    reference_text = (

        f"[S1] "
        f"{agent_reference_text} "

        f"[S2] "
        f"{customer_reference_text}"
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
        f"Chunk size        : "
        f"{chunk_words} words"
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


    for (
        chunk_index,
        chunk_text,

    ) in enumerate(
        chunks,
        start=1,
    ):


        clean_words = re.sub(

            r"\[(S1|S2)\]",

            "",

            chunk_text,

            flags=re.IGNORECASE,

        ).split()


        chunk_word_count = (
            len(clean_words)
        )


        conditioned_text = (

            f"{reference_text} "
            f"{chunk_text}"
        )


        print("")
        print("-" * 80)


        print(
            f"Chunk "
            f"{chunk_index}/"
            f"{len(chunks)}"
        )


        print(
            f"Words             : "
            f"{chunk_word_count}"
        )


        print(
            f"Text              : "
            f"{chunk_text}"
        )


        (
            audio,

            preprocess_ms,

            inference_ms,

            decode_ms,

            min_tokens,

            local_max_tokens,

            retried,

        ) = generate_one_chunk(

            conditioned_text=
                conditioned_text,

            reference_audio=
                reference_audio,

            chunk_text=
                chunk_text,

            requested_max_new_tokens=
                max_new_tokens,
        )


        total_preprocess_ms += (
            preprocess_ms
        )

        total_inference_ms += (
            inference_ms
        )

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
            f"Min tokens        : "
            f"{min_tokens}"
        )


        print(
            f"Local max tokens  : "
            f"{local_max_tokens}"
        )


        print(
            f"Inference         : "
            f"{inference_ms:.2f} ms"
        )


        print(
            f"Audio duration    : "
            f"{duration:.2f}s"
        )


        print(
            f"Retry             : "
            f"{retried}"
        )


        yield make_frame(

            {

                "type":
                    "audio",

                "request_id":
                    request_id,

                "chunk_index":
                    chunk_index,

                "chunk_count":
                    len(chunks),

                "chunk_words":
                    chunk_word_count,

                "chunk_text":
                    chunk_text,

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

                "min_new_tokens":
                    min_tokens,

                "local_max_new_tokens":
                    local_max_tokens,

                "retried":
                    retried,
            },

            pcm.tobytes(),
        )


        del audio
        del pcm


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

            "chunk_words":
                chunk_words,

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

    chunk_words: int = Form(
        DEFAULT_CHUNK_WORDS
    ),

    max_new_tokens: int = Form(
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

            detail=
                "Model is not loaded yet.",
        )


    if not (
        5
        <= chunk_words
        <= 100
    ):

        raise HTTPException(

            status_code=400,

            detail=(
                "chunk_words must be "
                "between 5 and 100."
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
                "Agent reference text is empty."
            )


        if not customer_text:

            raise ValueError(
                "Customer reference text is empty."
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
                "Agent WAV is empty."
            )


        if not customer_bytes:

            raise ValueError(
                "Customer WAV is empty."
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
            f"[reference] Agent    : "
            f"{agent_audio.filename}"
        )

        print(
            f"[reference] Customer : "
            f"{customer_audio.filename}"
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

                chunk_words=
                    chunk_words,

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
                    str(SAMPLE_RATE),

                "X-Chunk-Mode":
                    "word-count",

                "X-Chunk-Words":
                    str(chunk_words),
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
# SEED /tts_seed
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

            detail=
                "Model is not loaded yet.",
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


        preprocess_start = (
            time.perf_counter_ns()
        )


        inputs = processor(

            text=[
                text
            ],

            padding=True,

            return_tensors=
                "pt",

        ).to(
            model.device
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


        decode_start = (
            time.perf_counter_ns()
        )


        audio = decode_plain_audio(
            outputs
        )


        decode_ms = (

            time.perf_counter_ns()
            - decode_start

        ) / 1_000_000


        duration = (
            len(audio)
            / SAMPLE_RATE
        )


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


        return Response(

            content=
                wav_bytes,

            media_type=
                "audio/wav",

            headers={

                "X-Request-ID":
                    request_id,

                "X-Seed":
                    str(req.seed),

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
                    str(SAMPLE_RATE),
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
