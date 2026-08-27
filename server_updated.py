#!/usr/bin/env python3
# server.py
#
# Dia reference-conditioned TTS.
#
# Same dialogue-aware BLOCK pipeline as before (this is what produced
# continuous audio), but the single reference.wav / reference.txt is now
# supplied as split dual references:
#
#   agent.wav    + agent.txt     -> [S1]
#   customer.wav + customer.txt  -> [S2]
#
# Internally they are recombined into ONE reference exactly like the old
# two-speaker seed:  [S1] agent  [S2] customer.
#
# Word-completeness changes vs the old code:
#   - temperature 1.8 -> 1.3 (Dia-recommended; 1.8 caused skips)
#   - min_new_tokens per block (blocks Dia's early-EOS tail clipping)
#   - one retry if a block comes back implausibly short

import io
import json
import os
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
from fastapi.responses import StreamingResponse
from scipy.signal import resample_poly
from transformers import (
    AutoProcessor,
    DiaForConditionalGeneration,
)


# =============================================================================
# CONFIG
# =============================================================================

MODEL_ID = "nari-labs/Dia-1.6B-0626"

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
# LONG-TEXT CONFIG
#
# Unchanged from the version that produced continuous audio.
# Keep several S1/S2 turns together; do NOT split every sentence.
# =============================================================================

TARGET_WORDS_PER_BLOCK = int(
    os.getenv("TARGET_WORDS_PER_BLOCK", "35")
)

MAX_WORDS_PER_BLOCK = int(
    os.getenv("MAX_WORDS_PER_BLOCK", "55")
)

DEFAULT_MAX_NEW_TOKENS = 3072


# =============================================================================
# GENERATION CONFIG
# =============================================================================

PRODUCTION_TEMPERATURE = float(
    os.getenv("PRODUCTION_TEMPERATURE", "1.3")
)

PRODUCTION_TOP_P = float(
    os.getenv("PRODUCTION_TOP_P", "0.95")
)

PRODUCTION_TOP_K = int(
    os.getenv("PRODUCTION_TOP_K", "45")
)

PRODUCTION_GUIDANCE_SCALE = float(
    os.getenv("PRODUCTION_GUIDANCE_SCALE", "3.0")
)

# Silence inserted between agent and customer clips in the combined reference.
REFERENCE_SILENCE_MS = int(
    os.getenv("REFERENCE_SILENCE_MS", "180")
)


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="Dia Dual-Reference Conditioned TTS",
    version="5.1.0",
)

processor = None
model = None
model_load_ms = None


# =============================================================================
# CUDA
# =============================================================================

def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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


def strip_leading_tag(text: str, tag: str) -> str:

    text = normalize_tags(text)

    text = re.sub(
        rf"^\s*\[{tag}\]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


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
            "Transcript must contain [S1] and [S2] tags."
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
                (speaker, speech)
            )

    if not turns:
        raise ValueError(
            "No dialogue found after speaker tags."
        )

    return turns


# =============================================================================
# LONG TURN SPLITTING
# =============================================================================

def split_sentences(text: str):

    return [
        value.strip()
        for value in re.split(
            r"(?<=[.!?])\s+",
            text,
        )
        if value.strip()
    ]


def split_long_turn(
    speaker: str,
    speech: str,
):

    if (
        len(speech.split())
        <= MAX_WORDS_PER_BLOCK
    ):
        return [
            (speaker, speech)
        ]

    sentences = split_sentences(
        speech
    )

    output = []

    current = []
    current_words = 0

    for sentence in sentences:

        sentence_words = len(
            sentence.split()
        )

        # One sentence itself is extremely large.
        if (
            sentence_words
            > MAX_WORDS_PER_BLOCK
        ):

            if current:

                output.append(
                    (
                        speaker,
                        " ".join(current),
                    )
                )

                current = []
                current_words = 0

            words = sentence.split()

            for i in range(
                0,
                len(words),
                MAX_WORDS_PER_BLOCK,
            ):

                output.append(
                    (
                        speaker,
                        " ".join(
                            words[
                                i:
                                i + MAX_WORDS_PER_BLOCK
                            ]
                        ),
                    )
                )

            continue

        if (
            current
            and
            current_words + sentence_words
            > MAX_WORDS_PER_BLOCK
        ):

            output.append(
                (
                    speaker,
                    " ".join(current),
                )
            )

            current = []
            current_words = 0

        current.append(sentence)
        current_words += sentence_words

    if current:

        output.append(
            (
                speaker,
                " ".join(current),
            )
        )

    return output


# =============================================================================
# DIALOGUE-AWARE BLOCKING
# =============================================================================

def build_dialogue_blocks(text: str):
    """
    1. Keep several S1/S2 turns together.
    2. Prefer starting a new block with S1.
    3. Avoid arbitrarily cutting a sentence.
    4. Avoid one enormous Dia generation.
    """

    original_turns = parse_turns(text)

    turns = []

    for speaker, speech in original_turns:

        turns.extend(
            split_long_turn(
                speaker,
                speech,
            )
        )

    blocks = []

    current = []
    current_words = 0

    for speaker, speech in turns:

        word_count = len(speech.split())

        # Best boundary: enough text already and next turn is S1.
        if (
            current
            and
            speaker == "S1"
            and
            current_words
            >= TARGET_WORDS_PER_BLOCK
        ):

            blocks.append(current)

            current = []
            current_words = 0

        # Hard safety check; still prefer breaking before S1.
        elif (
            current
            and
            speaker == "S1"
            and
            current_words + word_count
            > MAX_WORDS_PER_BLOCK
        ):

            blocks.append(current)

            current = []
            current_words = 0

        current.append(
            (speaker, speech)
        )

        current_words += word_count

    if current:

        blocks.append(current)

    formatted_blocks = []

    for block in blocks:

        formatted_blocks.append(
            " ".join(
                f"[{speaker}] {speech}"
                for speaker, speech
                in block
            )
        )

    return formatted_blocks


# =============================================================================
# REFERENCE AUDIO
# =============================================================================

def load_reference_audio(
    wav_bytes: bytes,
):

    audio, sample_rate = sf.read(
        io.BytesIO(wav_bytes),
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
            "Reference audio is empty."
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
        )

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

    duration = (
        len(audio)
        / SAMPLE_RATE
    )

    return audio, duration


def trim_edge_silence(
    audio,
    threshold_ratio=0.02,
    pad_ms=40,
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


def combine_references(
    agent_audio,
    customer_audio,
):
    """
    Rebuild the single two-speaker reference the block pipeline expects:

        [S1] agent   [S2] customer

    from the split agent.wav + customer.wav.
    """

    agent_audio = trim_edge_silence(agent_audio)
    customer_audio = trim_edge_silence(customer_audio)

    silence = np.zeros(
        int(SAMPLE_RATE * REFERENCE_SILENCE_MS / 1000),
        dtype=np.float32,
    )

    return np.concatenate(
        [
            agent_audio,
            silence,
            customer_audio,
        ]
    ).astype(np.float32)


# =============================================================================
# GENERATED AUDIO DECODING
# =============================================================================

def decode_generated_audio(
    outputs,
    prompt_len,
):

    decoded = processor.batch_decode(
        outputs,
        audio_prompt_len=prompt_len,
    )

    if isinstance(decoded, (list, tuple)):
        audio = decoded[0]
    else:
        audio = decoded

    if torch.is_tensor(audio):

        audio = (
            audio
            .detach()
            .float()
            .cpu()
            .numpy()
        )

    audio = np.asarray(audio, dtype=np.float32)

    audio = np.squeeze(audio)

    if audio.ndim != 1:

        raise RuntimeError(
            f"Unexpected decoded audio shape: {audio.shape}"
        )

    if len(audio) == 0:

        raise RuntimeError(
            "Dia returned empty generated audio."
        )

    if not np.all(np.isfinite(audio)):

        raise RuntimeError(
            "Generated audio contains NaN/Inf."
        )

    return audio


# =============================================================================
# TOKEN FLOOR
#
# Approx Dia rate ~86 audio frames/sec, ~2.8 words/sec -> ~30 tokens/word.
# We floor generation at ~12 tokens/word so Dia cannot emit an early EOS
# that cuts off the last words of a block, without forcing it too long.
# =============================================================================

def block_min_new_tokens(block_text, max_new_tokens):

    words = max(
        1,
        len(re.findall(r"[A-Za-z0-9']+", block_text)),
    )

    floor = max(128, words * 12)

    return min(max_new_tokens - 32, floor)


def block_min_reasonable_seconds(block_text):

    words = max(
        1,
        len(re.findall(r"[A-Za-z0-9']+", block_text)),
    )

    # ~6 words/sec is faster than real speech; anything shorter than this
    # is almost certainly a truncated/skipped block.
    return words / 6.0


# =============================================================================
# CUSTOM STREAM FRAMING
# =============================================================================

def make_frame(
    metadata: dict,
    pcm_bytes: bytes = b"",
):

    metadata_bytes = json.dumps(metadata).encode("utf-8")

    return (
        struct.pack(">I", len(metadata_bytes))
        + metadata_bytes
        + struct.pack(">Q", len(pcm_bytes))
        + pcm_bytes
    )


# =============================================================================
# MODEL STARTUP
# =============================================================================

@app.on_event("startup")
def load_model():

    global processor
    global model
    global model_load_ms

    print("")
    print("=" * 80)
    print("DIA DUAL-REFERENCE TTS STARTUP")
    print("=" * 80)

    print(f"Model             : {MODEL_ID}")
    print(f"Device            : {DEVICE}")
    print(f"PyTorch           : {torch.__version__}")
    print(f"PyTorch CUDA      : {torch.version.cuda}")
    print(f"CUDA available    : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU               : {torch.cuda.get_device_name(0)}")

    print(f"DTYPE             : {DTYPE}")
    print(f"Target words/block: {TARGET_WORDS_PER_BLOCK}")
    print(f"Max words/block   : {MAX_WORDS_PER_BLOCK}")
    print(f"Temperature       : {PRODUCTION_TEMPERATURE}")
    print("=" * 80)

    started = time.perf_counter_ns()

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = (
        DiaForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        )
        .to(DEVICE)
    )

    model.eval()

    sync_cuda()

    model_load_ms = (
        time.perf_counter_ns() - started
    ) / 1_000_000

    print(f"Model loaded      : {model_load_ms:.2f} ms")
    print(f"Model loaded      : {model_load_ms / 1000:.2f} sec")
    print("=" * 80)


# =============================================================================
# HEALTH
# =============================================================================

@app.get("/health")
def health():

    data = {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "model_load_ms": model_load_ms,
        "sample_rate": SAMPLE_RATE,
        "generation_mode": "dual-reference-block-conditioned",
        "speaker_mapping": {"S1": "agent", "S2": "customer"},
    }

    if torch.cuda.is_available():
        data["gpu"] = torch.cuda.get_device_name(0)

    return data


# =============================================================================
# GENERATOR
# =============================================================================

def generate_blocks(
    full_text: str,
    reference_text: str,
    reference_audio: np.ndarray,
    max_new_tokens: int,
    request_id: str,
):

    server_start = time.perf_counter_ns()

    full_text = normalize_tags(full_text)
    reference_text = normalize_tags(reference_text)

    blocks = build_dialogue_blocks(full_text)

    print("")
    print("=" * 80)
    print("DUAL-REFERENCE BLOCK REQUEST")
    print("=" * 80)
    print(f"Request ID        : {request_id}")
    print(f"Blocks            : {len(blocks)}")
    print(
        f"Reference duration: "
        f"{len(reference_audio) / SAMPLE_RATE:.2f}s"
    )
    print(f"Max new tokens    : {max_new_tokens}")
    print("=" * 80)

    total_preprocess_ms = 0.0
    total_inference_ms = 0.0
    total_decode_ms = 0.0
    total_samples = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for block_index, block_text in enumerate(blocks, start=1):

        print("")
        print("-" * 80)
        print(f"Block {block_index}/{len(blocks)}")
        print(block_text)

        # Reference transcript corresponds to reference_audio; new dialogue
        # is appended after it. (Same as the continuous-audio version.)
        conditioned_text = (
            reference_text
            + " "
            + block_text
        )

        min_new = block_min_new_tokens(
            block_text,
            max_new_tokens,
        )

        # -------------------------------------------------------------------
        # PREPROCESS
        # -------------------------------------------------------------------

        preprocess_start = time.perf_counter_ns()

        inputs = processor(
            text=[conditioned_text],
            audio=reference_audio,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(model.device)

        prompt_len = (
            processor
            .get_audio_prompt_len(
                inputs["decoder_attention_mask"]
            )
        )

        sync_cuda()

        preprocess_ms = (
            time.perf_counter_ns() - preprocess_start
        ) / 1_000_000

        total_preprocess_ms += preprocess_ms

        # -------------------------------------------------------------------
        # INFERENCE
        # -------------------------------------------------------------------

        inference_start = time.perf_counter_ns()

        with torch.inference_mode():

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new,
                guidance_scale=PRODUCTION_GUIDANCE_SCALE,
                temperature=PRODUCTION_TEMPERATURE,
                top_p=PRODUCTION_TOP_P,
                top_k=PRODUCTION_TOP_K,
            )

        sync_cuda()

        inference_ms = (
            time.perf_counter_ns() - inference_start
        ) / 1_000_000

        # -------------------------------------------------------------------
        # DECODE
        # -------------------------------------------------------------------

        decode_start = time.perf_counter_ns()

        audio = decode_generated_audio(outputs, prompt_len)

        duration = len(audio) / SAMPLE_RATE

        # -------------------------------------------------------------------
        # RETRY IF A BLOCK CAME BACK IMPLAUSIBLY SHORT (likely skipped words)
        # -------------------------------------------------------------------

        retried = False

        if duration < block_min_reasonable_seconds(block_text):

            retried = True

            print(
                f"[WARN] Block {block_index} short "
                f"({duration:.2f}s). Retrying once."
            )

            stronger_min = min(
                max_new_tokens - 16,
                int(min_new * 1.4),
            )

            with torch.inference_mode():

                retry_outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=stronger_min,
                    guidance_scale=PRODUCTION_GUIDANCE_SCALE,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=50,
                )

            retry_audio = decode_generated_audio(
                retry_outputs,
                prompt_len,
            )

            if len(retry_audio) > len(audio):
                audio = retry_audio
                duration = len(audio) / SAMPLE_RATE

        decode_ms = (
            time.perf_counter_ns() - decode_start
        ) / 1_000_000

        total_inference_ms += inference_ms
        total_decode_ms += decode_ms
        total_samples += len(audio)

        # -------------------------------------------------------------------
        # FLOAT32 -> PCM16
        # -------------------------------------------------------------------

        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")
        pcm_bytes = pcm.tobytes()

        print(f"Inference         : {inference_ms:.2f} ms")
        print(f"Audio duration    : {duration:.2f} sec")
        print(f"Min new tokens    : {min_new}")
        print(f"Retried           : {retried}")

        yield make_frame(
            {
                "type": "audio",
                "request_id": request_id,
                "block_index": block_index,
                "block_count": len(blocks),
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
                "audio_duration_s": duration,
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "decode_ms": decode_ms,
                "min_new_tokens": min_new,
                "retried": retried,
            },
            pcm_bytes,
        )

    # -----------------------------------------------------------------------
    # FINAL METRICS
    # -----------------------------------------------------------------------

    server_total_ms = (
        time.perf_counter_ns() - server_start
    ) / 1_000_000

    audio_duration_s = total_samples / SAMPLE_RATE

    generation_rtf = (
        (total_inference_ms / 1000.0) / audio_duration_s
        if audio_duration_s > 0
        else 0.0
    )

    total_rtf = (
        (server_total_ms / 1000.0) / audio_duration_s
        if audio_duration_s > 0
        else 0.0
    )

    gpu_allocated = gpu_reserved = gpu_peak = 0.0

    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024**2
        gpu_reserved = torch.cuda.memory_reserved() / 1024**2
        gpu_peak = torch.cuda.max_memory_allocated() / 1024**2

    print("")
    print("=" * 80)
    print("SERVER TOTAL")
    print("=" * 80)
    print(f"Blocks            : {len(blocks)}")
    print(f"Inference         : {total_inference_ms:.2f} ms")
    print(f"SERVER TOTAL      : {server_total_ms:.2f} ms")
    print(f"Audio duration    : {audio_duration_s:.2f} sec")
    print(f"Generation RTF    : {generation_rtf:.4f}")
    print("=" * 80)

    yield make_frame(
        {
            "type": "end",
            "request_id": request_id,
            "block_count": len(blocks),
            "preprocess_ms": total_preprocess_ms,
            "inference_ms": total_inference_ms,
            "decode_ms": total_decode_ms,
            "server_total_ms": server_total_ms,
            "audio_duration_s": audio_duration_s,
            "generation_rtf": generation_rtf,
            "total_rtf": total_rtf,
            "gpu_allocated_mb": gpu_allocated,
            "gpu_reserved_mb": gpu_reserved,
            "gpu_peak_mb": gpu_peak,
        }
    )


# =============================================================================
# TTS API  (dual reference)
# =============================================================================

@app.post("/tts")
def tts(
    text: str = Form(...),
    agent_reference_text: str = Form(...),
    customer_reference_text: str = Form(...),
    max_new_tokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
    agent_audio: UploadFile = File(...),
    customer_audio: UploadFile = File(...),
):

    if model is None or processor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet.",
        )

    if max_new_tokens < 256 or max_new_tokens > 4096:
        raise HTTPException(
            status_code=400,
            detail="max_new_tokens must be between 256 and 4096.",
        )

    request_id = str(uuid.uuid4())

    try:

        agent_bytes = agent_audio.file.read()
        customer_bytes = customer_audio.file.read()

        if not agent_bytes:
            raise ValueError("Agent reference WAV is empty.")

        if not customer_bytes:
            raise ValueError("Customer reference WAV is empty.")

        agent_array, agent_dur = load_reference_audio(agent_bytes)
        customer_array, customer_dur = load_reference_audio(customer_bytes)

        # Rebuild the single [S1] agent [S2] customer reference.
        combined_audio = combine_references(
            agent_array,
            customer_array,
        )

        agent_text = strip_leading_tag(agent_reference_text, "S1")
        customer_text = strip_leading_tag(customer_reference_text, "S2")

        combined_reference_text = (
            f"[S1] {agent_text} [S2] {customer_text}"
        )

        print("")
        print(f"[request] Agent WAV    : {agent_audio.filename} ({agent_dur:.2f}s)")
        print(f"[request] Customer WAV : {customer_audio.filename} ({customer_dur:.2f}s)")
        print(
            f"[request] Combined ref : "
            f"{len(combined_audio) / SAMPLE_RATE:.2f}s"
        )

        return StreamingResponse(
            generate_blocks(
                full_text=text,
                reference_text=combined_reference_text,
                reference_audio=combined_audio,
                max_new_tokens=max_new_tokens,
                request_id=request_id,
            ),
            media_type="application/x-dia-reference-stream",
            headers={
                "X-Stream-Mode": "reference-pcm",
                "X-Request-ID": request_id,
                "X-Sample-Rate": str(SAMPLE_RATE),
            },
        )

    except Exception as exc:

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[ERROR] {type(exc).__name__}: {exc}")

        raise HTTPException(
            status_code=500,
            detail=(
                "TTS generation failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc
