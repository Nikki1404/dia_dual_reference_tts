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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from scipy.signal import resample_poly
from transformers import AutoProcessor, DiaForConditionalGeneration

MODEL_ID = os.getenv("MODEL_ID", "nari-labs/Dia-1.6B-0626")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
SAMPLE_RATE = 44100

# Long-text handling: one S1/S2 turn per generation. Long turns are split further.
MAX_WORDS_PER_CHUNK = int(os.getenv("MAX_WORDS_PER_CHUNK", "18"))
DEFAULT_PRODUCTION_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_PRODUCTION_MAX_NEW_TOKENS", "3072"))
DEFAULT_SEED_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_SEED_MAX_NEW_TOKENS", "1024"))
REFERENCE_SILENCE_MS = int(os.getenv("REFERENCE_SILENCE_MS", "200"))

# Production is slightly less random to reduce skipped/repeated text.
PRODUCTION_GUIDANCE_SCALE = float(os.getenv("PRODUCTION_GUIDANCE_SCALE", "3.0"))
PRODUCTION_TEMPERATURE = float(os.getenv("PRODUCTION_TEMPERATURE", "1.0"))
PRODUCTION_TOP_P = float(os.getenv("PRODUCTION_TOP_P", "0.95"))
PRODUCTION_TOP_K = int(os.getenv("PRODUCTION_TOP_K", "50"))

# Seed audition uses the native-style settings used in your earlier testing.
SEED_GUIDANCE_SCALE = float(os.getenv("SEED_GUIDANCE_SCALE", "3.0"))
SEED_TEMPERATURE = float(os.getenv("SEED_TEMPERATURE", "1.8"))
SEED_TOP_P = float(os.getenv("SEED_TOP_P", "0.90"))
SEED_TOP_K = int(os.getenv("SEED_TOP_K", "45"))

app = FastAPI(title="Dia Final Dual-Reference TTS", version="1.0.0")
processor = None
model = None
model_load_ms = None


class SeedTTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    seed: int = Field(..., ge=0, le=2147483647)
    max_new_tokens: int = Field(default=DEFAULT_SEED_MAX_NEW_TOKENS, ge=256, le=4096)


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def normalize_tags(text: str) -> str:
    text = re.sub(r"\[\s*s1\s*\]", "[S1]", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*s2\s*\]", "[S2]", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_leading_tag(text: str, tag: str) -> str:
    text = normalize_tags(text)
    text = re.sub(rf"^\s*\[{tag}\]\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def parse_turns(text: str):
    text = normalize_tags(text)
    matches = list(re.finditer(r"\[(S1|S2)\]\s*", text, flags=re.IGNORECASE))
    if not matches:
        raise ValueError("Transcript must contain [S1] and/or [S2] tags.")

    turns = []
    for i, match in enumerate(matches):
        speaker = match.group(1).upper()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        speech = text[start:end].strip()
        if speech:
            turns.append((speaker, speech))

    if not turns:
        raise ValueError("No speech found after [S1]/[S2] tags.")
    return turns


def split_sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def split_long_turn(speaker: str, speech: str):
    if len(speech.split()) <= MAX_WORDS_PER_CHUNK:
        return [(speaker, speech)]

    sentences = split_sentences(speech) or [speech]
    out = []
    current = []
    current_words = 0

    for sentence in sentences:
        words = sentence.split()
        count = len(words)

        if count > MAX_WORDS_PER_CHUNK:
            if current:
                out.append((speaker, " ".join(current)))
                current = []
                current_words = 0
            for j in range(0, count, MAX_WORDS_PER_CHUNK):
                out.append((speaker, " ".join(words[j:j + MAX_WORDS_PER_CHUNK])))
            continue

        if current and current_words + count > MAX_WORDS_PER_CHUNK:
            out.append((speaker, " ".join(current)))
            current = []
            current_words = 0

        current.append(sentence)
        current_words += count

    if current:
        out.append((speaker, " ".join(current)))

    return out


def build_chunks(text: str):
    """External mapping: [S1]=agent, [S2]=customer."""
    chunks = []
    for speaker, speech in parse_turns(text):
        chunks.extend(split_long_turn(speaker, speech))
    return chunks


def read_reference_wav(wav_bytes: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == 0:
        raise ValueError("Reference WAV is empty.")

    if sr != SAMPLE_RATE:
        divisor = gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, sr // divisor).astype(np.float32)
    return audio


def trim_outer_silence(audio: np.ndarray, threshold_ratio: float = 0.02, padding_ms: int = 40):
    if len(audio) == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return audio
    active = np.flatnonzero(np.abs(audio) >= peak * threshold_ratio)
    if len(active) == 0:
        return audio
    padding = int(SAMPLE_RATE * padding_ms / 1000)
    start = max(0, int(active[0]) - padding)
    end = min(len(audio), int(active[-1]) + padding + 1)
    return audio[start:end]


def combine_two_references(first_audio: np.ndarray, second_audio: np.ndarray) -> np.ndarray:
    first_audio = trim_outer_silence(first_audio)
    second_audio = trim_outer_silence(second_audio)
    silence = np.zeros(int(SAMPLE_RATE * REFERENCE_SILENCE_MS / 1000), dtype=np.float32)
    return np.concatenate([first_audio, silence, second_audio]).astype(np.float32)


def decode_conditioned_audio(outputs, prompt_len):
    decoded = processor.batch_decode(outputs, audio_prompt_len=prompt_len)
    audio = decoded[0] if isinstance(decoded, (list, tuple)) else decoded
    if torch.is_tensor(audio):
        audio = audio.detach().float().cpu().numpy()
    audio = np.squeeze(np.asarray(audio, dtype=np.float32))
    if audio.ndim != 1:
        raise RuntimeError(f"Unexpected audio shape: {audio.shape}")
    if len(audio) == 0:
        raise RuntimeError("Dia returned empty audio.")
    if not np.all(np.isfinite(audio)):
        raise RuntimeError("Generated audio contains NaN/Inf.")
    return audio


def decode_plain_audio(outputs):
    decoded = processor.batch_decode(outputs)
    audio = decoded[0] if isinstance(decoded, (list, tuple)) else decoded
    if torch.is_tensor(audio):
        audio = audio.detach().float().cpu().numpy()
    audio = np.squeeze(np.asarray(audio, dtype=np.float32))
    if audio.ndim != 1:
        raise RuntimeError(f"Unexpected audio shape: {audio.shape}")
    if len(audio) == 0:
        raise RuntimeError("Dia returned empty audio.")
    return audio


def make_frame(metadata: dict, pcm_bytes: bytes = b"") -> bytes:
    metadata_bytes = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    return (
        struct.pack(">I", len(metadata_bytes))
        + metadata_bytes
        + struct.pack(">Q", len(pcm_bytes))
        + pcm_bytes
    )


@app.on_event("startup")
def load_model():
    global processor, model, model_load_ms

    print("\n" + "=" * 80)
    print("DIA FINAL TTS STARTUP")
    print("=" * 80)
    print(f"Model             : {MODEL_ID}")
    print(f"Device            : {DEVICE}")
    print(f"PyTorch           : {torch.__version__}")
    print(f"PyTorch CUDA      : {torch.version.cuda}")
    print(f"CUDA available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU               : {torch.cuda.get_device_name(0)}")
    print(f"DTYPE             : {DTYPE}")
    print(f"Max words/chunk   : {MAX_WORDS_PER_CHUNK}")
    print("=" * 80)

    started = time.perf_counter_ns()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = DiaForConditionalGeneration.from_pretrained(MODEL_ID)
    model = model.to(device=DEVICE, dtype=DTYPE)
    model.eval()
    sync_cuda()

    model_load_ms = (time.perf_counter_ns() - started) / 1_000_000
    print(f"Model loaded      : {model_load_ms:.2f} ms")
    print("=" * 80)


@app.get("/health")
def health():
    result = {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "cuda_available": torch.cuda.is_available(),
        "sample_rate": SAMPLE_RATE,
        "max_words_per_chunk": MAX_WORDS_PER_CHUNK,
        "production_max_new_tokens": DEFAULT_PRODUCTION_MAX_NEW_TOKENS,
        "seed_max_new_tokens": DEFAULT_SEED_MAX_NEW_TOKENS,
        "endpoints": {
            "/tts": "dual-reference long production TTS",
            "/tts_seed": "native Dia seed auditioning",
        },
    }
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
    return result


def generate_production_chunks(
    full_text: str,
    agent_reference_text: str,
    customer_reference_text: str,
    agent_audio: np.ndarray,
    customer_audio: np.ndarray,
    max_new_tokens: int,
    request_id: str,
) -> Iterator[bytes]:
    """
    External mapping:
      S1 = agent
      S2 = customer

    To preserve Dia's recommended alternating prompt structure while using
    independently selected voices:

    Agent target:
      audio prompt = agent then customer
      text prompt  = [S1] agent_ref [S2] customer_ref [S1] target

    Customer target:
      audio prompt = customer then agent
      text prompt  = [S1] customer_ref [S2] agent_ref [S1] target

    The target is internally [S1] in both cases. External speaker identity is
    carried in metadata and final chunk order.
    """
    server_start = time.perf_counter_ns()
    chunks = build_chunks(full_text)

    agent_first_audio = combine_two_references(agent_audio, customer_audio)
    customer_first_audio = combine_two_references(customer_audio, agent_audio)

    agent_prompt = f"[S1] {agent_reference_text} [S2] {customer_reference_text}"
    customer_prompt = f"[S1] {customer_reference_text} [S2] {agent_reference_text}"

    print("\n" + "=" * 80)
    print("PRODUCTION TTS REQUEST")
    print("=" * 80)
    print(f"Request ID        : {request_id}")
    print(f"Chunks            : {len(chunks)}")
    print(f"Max words/chunk   : {MAX_WORDS_PER_CHUNK}")
    print(f"Max new tokens    : {max_new_tokens}")
    print("=" * 80)

    total_preprocess_ms = 0.0
    total_inference_ms = 0.0
    total_decode_ms = 0.0
    total_samples = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for chunk_index, (speaker, speech) in enumerate(chunks, start=1):
        if speaker == "S1":
            reference_audio = agent_first_audio
            reference_text = agent_prompt
            voice_name = "agent"
        else:
            reference_audio = customer_first_audio
            reference_text = customer_prompt
            voice_name = "customer"

        target_text = f"[S1] {speech}"
        conditioned_text = f"{reference_text} {target_text}"
        chunk_words = len(speech.split())

        print("\n" + "-" * 80)
        print(f"Chunk {chunk_index}/{len(chunks)}")
        print(f"External speaker  : {speaker} ({voice_name})")
        print(f"Words             : {chunk_words}")
        print(f"Text              : {speech}")

        preprocess_start = time.perf_counter_ns()
        inputs = processor(
            text=[conditioned_text],
            audio=reference_audio,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        prompt_len = processor.get_audio_prompt_len(inputs["decoder_attention_mask"])
        sync_cuda()
        preprocess_ms = (time.perf_counter_ns() - preprocess_start) / 1_000_000
        total_preprocess_ms += preprocess_ms

        inference_start = time.perf_counter_ns()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                guidance_scale=PRODUCTION_GUIDANCE_SCALE,
                temperature=PRODUCTION_TEMPERATURE,
                top_p=PRODUCTION_TOP_P,
                top_k=PRODUCTION_TOP_K,
            )
        sync_cuda()
        inference_ms = (time.perf_counter_ns() - inference_start) / 1_000_000
        total_inference_ms += inference_ms

        decode_start = time.perf_counter_ns()
        audio = decode_conditioned_audio(outputs, prompt_len)
        decode_ms = (time.perf_counter_ns() - decode_start) / 1_000_000
        total_decode_ms += decode_ms
        total_samples += len(audio)
        duration = len(audio) / SAMPLE_RATE

        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")

        print(f"Inference         : {inference_ms:.2f} ms")
        print(f"Audio duration    : {duration:.2f}s")

        frame = make_frame(
            {
                "type": "audio",
                "request_id": request_id,
                "chunk_index": chunk_index,
                "chunk_count": len(chunks),
                "speaker": speaker,
                "voice": voice_name,
                "chunk_words": chunk_words,
                "chunk_text": speech,
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
                "audio_duration_s": duration,
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "decode_ms": decode_ms,
            },
            pcm.tobytes(),
        )

        del inputs, outputs, audio, pcm
        yield frame

    server_total_ms = (time.perf_counter_ns() - server_start) / 1_000_000
    audio_duration_s = total_samples / SAMPLE_RATE
    generation_rtf = (
        (total_inference_ms / 1000.0) / audio_duration_s if audio_duration_s > 0 else 0.0
    )
    total_rtf = (
        (server_total_ms / 1000.0) / audio_duration_s if audio_duration_s > 0 else 0.0
    )

    yield make_frame(
        {
            "type": "end",
            "request_id": request_id,
            "chunk_count": len(chunks),
            "preprocess_ms": total_preprocess_ms,
            "inference_ms": total_inference_ms,
            "decode_ms": total_decode_ms,
            "server_total_ms": server_total_ms,
            "audio_duration_s": audio_duration_s,
            "generation_rtf": generation_rtf,
            "total_rtf": total_rtf,
            "gpu_peak_mb": (
                torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
            ),
        }
    )


@app.post("/tts")
def tts(
    text: str = Form(...),
    agent_reference_text: str = Form(...),
    customer_reference_text: str = Form(...),
    max_new_tokens: int = Form(DEFAULT_PRODUCTION_MAX_NEW_TOKENS),
    agent_audio: UploadFile = File(...),
    customer_audio: UploadFile = File(...),
):
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    if not 256 <= max_new_tokens <= 4096:
        raise HTTPException(status_code=400, detail="max_new_tokens must be between 256 and 4096.")

    request_id = str(uuid.uuid4())

    try:
        normalized_text = normalize_tags(text)
        parse_turns(normalized_text)

        agent_text = strip_leading_tag(agent_reference_text, "S1")
        customer_text = strip_leading_tag(customer_reference_text, "S2")
        if not agent_text:
            raise ValueError("Agent reference text is empty.")
        if not customer_text:
            raise ValueError("Customer reference text is empty.")

        agent_bytes = agent_audio.file.read()
        customer_bytes = customer_audio.file.read()
        if not agent_bytes:
            raise ValueError("Agent reference WAV is empty.")
        if not customer_bytes:
            raise ValueError("Customer reference WAV is empty.")

        agent_array = read_reference_wav(agent_bytes)
        customer_array = read_reference_wav(customer_bytes)

        print(f"[reference] Agent file       : {agent_audio.filename}")
        print(f"[reference] Customer file    : {customer_audio.filename}")
        print(f"[reference] Agent duration   : {len(agent_array) / SAMPLE_RATE:.2f}s")
        print(f"[reference] Customer duration: {len(customer_array) / SAMPLE_RATE:.2f}s")

        return StreamingResponse(
            generate_production_chunks(
                full_text=normalized_text,
                agent_reference_text=agent_text,
                customer_reference_text=customer_text,
                agent_audio=agent_array,
                customer_audio=customer_array,
                max_new_tokens=max_new_tokens,
                request_id=request_id,
            ),
            media_type="application/x-dia-production-stream",
            headers={
                "X-Request-ID": request_id,
                "X-Sample-Rate": str(SAMPLE_RATE),
                "X-Chunk-Mode": "one-speaker-turn",
            },
        )

    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Production TTS failed: {type(exc).__name__}: {exc}",
        ) from exc


@app.post("/tts_seed")
def tts_seed(req: SeedTTSRequest):
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    request_id = str(uuid.uuid4())
    server_start = time.perf_counter_ns()

    try:
        text = normalize_tags(req.text)
        set_seed(req.seed)

        preprocess_start = time.perf_counter_ns()
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)
        sync_cuda()
        preprocess_ms = (time.perf_counter_ns() - preprocess_start) / 1_000_000

        inference_start = time.perf_counter_ns()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                guidance_scale=SEED_GUIDANCE_SCALE,
                temperature=SEED_TEMPERATURE,
                top_p=SEED_TOP_P,
                top_k=SEED_TOP_K,
            )
        sync_cuda()
        inference_ms = (time.perf_counter_ns() - inference_start) / 1_000_000

        decode_start = time.perf_counter_ns()
        audio = decode_plain_audio(outputs)
        decode_ms = (time.perf_counter_ns() - decode_start) / 1_000_000
        duration = len(audio) / SAMPLE_RATE

        encode_start = time.perf_counter_ns()
        buffer = io.BytesIO()
        sf.write(buffer, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        wav_bytes = buffer.getvalue()
        encode_ms = (time.perf_counter_ns() - encode_start) / 1_000_000

        server_total_ms = (time.perf_counter_ns() - server_start) / 1_000_000

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Request-ID": request_id,
                "X-Seed": str(req.seed),
                "X-Preprocess-Time-MS": f"{preprocess_ms:.2f}",
                "X-Inference-Time-MS": f"{inference_ms:.2f}",
                "X-Decode-Time-MS": f"{decode_ms:.2f}",
                "X-Encoding-Time-MS": f"{encode_ms:.2f}",
                "X-Server-Total-MS": f"{server_total_ms:.2f}",
                "X-Audio-Duration-S": f"{duration:.3f}",
                "X-Sample-Rate": str(SAMPLE_RATE),
            },
        )

    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(
            status_code=500,
            detail=f"Seed TTS failed: {type(exc).__name__}: {exc}",
        ) from exc
