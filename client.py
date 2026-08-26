#!/usr/bin/env python3

import argparse
import io
import json
import queue
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf


def read_text_file(path_value: str) -> str:
    path = Path(path_value)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)
    return path.read_text(encoding="utf-8").strip()


def resolve_main_text(args) -> str:
    if args.text is not None:
        value = args.text.strip()
    else:
        value = read_text_file(args.text_file)
    if not value:
        print("Input text is empty.")
        sys.exit(1)
    return value


def resolve_reference_text(direct_value, file_value, name) -> str:
    if direct_value is not None:
        value = direct_value.strip()
    elif file_value is not None:
        value = read_text_file(file_value)
    else:
        print(f"Production mode requires {name} reference text.")
        sys.exit(1)
    if not value:
        print(f"{name} reference text is empty.")
        sys.exit(1)
    return value


def receive_exact(raw, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        part = raw.read(size - len(data))
        if not part:
            raise EOFError("Server stream ended unexpectedly.")
        data.extend(part)
    return bytes(data)


def read_frame(raw):
    metadata_size = struct.unpack(">I", receive_exact(raw, 4))[0]
    metadata = json.loads(receive_exact(raw, metadata_size).decode("utf-8"))
    pcm_size = struct.unpack(">Q", receive_exact(raw, 8))[0]
    pcm_bytes = receive_exact(raw, pcm_size) if pcm_size > 0 else b""
    return metadata, pcm_bytes


def adjust_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    if speed == 1.0:
        return audio
    try:
        import librosa
        return librosa.effects.time_stretch(audio, rate=speed)
    except Exception as exc:
        print(f"[WARN] Speed adjustment failed: {exc}")
        return audio


def run_seed_mode(args, text: str):
    url = args.server.rstrip("/") + "/tts_seed"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else 1024
    payload = {
        "text": text,
        "seed": args.seed,
        "max_new_tokens": max_new_tokens,
    }

    print("\n" + "=" * 80)
    print("DIA SEED TEST MODE")
    print("=" * 80)
    print(f"Server            : {url}")
    print(f"Seed              : {args.seed}")
    print(f"Words             : {len(text.split())}")
    print(f"Max new tokens    : {max_new_tokens}")
    print(f"Output            : {output_path}")
    print("=" * 80)

    started = time.perf_counter()
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.post(url, json=payload, timeout=args.timeout)
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        sys.exit(1)
    finally:
        session.close()

    if response.status_code != 200:
        print(f"Request failed ({response.status_code}): {response.text}")
        sys.exit(1)

    output_path.write_bytes(response.content)
    total_ms = (time.perf_counter() - started) * 1000

    print("\n" + "=" * 80)
    print("SEED RESULT")
    print("=" * 80)
    print(f"Seed              : {args.seed}")
    print(f"Saved             : {output_path}")
    print(f"Client total      : {total_ms:.2f} ms")
    print(f"Server inference  : {response.headers.get('X-Inference-Time-MS', '0')} ms")
    print(f"Server total      : {response.headers.get('X-Server-Total-MS', '0')} ms")
    print(f"Audio duration    : {response.headers.get('X-Audio-Duration-S', '0')} s")
    print("=" * 80)

    if args.play:
        try:
            import sounddevice as sd
            audio, sr = sf.read(str(output_path), dtype="float32")
            audio = adjust_speed(audio, args.speed)
            sd.play(audio, sr)
            sd.wait()
        except Exception as exc:
            print(f"Playback failed: {exc}")


def run_production_mode(args, text: str):
    if not args.agent_audio:
        print("Production mode requires --agent-audio.")
        sys.exit(1)
    if not args.customer_audio:
        print("Production mode requires --customer-audio.")
        sys.exit(1)

    agent_audio_path = Path(args.agent_audio)
    customer_audio_path = Path(args.customer_audio)
    for path in (agent_audio_path, customer_audio_path):
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)

    agent_reference_text = resolve_reference_text(
        args.agent_text, args.agent_text_file, "agent"
    )
    customer_reference_text = resolve_reference_text(
        args.customer_text, args.customer_text_file, "customer"
    )

    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else 3072
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = args.server.rstrip("/") + "/tts"

    print("\n" + "=" * 80)
    print("DIA PRODUCTION TTS")
    print("=" * 80)
    print(f"Server            : {url}")
    print(f"Input words       : {len(text.split())}")
    print(f"Agent WAV         : {agent_audio_path}")
    print(f"Customer WAV      : {customer_audio_path}")
    print(f"Max new tokens    : {max_new_tokens}")
    print(f"Output            : {output_path}")
    print("=" * 80)

    playback_queue = queue.Queue()
    playback_enabled = False
    playback_thread = None

    if args.play:
        try:
            import sounddevice as sd
            playback_enabled = True

            def playback_worker():
                while True:
                    item = playback_queue.get()
                    if item is None:
                        playback_queue.task_done()
                        break
                    chunk_index, chunk_count, speaker, audio, sample_rate = item
                    playback_audio = adjust_speed(audio, args.speed)
                    print(f"[PLAY] {chunk_index}/{chunk_count} {speaker}")
                    sd.play(playback_audio, sample_rate)
                    sd.wait()  # next chunk waits until current chunk finishes
                    playback_queue.task_done()

            playback_thread = threading.Thread(target=playback_worker, daemon=True)
            playback_thread.start()
        except Exception as exc:
            print(f"Playback disabled: {exc}")
            playback_enabled = False

    request_start = time.perf_counter()
    session = requests.Session()
    session.trust_env = False

    try:
        with open(agent_audio_path, "rb") as agent_handle, open(customer_audio_path, "rb") as customer_handle:
            files = {
                "agent_audio": (agent_audio_path.name, agent_handle, "audio/wav"),
                "customer_audio": (customer_audio_path.name, customer_handle, "audio/wav"),
            }
            data = {
                "text": text,
                "agent_reference_text": agent_reference_text,
                "customer_reference_text": customer_reference_text,
                "max_new_tokens": str(max_new_tokens),
            }
            response = session.post(
                url,
                data=data,
                files=files,
                stream=True,
                timeout=args.timeout,
            )
    except requests.RequestException as exc:
        session.close()
        print(f"Request failed: {exc}")
        sys.exit(1)

    headers_received = time.perf_counter()

    if response.status_code != 200:
        print(f"Request failed ({response.status_code}): {response.text}")
        response.close()
        session.close()
        sys.exit(1)

    wav_writer = sf.SoundFile(
        str(output_path),
        mode="w",
        samplerate=44100,
        channels=1,
        subtype="PCM_16",
        format="WAV",
    )

    audio_parts = []
    first_audio_time = None
    chunk_counter = 0
    server_metrics = {}

    try:
        while True:
            metadata, pcm_bytes = read_frame(response.raw)
            frame_type = metadata.get("type")

            if frame_type == "audio":
                chunk_counter += 1
                if first_audio_time is None:
                    first_audio_time = time.perf_counter()

                pcm = np.frombuffer(pcm_bytes, dtype="<i2")
                audio = pcm.astype(np.float32) / 32767.0

                # Append every chunk into one final WAV in exact transcript order.
                wav_writer.write(audio)
                audio_parts.append(audio.copy())

                print("")
                print(f"[RECV] Chunk {metadata['chunk_index']}/{metadata['chunk_count']}")
                print(f"       Speaker    : {metadata.get('speaker')} ({metadata.get('voice')})")
                print(f"       Text       : {metadata.get('chunk_text')}")
                print(f"       Words      : {metadata.get('chunk_words')}")
                print(f"       Duration   : {metadata['audio_duration_s']:.2f}s")
                print(f"       Inference  : {metadata['inference_ms']:.2f}ms")

                if playback_enabled:
                    playback_queue.put(
                        (
                            metadata["chunk_index"],
                            metadata["chunk_count"],
                            metadata["speaker"],
                            audio.copy(),
                            metadata["sample_rate"],
                        )
                    )

            elif frame_type == "end":
                server_metrics = metadata
                break

    finally:
        wav_writer.close()
        response.close()
        session.close()

    receive_complete_time = time.perf_counter()

    if playback_enabled:
        playback_queue.join()
        playback_queue.put(None)
        playback_queue.join()
        if playback_thread is not None:
            playback_thread.join(timeout=10)

    adjusted_path = None
    if args.save_adjusted and args.speed != 1.0 and audio_parts:
        full_audio = np.concatenate(audio_parts)
        adjusted_audio = adjust_speed(full_audio, args.speed)
        speed_name = str(args.speed).replace(".", "_")
        adjusted_path = output_path.with_name(
            output_path.stem + "_speed_" + speed_name + output_path.suffix
        )
        sf.write(str(adjusted_path), adjusted_audio, 44100, subtype="PCM_16")

    ttfb_ms = (headers_received - request_start) * 1000
    ttfa_ms = ((first_audio_time - request_start) * 1000) if first_audio_time else 0.0
    total_ms = (receive_complete_time - request_start) * 1000

    print("\n" + "=" * 80)
    print("CLIENT LATENCY")
    print("=" * 80)
    print(f"HTTP headers/TTFB : {ttfb_ms:.2f} ms")
    print(f"TTFA / TTFT       : {ttfa_ms:.2f} ms")
    print(f"Receive total     : {total_ms:.2f} ms")
    print(f"Chunks received   : {chunk_counter}")

    print("\n" + "=" * 80)
    print("SERVER METRICS")
    print("=" * 80)
    print(f"Chunks            : {server_metrics.get('chunk_count', 0)}")
    print(f"Preprocess        : {server_metrics.get('preprocess_ms', 0):.2f} ms")
    print(f"Inference         : {server_metrics.get('inference_ms', 0):.2f} ms")
    print(f"Decode            : {server_metrics.get('decode_ms', 0):.2f} ms")
    print(f"Server total      : {server_metrics.get('server_total_ms', 0):.2f} ms")
    print(f"Audio duration    : {server_metrics.get('audio_duration_s', 0):.2f}s")
    print(f"Generation RTF    : {server_metrics.get('generation_rtf', 0):.4f}")
    print(f"Total RTF         : {server_metrics.get('total_rtf', 0):.4f}")
    print(f"GPU peak          : {server_metrics.get('gpu_peak_mb', 0):.2f} MB")

    print("\n" + "=" * 80)
    print("OUTPUT")
    print("=" * 80)
    print(f"WAV               : {output_path}")
    if adjusted_path is not None:
        print(f"Adjusted WAV      : {adjusted_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Dia seed audition + dual-reference production TTS client"
    )
    parser.add_argument("--server", default="http://localhost:8000")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Direct [S1]/[S2] text")
    input_group.add_argument("--text-file", help="TXT file containing [S1]/[S2] text")

    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--agent-audio", default=None)
    parser.add_argument("--customer-audio", default=None)
    parser.add_argument("--agent-text", default=None)
    parser.add_argument("--agent-text-file", default=None)
    parser.add_argument("--customer-text", default=None)
    parser.add_argument("--customer-text-file", default=None)

    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--save-adjusted", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800.0)

    args = parser.parse_args()

    if args.speed <= 0:
        print("--speed must be greater than 0")
        sys.exit(1)

    text = resolve_main_text(args)

    if args.seed is not None:
        run_seed_mode(args, text)
    else:
        run_production_mode(args, text)


if __name__ == "__main__":
    main()
