#!/usr/bin/env python3
# client.py
#
# Same block-streaming client that produced continuous audio.
# Only the reference inputs changed: instead of one --reference-audio /
# --reference-text, it now sends the split dual references:
#
#   --agent-audio / --agent-text-file
#   --customer-audio / --customer-text-file
#
# The receive/append/playback path is unchanged, so audio stays continuous.

import argparse
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


# =============================================================================
# FILE HELPERS
# =============================================================================

def read_text_file(path_value):

    path = Path(path_value)

    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    return path.read_text(encoding="utf-8").strip()


def resolve_reference_text(direct, file_path, name):

    if direct is not None:
        return direct.strip()

    if file_path is not None:
        return read_text_file(file_path)

    print(f"Missing {name} reference text.")
    sys.exit(1)


# =============================================================================
# NETWORK STREAM HELPERS
# =============================================================================

def receive_exact(raw, size):

    data = bytearray()

    while len(data) < size:

        part = raw.read(size - len(data))

        if not part:
            raise EOFError("Server stream ended unexpectedly.")

        data.extend(part)

    return bytes(data)


def read_frame(raw):

    metadata_size = struct.unpack(
        ">I",
        receive_exact(raw, 4),
    )[0]

    metadata = json.loads(
        receive_exact(raw, metadata_size).decode("utf-8")
    )

    pcm_size = struct.unpack(
        ">Q",
        receive_exact(raw, 8),
    )[0]

    pcm_bytes = b""

    if pcm_size > 0:
        pcm_bytes = receive_exact(raw, pcm_size)

    return metadata, pcm_bytes


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Dia long-text TTS client with dual "
            "(agent + customer) reference conditioning"
        )
    )

    parser.add_argument(
        "--server",
        default="http://localhost:8000",
    )

    parser.add_argument(
        "--text-file",
        required=True,
        help="Long [S1]/[S2] transcript.",
    )

    # --- dual reference inputs ------------------------------------------------

    parser.add_argument(
        "--agent-audio",
        required=True,
        help="Local agent reference WAV ([S1]).",
    )

    parser.add_argument(
        "--customer-audio",
        required=True,
        help="Local customer reference WAV ([S2]).",
    )

    parser.add_argument("--agent-text")
    parser.add_argument("--agent-text-file")

    parser.add_argument("--customer-text")
    parser.add_argument("--customer-text-file")

    # --- output / generation --------------------------------------------------

    parser.add_argument(
        "--output",
        default="full_call.wav",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=3072,
    )

    parser.add_argument(
        "--play",
        action="store_true",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Client playback speed. 1.0=original, 0.9=10%% slower.",
    )

    parser.add_argument(
        "--save-adjusted",
        action="store_true",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
    )

    args = parser.parse_args()

    # =========================================================================
    # VALIDATION
    # =========================================================================

    transcript_path = Path(args.text_file)
    agent_audio_path = Path(args.agent_audio)
    customer_audio_path = Path(args.customer_audio)

    for path in [
        transcript_path,
        agent_audio_path,
        customer_audio_path,
    ]:
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)

    if args.speed <= 0:
        print("--speed must be > 0")
        sys.exit(1)

    # =========================================================================
    # LOAD TEXT
    # =========================================================================

    text = read_text_file(transcript_path)

    agent_text = resolve_reference_text(
        args.agent_text,
        args.agent_text_file,
        "agent",
    )

    customer_text = resolve_reference_text(
        args.customer_text,
        args.customer_text_file,
        "customer",
    )

    url = args.server.rstrip("/") + "/tts"

    print("")
    print("=" * 80)
    print("DIA DUAL-REFERENCE TTS CLIENT")
    print("=" * 80)
    print(f"Server            : {url}")
    print(f"Transcript        : {transcript_path}")
    print(f"Agent WAV         : {agent_audio_path}")
    print(f"Customer WAV      : {customer_audio_path}")
    print(f"Output            : {args.output}")
    print(f"Transcript words  : {len(text.split())}")
    print(f"Max new tokens    : {args.max_new_tokens}")
    print(f"Playback          : {args.play}")
    print(f"Playback speed    : {args.speed}")
    print("=" * 80)

    # =========================================================================
    # PLAYBACK QUEUE
    # =========================================================================

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

                    (
                        block_index,
                        block_count,
                        audio,
                        sample_rate,
                    ) = item

                    playback_audio = audio

                    if args.speed != 1.0:

                        try:

                            import librosa

                            playback_audio = (
                                librosa
                                .effects
                                .time_stretch(
                                    audio,
                                    rate=args.speed,
                                )
                            )

                        except Exception as exc:
                            print(f"[WARN] Speed adjustment failed: {exc}")

                    print("")
                    print(
                        f"[PLAY] Starting block "
                        f"{block_index}/{block_count}"
                    )

                    # Strict sequential playback.
                    sd.play(playback_audio, sample_rate)
                    sd.wait()

                    print(
                        f"[PLAY] Finished block "
                        f"{block_index}/{block_count}"
                    )

                    playback_queue.task_done()

            playback_thread = threading.Thread(
                target=playback_worker,
                daemon=True,
            )

            playback_thread.start()

        except Exception as exc:
            print(f"Playback disabled: {exc}")
            playback_enabled = False

    # =========================================================================
    # MULTIPART REQUEST
    # =========================================================================

    request_start = time.perf_counter()

    agent_handle = open(agent_audio_path, "rb")
    customer_handle = open(customer_audio_path, "rb")

    try:

        files = {
            "agent_audio": (
                agent_audio_path.name,
                agent_handle,
                "audio/wav",
            ),
            "customer_audio": (
                customer_audio_path.name,
                customer_handle,
                "audio/wav",
            ),
        }

        data = {
            "text": text,
            "agent_reference_text": agent_text,
            "customer_reference_text": customer_text,
            "max_new_tokens": str(args.max_new_tokens),
        }

        response = requests.post(
            url,
            data=data,
            files=files,
            stream=True,
            timeout=args.timeout,
        )

    except requests.RequestException as exc:
        agent_handle.close()
        customer_handle.close()
        print(f"Request failed: {exc}")
        sys.exit(1)

    finally:
        agent_handle.close()
        customer_handle.close()

    headers_received = time.perf_counter()

    if response.status_code != 200:
        print(
            f"Request failed ({response.status_code}): {response.text}"
        )
        sys.exit(1)

    # =========================================================================
    # FINAL WAV WRITER  (unchanged: append each block -> continuous audio)
    # =========================================================================

    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    wav_writer = sf.SoundFile(
        str(output_path),
        mode="w",
        samplerate=44100,
        channels=1,
        subtype="PCM_16",
        format="WAV",
    )

    complete_audio_parts = []

    first_audio_time = None
    receive_complete_time = None
    block_counter = 0
    server_metrics = {}

    # =========================================================================
    # RECEIVE STREAM
    # =========================================================================

    try:

        while True:

            metadata, pcm_bytes = read_frame(response.raw)

            frame_type = metadata.get("type")

            if frame_type == "audio":

                block_counter += 1

                if first_audio_time is None:
                    first_audio_time = time.perf_counter()

                pcm = np.frombuffer(pcm_bytes, dtype="<i2")

                audio = pcm.astype(np.float32) / 32767.0

                # Append directly to ONE final WAV.
                wav_writer.write(audio)

                complete_audio_parts.append(audio.copy())

                print("")
                print(
                    f"[RECV] Block "
                    f"{metadata['block_index']}/{metadata['block_count']}"
                )
                print(f"       Duration  : {metadata['audio_duration_s']:.2f}s")
                print(f"       Inference : {metadata['inference_ms']:.2f}ms")

                if metadata.get("retried"):
                    print(f"       Retried   : True")

                if playback_enabled:

                    playback_queue.put(
                        (
                            metadata["block_index"],
                            metadata["block_count"],
                            audio.copy(),
                            metadata["sample_rate"],
                        )
                    )

            elif frame_type == "end":
                server_metrics = metadata
                break

    finally:
        wav_writer.close()

    receive_complete_time = time.perf_counter()

    # =========================================================================
    # WAIT FOR ALL PLAYBACK
    # =========================================================================

    if playback_enabled:

        playback_queue.join()
        playback_queue.put(None)
        playback_queue.join()

        if playback_thread:
            playback_thread.join(timeout=10)

    playback_complete_time = time.perf_counter()

    # =========================================================================
    # OPTIONAL FINAL SPEED-ADJUSTED FILE
    # =========================================================================

    adjusted_path = None

    if (
        args.save_adjusted
        and args.speed != 1.0
        and complete_audio_parts
    ):

        try:

            import librosa

            full_audio = np.concatenate(complete_audio_parts)

            adjusted_audio = librosa.effects.time_stretch(
                full_audio,
                rate=args.speed,
            )

            speed_name = str(args.speed).replace(".", "_")

            adjusted_path = output_path.with_name(
                output_path.stem
                + "_speed_"
                + speed_name
                + output_path.suffix
            )

            sf.write(
                str(adjusted_path),
                adjusted_audio,
                44100,
                subtype="PCM_16",
            )

        except Exception as exc:
            print(f"Adjusted WAV save failed: {exc}")

    # =========================================================================
    # METRICS
    # =========================================================================

    ttfb_ms = (headers_received - request_start) * 1000

    ttfa_ms = (
        (first_audio_time - request_start) * 1000
        if first_audio_time is not None
        else 0.0
    )

    receive_total_ms = (receive_complete_time - request_start) * 1000
    playback_total_ms = (playback_complete_time - request_start) * 1000

    print("")
    print("=" * 80)
    print("CLIENT LATENCY")
    print("=" * 80)
    print(f"HTTP headers / TTFB : {ttfb_ms:.2f} ms")
    print(f"TTFA / TTFT         : {ttfa_ms:.2f} ms")
    print(f"Receive total       : {receive_total_ms:.2f} ms")

    if args.play:
        print(f"E2E incl playback   : {playback_total_ms:.2f} ms")

    print(f"Blocks received     : {block_counter}")

    print("")
    print("=" * 80)
    print("SERVER METRICS")
    print("=" * 80)
    print(f"Blocks              : {server_metrics.get('block_count', 0)}")
    print(f"Inference           : {server_metrics.get('inference_ms', 0):.2f} ms")
    print(f"SERVER TOTAL        : {server_metrics.get('server_total_ms', 0):.2f} ms")
    print(f"Audio duration      : {server_metrics.get('audio_duration_s', 0):.2f}s")
    print(f"Generation RTF      : {server_metrics.get('generation_rtf', 0):.4f}")
    print(f"GPU peak            : {server_metrics.get('gpu_peak_mb', 0):.2f} MB")

    print("")
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)
    print(f"Original WAV        : {output_path}")

    if adjusted_path:
        print(f"Adjusted WAV        : {adjusted_path}")

    print("=" * 80)


if __name__ == "__main__":
    main()
