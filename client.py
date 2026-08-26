#!/usr/bin/env python3

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

def read_text_file(
    path_value,
):

    path = Path(
        path_value
    )

    if not path.exists():

        print(
            f"File not found: {path}"
        )

        sys.exit(1)

    return path.read_text(
        encoding="utf-8"
    ).strip()


def resolve_text(args):

    if args.text is not None:

        return args.text.strip()

    return read_text_file(
        args.text_file
    )


def resolve_reference_text(
    direct,
    file_path,
    name,
):

    if direct is not None:

        return direct.strip()

    if file_path is not None:

        return read_text_file(
            file_path
        )

    print(
        f"Missing {name} reference text."
    )

    sys.exit(1)


# =============================================================================
# STREAM
# =============================================================================

def receive_exact(
    raw,
    size,
):

    data = bytearray()

    while len(data) < size:

        part = raw.read(
            size - len(data)
        )

        if not part:

            raise EOFError(
                "Stream ended unexpectedly."
            )

        data.extend(
            part
        )

    return bytes(
        data
    )


def read_frame(raw):

    metadata_size = struct.unpack(
        ">I",
        receive_exact(
            raw,
            4,
        ),
    )[0]


    metadata = json.loads(
        receive_exact(
            raw,
            metadata_size,
        ).decode(
            "utf-8"
        )
    )


    pcm_size = struct.unpack(
        ">Q",
        receive_exact(
            raw,
            8,
        ),
    )[0]


    pcm_bytes = (
        receive_exact(
            raw,
            pcm_size,
        )

        if pcm_size

        else b""
    )


    return (
        metadata,
        pcm_bytes,
    )


# =============================================================================
# SPEED
# =============================================================================

def adjust_speed(
    audio,
    speed,
):

    if speed == 1.0:

        return audio


    try:

        import librosa


        return (
            librosa.effects
            .time_stretch(
                audio,
                rate=speed,
            )
        )


    except Exception as exc:

        print(
            f"[WARN] Speed adjustment failed: "
            f"{exc}"
        )

        return audio


# =============================================================================
# SEED MODE
# =============================================================================

def run_seed_mode(
    args,
    text,
):

    url = (
        args.server.rstrip("/")
        + "/tts_seed"
    )


    payload = {

        "text":
            text,

        "seed":
            args.seed,

        "max_new_tokens":
            args.max_new_tokens
            or 1024,
    }


    started = (
        time.perf_counter()
    )


    session = (
        requests.Session()
    )

    session.trust_env = False


    try:

        response = session.post(
            url,
            json=payload,
            timeout=args.timeout,
        )


    except requests.RequestException as exc:

        print(
            f"Request failed: {exc}"
        )

        session.close()

        sys.exit(1)


    if response.status_code != 200:

        print(
            response.text
        )

        session.close()

        sys.exit(1)


    output_path = Path(
        args.output
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    output_path.write_bytes(
        response.content
    )


    total_ms = (
        time.perf_counter()
        - started
    ) * 1000


    print("")
    print("=" * 80)
    print("SEED RESULT")
    print("=" * 80)

    print(
        f"Seed              : "
        f"{args.seed}"
    )

    print(
        f"Saved             : "
        f"{output_path}"
    )

    print(
        f"Client total      : "
        f"{total_ms:.2f} ms"
    )

    print("=" * 80)


    if args.play:

        try:

            import sounddevice as sd


            audio, sr = sf.read(
                str(output_path),
                dtype="float32",
            )


            audio = adjust_speed(
                audio,
                args.speed,
            )


            sd.play(
                audio,
                sr,
            )


            sd.wait()


        except Exception as exc:

            print(
                f"Playback failed: {exc}"
            )


    session.close()


# =============================================================================
# PRODUCTION
# =============================================================================

def run_production(
    args,
    text,
):

    if not args.agent_audio:

        print(
            "Missing --agent-audio"
        )

        sys.exit(1)


    if not args.customer_audio:

        print(
            "Missing --customer-audio"
        )

        sys.exit(1)


    agent_audio = Path(
        args.agent_audio
    )


    customer_audio = Path(
        args.customer_audio
    )


    if not agent_audio.exists():

        print(
            f"Agent WAV not found: "
            f"{agent_audio}"
        )

        sys.exit(1)


    if not customer_audio.exists():

        print(
            f"Customer WAV not found: "
            f"{customer_audio}"
        )

        sys.exit(1)


    agent_text = (
        resolve_reference_text(
            args.agent_text,
            args.agent_text_file,
            "agent",
        )
    )


    customer_text = (
        resolve_reference_text(
            args.customer_text,
            args.customer_text_file,
            "customer",
        )
    )


    url = (
        args.server.rstrip("/")
        + "/tts"
    )


    max_new_tokens = (
        args.max_new_tokens
        or 3072
    )


    print("")
    print("=" * 80)
    print("DIA PRODUCTION")
    print("=" * 80)

    print(
        f"Server            : "
        f"{url}"
    )

    print(
        f"Agent WAV         : "
        f"{agent_audio}"
    )

    print(
        f"Customer WAV      : "
        f"{customer_audio}"
    )

    print(
        f"Max new tokens    : "
        f"{max_new_tokens}"
    )

    print(
        f"Output            : "
        f"{args.output}"
    )

    print("=" * 80)


    # =========================================================================
    # PLAYBACK
    # =========================================================================

    playback_queue = (
        queue.Queue()
    )


    playback_thread = None


    if args.play:

        try:

            import sounddevice as sd


            def playback_worker():

                while True:

                    item = (
                        playback_queue.get()
                    )


                    if item is None:

                        playback_queue.task_done()

                        break


                    (
                        index,
                        count,
                        speaker,
                        voice,
                        audio,
                        sr,
                    ) = item


                    print(
                        f"[PLAY] "
                        f"{index}/{count} "
                        f"{speaker} "
                        f"({voice})"
                    )


                    sd.play(
                        adjust_speed(
                            audio,
                            args.speed,
                        ),
                        sr,
                    )


                    # Strict sequential playback.
                    sd.wait()


                    playback_queue.task_done()


            playback_thread = (
                threading.Thread(
                    target=
                        playback_worker,
                    daemon=True,
                )
            )


            playback_thread.start()


        except Exception as exc:

            print(
                f"Playback disabled: "
                f"{exc}"
            )

            args.play = False


    # =========================================================================
    # HTTP
    # =========================================================================

    session = (
        requests.Session()
    )

    session.trust_env = False


    request_start = (
        time.perf_counter()
    )


    try:

        with (
            open(
                agent_audio,
                "rb",
            ) as agent_file,

            open(
                customer_audio,
                "rb",
            ) as customer_file,
        ):


            response = session.post(

                url,

                data={

                    "text":
                        text,

                    "agent_reference_text":
                        agent_text,

                    "customer_reference_text":
                        customer_text,

                    "max_new_tokens":
                        str(
                            max_new_tokens
                        ),
                },

                files={

                    "agent_audio": (
                        agent_audio.name,
                        agent_file,
                        "audio/wav",
                    ),

                    "customer_audio": (
                        customer_audio.name,
                        customer_file,
                        "audio/wav",
                    ),
                },

                stream=True,

                timeout=
                    args.timeout,
            )


    except requests.RequestException as exc:

        session.close()

        print(
            f"Request failed: {exc}"
        )

        sys.exit(1)


    headers_received = (
        time.perf_counter()
    )


    if response.status_code != 200:

        print(
            response.text
        )

        response.close()

        session.close()

        sys.exit(1)


    # =========================================================================
    # FINAL WAV
    #
    # IMPORTANT:
    #
    # Every returned audio chunk is written EXACTLY AS RECEIVED.
    #
    # No trimming.
    # No inserted silence.
    # No crossfade.
    # =============================================================================

    output_path = Path(
        args.output
    )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    writer = sf.SoundFile(

        str(
            output_path
        ),

        mode="w",

        samplerate=44100,

        channels=1,

        subtype="PCM_16",

        format="WAV",
    )


    first_audio_time = None

    chunk_counter = 0

    server_metrics = {}


    try:

        while True:

            metadata, pcm_bytes = (
                read_frame(
                    response.raw
                )
            )


            frame_type = (
                metadata.get(
                    "type"
                )
            )


            if frame_type == "end":

                server_metrics = (
                    metadata
                )

                break


            if frame_type != "audio":

                continue


            if first_audio_time is None:

                first_audio_time = (
                    time.perf_counter()
                )


            pcm = np.frombuffer(
                pcm_bytes,
                dtype="<i2",
            )


            audio = (
                pcm.astype(
                    np.float32
                )
                / 32767.0
            )


            sample_rate = int(
                metadata.get(
                    "sample_rate",
                    44100,
                )
            )


            # =============================================================
            # DIRECT APPEND
            #
            # This is exactly the old behavior you wanted.
            # =============================================================

            writer.write(
                audio
            )


            chunk_counter += 1


            print("")
            print(
                f"[RECV] "
                f"{metadata.get('chunk_index')}/"
                f"{metadata.get('chunk_count')}"
            )

            print(
                f"Speaker           : "
                f"{metadata.get('speaker')} "
                f"({metadata.get('voice')})"
            )

            print(
                f"Text              : "
                f"{metadata.get('chunk_text')}"
            )

            print(
                f"Words             : "
                f"{metadata.get('chunk_words')}"
            )

            print(
                f"Audio duration    : "
                f"{metadata.get('audio_duration_s', 0):.2f}s"
            )

            print(
                f"Min new tokens    : "
                f"{metadata.get('min_new_tokens')}"
            )

            print(
                f"Local max tokens  : "
                f"{metadata.get('local_max_new_tokens')}"
            )

            print(
                f"Retried           : "
                f"{metadata.get('retried')}"
            )


            # =============================================================
            # PLAY EXACT SAME AUDIO
            # =============================================================

            if args.play:

                playback_queue.put(
                    (
                        metadata.get(
                            "chunk_index"
                        ),

                        metadata.get(
                            "chunk_count"
                        ),

                        metadata.get(
                            "speaker"
                        ),

                        metadata.get(
                            "voice"
                        ),

                        audio.copy(),

                        sample_rate,
                    )
                )


    finally:

        writer.close()

        response.close()

        session.close()


    receive_end = (
        time.perf_counter()
    )


    # =========================================================================
    # WAIT FOR PLAYBACK
    # =========================================================================

    if args.play:

        playback_queue.join()

        playback_queue.put(
            None
        )

        playback_queue.join()


        if playback_thread:

            playback_thread.join(
                timeout=10
            )


    # =========================================================================
    # METRICS
    # =========================================================================

    ttfb_ms = (
        headers_received
        - request_start
    ) * 1000


    ttfa_ms = (
        (
            first_audio_time
            - request_start
        )
        * 1000

        if first_audio_time

        else 0.0
    )


    receive_total_ms = (
        receive_end
        - request_start
    ) * 1000


    print("")
    print("=" * 80)
    print("CLIENT LATENCY")
    print("=" * 80)

    print(
        f"TTFB              : "
        f"{ttfb_ms:.2f} ms"
    )

    print(
        f"TTFA              : "
        f"{ttfa_ms:.2f} ms"
    )

    print(
        f"Receive total     : "
        f"{receive_total_ms:.2f} ms"
    )

    print(
        f"Chunks received   : "
        f"{chunk_counter}"
    )


    print("")
    print("=" * 80)
    print("SERVER METRICS")
    print("=" * 80)

    print(
        f"Chunks            : "
        f"{server_metrics.get('chunk_count', 0)}"
    )

    print(
        f"Inference         : "
        f"{server_metrics.get('inference_ms', 0):.2f} ms"
    )

    print(
        f"Server total      : "
        f"{server_metrics.get('server_total_ms', 0):.2f} ms"
    )

    print(
        f"Audio duration    : "
        f"{server_metrics.get('audio_duration_s', 0):.2f}s"
    )

    print(
        f"Generation RTF    : "
        f"{server_metrics.get('generation_rtf', 0):.4f}"
    )

    print(
        f"GPU peak          : "
        f"{server_metrics.get('gpu_peak_mb', 0):.2f} MB"
    )


    print("")
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(
        f"Saved             : "
        f"{output_path}"
    )

    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--server",
        default=
            "http://localhost:8000",
    )


    input_group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )


    input_group.add_argument(
        "--text"
    )


    input_group.add_argument(
        "--text-file"
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )


    parser.add_argument(
        "--agent-audio"
    )


    parser.add_argument(
        "--customer-audio"
    )


    parser.add_argument(
        "--agent-text"
    )


    parser.add_argument(
        "--agent-text-file"
    )


    parser.add_argument(
        "--customer-text"
    )


    parser.add_argument(
        "--customer-text-file"
    )


    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
    )


    parser.add_argument(
        "--output",
        default=
            "output.wav",
    )


    parser.add_argument(
        "--play",
        action=
            "store_true",
    )


    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
    )


    parser.add_argument(
        "--timeout",
        type=float,
        default=1800,
    )


    args = parser.parse_args()


    if args.speed <= 0:

        print(
            "--speed must be > 0"
        )

        sys.exit(1)


    text = resolve_text(
        args
    )


    if args.seed is not None:

        run_seed_mode(
            args,
            text,
        )

    else:

        run_production(
            args,
            text,
        )


if __name__ == "__main__":

    main()
