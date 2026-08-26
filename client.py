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
# TEXT
# =============================================================================

def read_text_file(path_value):

    path = Path(path_value)

    if not path.exists():

        print(
            f"File not found: {path}"
        )

        sys.exit(1)

    return path.read_text(
        encoding="utf-8"
    ).strip()


def resolve_text(args):

    if args.text:

        return args.text.strip()

    return read_text_file(
        args.text_file
    )


def resolve_reference_text(
    direct,
    file_path,
    name,
):

    if direct:

        return direct.strip()

    if file_path:

        return read_text_file(
            file_path
        )

    print(
        f"Missing {name} reference text"
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
                "Stream ended unexpectedly"
            )

        data.extend(part)

    return bytes(data)


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


    pcm = (
        receive_exact(
            raw,
            pcm_size,
        )
        if pcm_size
        else b""
    )


    return (
        metadata,
        pcm,
    )


# =============================================================================
# TRIM EDGE SILENCE
# =============================================================================

def trim_chunk_silence(
    audio,
    sample_rate,
    threshold_db=-45.0,
    padding_ms=8.0,
):

    if len(audio) == 0:

        return audio


    peak = float(
        np.max(
            np.abs(audio)
        )
    )


    if peak <= 1e-8:

        return audio


    threshold = (
        peak
        * (
            10.0
            ** (
                threshold_db
                / 20.0
            )
        )
    )


    active = np.flatnonzero(
        np.abs(audio)
        >= threshold
    )


    if len(active) == 0:

        return audio


    padding = int(
        sample_rate
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

        return librosa.effects.time_stretch(
            audio,
            rate=speed,
        )


    except Exception:

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


    response = requests.post(
        url,
        json=payload,
        timeout=args.timeout,
    )


    if response.status_code != 200:

        print(
            response.text
        )

        sys.exit(1)


    Path(
        args.output
    ).write_bytes(
        response.content
    )


    print(
        f"Saved: {args.output}"
    )


# =============================================================================
# PRODUCTION
# =============================================================================

def run_production(
    args,
    text,
):

    agent_audio = Path(
        args.agent_audio
    )

    customer_audio = Path(
        args.customer_audio
    )


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


    playback_queue = queue.Queue()


    if args.play:

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
                    f"{speaker} ({voice})"
                )


                sd.play(
                    adjust_speed(
                        audio,
                        args.speed,
                    ),
                    sr,
                )


                # Strictly sequential playback.
                sd.wait()


                playback_queue.task_done()


        thread = threading.Thread(
            target=playback_worker,
            daemon=True,
        )

        thread.start()


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


        response = requests.post(

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
                        args.max_new_tokens
                        or 3072
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


    if response.status_code != 200:

        print(
            response.text
        )

        sys.exit(1)


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


    chunk_count = 0


    try:

        while True:

            metadata, pcm_bytes = read_frame(
                response.raw
            )


            if metadata.get(
                "type"
            ) == "end":

                break


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


            sr = int(
                metadata.get(
                    "sample_rate",
                    44100,
                )
            )


            original_duration = (
                len(audio)
                / sr
            )


            # =============================================================
            # REMOVE ONLY LEADING / TRAILING SILENCE
            # =============================================================

            if args.trim_silence:

                audio = trim_chunk_silence(

                    audio,

                    sr,

                    threshold_db=
                        args.trim_threshold_db,

                    padding_ms=
                        args.trim_padding_ms,
                )


            final_duration = (
                len(audio)
                / sr
            )


            # =============================================================
            # DIRECT APPEND
            #
            # No silence inserted between chunks.
            # =============================================================

            writer.write(
                audio
            )


            chunk_count += 1


            print("")
            print(
                f"[RECV] "
                f"{metadata.get('chunk_index')}/"
                f"{metadata.get('chunk_count')}"
            )

            print(
                f"Speaker   : "
                f"{metadata.get('speaker')} "
                f"({metadata.get('voice')})"
            )

            print(
                f"Text      : "
                f"{metadata.get('chunk_text')}"
            )

            print(
                f"Generated : "
                f"{original_duration:.2f}s"
            )

            print(
                f"Trimmed   : "
                f"{final_duration:.2f}s"
            )

            print(
                f"Min tokens: "
                f"{metadata.get('min_new_tokens')}"
            )

            print(
                f"Max tokens: "
                f"{metadata.get('local_max_new_tokens')}"
            )

            print(
                f"Retry     : "
                f"{metadata.get('retried')}"
            )


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

                        sr,
                    )
                )


    finally:

        writer.close()


    if args.play:

        playback_queue.join()

        playback_queue.put(
            None
        )

        playback_queue.join()


    print("")
    print(
        f"Saved final WAV: "
        f"{output_path}"
    )

    print(
        f"Chunks appended: "
        f"{chunk_count}"
    )


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
        default="output.wav",
    )


    parser.add_argument(
        "--play",
        action="store_true",
    )


    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
    )


    parser.add_argument(
        "--trim-silence",
        action=
            argparse.BooleanOptionalAction,
        default=True,
    )


    parser.add_argument(
        "--trim-threshold-db",
        type=float,
        default=-45.0,
    )


    parser.add_argument(
        "--trim-padding-ms",
        type=float,
        default=8.0,
    )


    parser.add_argument(
        "--timeout",
        type=float,
        default=1800,
    )


    args = parser.parse_args()


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
