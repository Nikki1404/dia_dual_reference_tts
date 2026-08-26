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

def read_text_file(path_value: str) -> str:

    path = Path(path_value)

    if not path.exists():

        print(
            f"File not found: {path}"
        )

        sys.exit(1)

    return path.read_text(
        encoding="utf-8"
    ).strip()


def resolve_main_text(args) -> str:

    if args.text is not None:

        value = args.text.strip()

    elif args.text_file is not None:

        value = read_text_file(
            args.text_file
        )

    else:

        print(
            "Provide --text or --text-file."
        )

        sys.exit(1)

    if not value:

        print(
            "Input text is empty."
        )

        sys.exit(1)

    return value


def resolve_reference_text(
    direct_value,
    file_value,
    name,
) -> str:

    if direct_value is not None:

        value = direct_value.strip()

    elif file_value is not None:

        value = read_text_file(
            file_value
        )

    else:

        print(
            f"Production mode requires "
            f"{name} reference text."
        )

        sys.exit(1)

    if not value:

        print(
            f"{name} reference text is empty."
        )

        sys.exit(1)

    return value


# =============================================================================
# STREAM HELPERS
# =============================================================================

def receive_exact(
    raw,
    size: int,
) -> bytes:

    data = bytearray()

    while len(data) < size:

        part = raw.read(
            size - len(data)
        )

        if not part:

            raise EOFError(
                "Server stream ended unexpectedly."
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

    pcm_bytes = b""

    if pcm_size > 0:

        pcm_bytes = receive_exact(
            raw,
            pcm_size,
        )

    return (
        metadata,
        pcm_bytes,
    )


# =============================================================================
# SPEED
# =============================================================================

def adjust_speed(
    audio: np.ndarray,
    speed: float,
) -> np.ndarray:

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
    text: str,
):

    url = (
        args.server.rstrip("/")
        + "/tts_seed"
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else 1024
    )

    payload = {

        "text":
            text,

        "seed":
            args.seed,

        "max_new_tokens":
            max_new_tokens,
    }


    print("")
    print("=" * 80)
    print("DIA SEED TEST MODE")
    print("=" * 80)

    print(
        f"Server            : "
        f"{url}"
    )

    print(
        f"Seed              : "
        f"{args.seed}"
    )

    print(
        f"Words             : "
        f"{len(text.split())}"
    )

    print(
        f"Max new tokens    : "
        f"{max_new_tokens}"
    )

    print(
        f"Output            : "
        f"{output_path}"
    )

    print(
        f"Play              : "
        f"{args.play}"
    )

    print("=" * 80)


    started = (
        time.perf_counter()
    )


    session = requests.Session()

    # Prevent localhost from accidentally going
    # through configured corporate proxy.
    session.trust_env = False


    try:

        response = session.post(
            url,
            json=payload,
            timeout=args.timeout,
        )

    except requests.RequestException as exc:

        print(
            f"Request failed: "
            f"{exc}"
        )

        session.close()

        sys.exit(1)


    if response.status_code != 200:

        print(
            f"Request failed "
            f"({response.status_code}): "
            f"{response.text}"
        )

        session.close()

        sys.exit(1)


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

    print(
        f"Server preprocess : "
        f"{response.headers.get('X-Preprocess-Time-MS', '0')} ms"
    )

    print(
        f"Server inference  : "
        f"{response.headers.get('X-Inference-Time-MS', '0')} ms"
    )

    print(
        f"Server decode     : "
        f"{response.headers.get('X-Decode-Time-MS', '0')} ms"
    )

    print(
        f"Server total      : "
        f"{response.headers.get('X-Server-Total-MS', '0')} ms"
    )

    print(
        f"Audio duration    : "
        f"{response.headers.get('X-Audio-Duration-S', '0')} sec"
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

            print("")
            print(
                f"[PLAY] Playing seed "
                f"{args.seed}"
            )

            sd.play(
                audio,
                sr,
            )

            sd.wait()

            print(
                "[PLAY] Finished"
            )

        except Exception as exc:

            print(
                f"Playback failed: "
                f"{exc}"
            )


    session.close()


# =============================================================================
# PRODUCTION MODE
# =============================================================================

def run_production_mode(
    args,
    text: str,
):

    # =========================================================================
    # VALIDATE REFERENCES
    # =========================================================================

    if not args.agent_audio:

        print(
            "Production mode requires "
            "--agent-audio."
        )

        sys.exit(1)


    if not args.customer_audio:

        print(
            "Production mode requires "
            "--customer-audio."
        )

        sys.exit(1)


    agent_audio_path = Path(
        args.agent_audio
    )

    customer_audio_path = Path(
        args.customer_audio
    )


    for path in (
        agent_audio_path,
        customer_audio_path,
    ):

        if not path.exists():

            print(
                f"File not found: "
                f"{path}"
            )

            sys.exit(1)


    agent_reference_text = (
        resolve_reference_text(
            args.agent_text,
            args.agent_text_file,
            "agent",
        )
    )


    customer_reference_text = (
        resolve_reference_text(
            args.customer_text,
            args.customer_text_file,
            "customer",
        )
    )


    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else 3072
    )


    if not (
        5
        <= args.chunk_words
        <= 100
    ):

        print(
            "--chunk-words must be "
            "between 5 and 100."
        )

        sys.exit(1)


    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    url = (
        args.server.rstrip("/")
        + "/tts"
    )


    print("")
    print("=" * 80)
    print("DIA PRODUCTION TTS")
    print("=" * 80)

    print(
        f"Server            : "
        f"{url}"
    )

    print(
        f"Input words       : "
        f"{len(text.split())}"
    )

    print(
        f"Chunk words       : "
        f"{args.chunk_words}"
    )

    print(
        f"Agent WAV         : "
        f"{agent_audio_path}"
    )

    print(
        f"Customer WAV      : "
        f"{customer_audio_path}"
    )

    print(
        f"Max new tokens    : "
        f"{max_new_tokens}"
    )

    print(
        f"Output            : "
        f"{output_path}"
    )

    print(
        f"Play              : "
        f"{args.play}"
    )

    print(
        f"Speed             : "
        f"{args.speed}"
    )

    print("=" * 80)


    # =========================================================================
    # PLAYBACK QUEUE
    #
    # IMPORTANT:
    #
    # New word-based chunks may contain BOTH:
    #
    # [S1] ...
    # [S2] ...
    #
    # Therefore there is NO single metadata["speaker"] anymore.
    # =========================================================================

    playback_queue = (
        queue.Queue()
    )

    playback_enabled = False

    playback_thread = None


    if args.play:

        try:

            import sounddevice as sd

            playback_enabled = True


            def playback_worker():

                while True:

                    item = (
                        playback_queue.get()
                    )


                    if item is None:

                        playback_queue.task_done()

                        break


                    (
                        chunk_index,
                        chunk_count,
                        audio,
                        sample_rate,
                    ) = item


                    playback_audio = (
                        adjust_speed(
                            audio,
                            args.speed,
                        )
                    )


                    print("")
                    print(
                        f"[PLAY] Starting chunk "
                        f"{chunk_index}/"
                        f"{chunk_count}"
                    )


                    sd.play(
                        playback_audio,
                        sample_rate,
                    )


                    # Strict sequential playback:
                    #
                    # chunk N+1 is not played
                    # until chunk N finishes.
                    sd.wait()


                    print(
                        f"[PLAY] Finished chunk "
                        f"{chunk_index}/"
                        f"{chunk_count}"
                    )


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

            playback_enabled = False


    # =========================================================================
    # HTTP REQUEST
    # =========================================================================

    request_start = (
        time.perf_counter()
    )


    session = (
        requests.Session()
    )

    session.trust_env = False


    try:

        with (
            open(
                agent_audio_path,
                "rb",
            ) as agent_handle,

            open(
                customer_audio_path,
                "rb",
            ) as customer_handle,
        ):


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

                "text":
                    text,

                "agent_reference_text":
                    agent_reference_text,

                "customer_reference_text":
                    customer_reference_text,

                "chunk_words":
                    str(
                        args.chunk_words
                    ),

                "max_new_tokens":
                    str(
                        max_new_tokens
                    ),
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

        print(
            f"Request failed: "
            f"{exc}"
        )

        sys.exit(1)


    headers_received = (
        time.perf_counter()
    )


    if response.status_code != 200:

        print(
            f"Request failed "
            f"({response.status_code}): "
            f"{response.text}"
        )

        response.close()

        session.close()

        sys.exit(1)


    # =========================================================================
    # WAV OUTPUT
    # =========================================================================

    wav_writer = sf.SoundFile(

        str(
            output_path
        ),

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


            # =================================================================
            # AUDIO FRAME
            # =================================================================

            if frame_type == "audio":

                chunk_counter += 1


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


                # =============================================================
                # Append audio sequentially into ONE WAV.
                # =============================================================

                wav_writer.write(
                    audio
                )


                audio_parts.append(
                    audio.copy()
                )


                print("")
                print(
                    f"[RECV] Chunk "
                    f"{metadata.get('chunk_index')}/"
                    f"{metadata.get('chunk_count')}"
                )


                print(
                    f"       Words      : "
                    f"{metadata.get('chunk_words')}"
                )


                print(
                    f"       Text       : "
                    f"{metadata.get('chunk_text')}"
                )


                print(
                    f"       Duration   : "
                    f"{metadata.get('audio_duration_s', 0):.2f}s"
                )


                print(
                    f"       Inference  : "
                    f"{metadata.get('inference_ms', 0):.2f}ms"
                )


                print(
                    f"       Min tokens : "
                    f"{metadata.get('min_new_tokens', 0)}"
                )


                print(
                    f"       Max tokens : "
                    f"{metadata.get('local_max_new_tokens', 0)}"
                )


                print(
                    f"       Retried    : "
                    f"{metadata.get('retried', False)}"
                )


                # =============================================================
                # PLAYBACK
                #
                # NO metadata["speaker"] here anymore.
                # =============================================================

                if playback_enabled:

                    playback_queue.put(
                        (
                            metadata.get(
                                "chunk_index"
                            ),

                            metadata.get(
                                "chunk_count"
                            ),

                            audio.copy(),

                            metadata.get(
                                "sample_rate",
                                44100,
                            ),
                        )
                    )


            # =================================================================
            # END FRAME
            # =================================================================

            elif frame_type == "end":

                server_metrics = (
                    metadata
                )

                break


            # =================================================================
            # UNKNOWN FRAME
            # =================================================================

            else:

                print(
                    f"[WARN] Unknown frame type: "
                    f"{frame_type}"
                )


    except EOFError as exc:

        print("")
        print(
            f"[ERROR] Stream ended early "
            f"after {chunk_counter} chunks: "
            f"{exc}"
        )

        raise


    except Exception as exc:

        print("")
        print(
            f"[ERROR] Stream receive failed: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        raise


    finally:

        wav_writer.close()

        response.close()

        session.close()


    receive_complete_time = (
        time.perf_counter()
    )


    # =========================================================================
    # WAIT FOR ALL PLAYBACK
    # =========================================================================

    if playback_enabled:

        playback_queue.join()


        playback_queue.put(
            None
        )


        playback_queue.join()


        if playback_thread is not None:

            playback_thread.join(
                timeout=10
            )


    playback_complete_time = (
        time.perf_counter()
    )


    # =========================================================================
    # OPTIONAL SPEED-ADJUSTED FILE
    # =========================================================================

    adjusted_path = None


    if (
        args.save_adjusted
        and
        args.speed != 1.0
        and
        audio_parts
    ):


        full_audio = np.concatenate(
            audio_parts
        )


        adjusted_audio = (
            adjust_speed(
                full_audio,
                args.speed,
            )
        )


        speed_name = (
            str(
                args.speed
            )
            .replace(
                ".",
                "_",
            )
        )


        adjusted_path = (
            output_path.with_name(

                output_path.stem

                + "_speed_"

                + speed_name

                + output_path.suffix
            )
        )


        sf.write(

            str(
                adjusted_path
            ),

            adjusted_audio,

            44100,

            subtype=
                "PCM_16",
        )


    # =========================================================================
    # CLIENT METRICS
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
        receive_complete_time
        - request_start
    ) * 1000


    playback_total_ms = (
        playback_complete_time
        - request_start
    ) * 1000


    print("")
    print("=" * 80)
    print("CLIENT LATENCY")
    print("=" * 80)


    print(
        f"HTTP headers/TTFB : "
        f"{ttfb_ms:.2f} ms"
    )


    print(
        f"TTFA / TTFT       : "
        f"{ttfa_ms:.2f} ms"
    )


    print(
        f"Receive total     : "
        f"{receive_total_ms:.2f} ms"
    )


    if args.play:

        print(
            f"E2E incl playback : "
            f"{playback_total_ms:.2f} ms"
        )


    print(
        f"Chunks received   : "
        f"{chunk_counter}"
    )


    # =========================================================================
    # SERVER METRICS
    # =========================================================================

    print("")
    print("=" * 80)
    print("SERVER METRICS")
    print("=" * 80)


    print(
        f"Chunks            : "
        f"{server_metrics.get('chunk_count', 0)}"
    )


    print(
        f"Chunk words       : "
        f"{server_metrics.get('chunk_words', args.chunk_words)}"
    )


    print(
        f"Preprocess        : "
        f"{server_metrics.get('preprocess_ms', 0):.2f} ms"
    )


    print(
        f"Inference         : "
        f"{server_metrics.get('inference_ms', 0):.2f} ms"
    )


    print(
        f"Decode            : "
        f"{server_metrics.get('decode_ms', 0):.2f} ms"
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
        f"Total RTF         : "
        f"{server_metrics.get('total_rtf', 0):.4f}"
    )


    print(
        f"GPU peak          : "
        f"{server_metrics.get('gpu_peak_mb', 0):.2f} MB"
    )


    # =========================================================================
    # OUTPUT
    # =========================================================================

    print("")
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)


    print(
        f"WAV               : "
        f"{output_path}"
    )


    if adjusted_path is not None:

        print(
            f"Adjusted WAV      : "
            f"{adjusted_path}"
        )


    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(

        description=(

            "Dia TTS client supporting "
            "seed auditioning and "
            "word-chunk dual-reference production TTS."
        )
    )


    parser.add_argument(

        "--server",

        default=
            "http://localhost:8000",
    )


    # =========================================================================
    # TEXT INPUT
    # =========================================================================

    input_group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )


    input_group.add_argument(

        "--text",

        help=
            "Direct [S1]/[S2] text",
    )


    input_group.add_argument(

        "--text-file",

        help=
            "TXT file containing [S1]/[S2] text",
    )


    # =========================================================================
    # SEED MODE
    # =========================================================================

    parser.add_argument(

        "--seed",

        type=int,

        default=None,
    )


    # =========================================================================
    # PRODUCTION REFERENCES
    # =========================================================================

    parser.add_argument(

        "--agent-audio",

        default=None,
    )


    parser.add_argument(

        "--customer-audio",

        default=None,
    )


    parser.add_argument(

        "--agent-text",

        default=None,
    )


    parser.add_argument(

        "--agent-text-file",

        default=None,
    )


    parser.add_argument(

        "--customer-text",

        default=None,
    )


    parser.add_argument(

        "--customer-text-file",

        default=None,
    )


    # =========================================================================
    # CHUNK SIZE
    #
    # This now controls chunk boundaries.
    #
    # Example:
    #
    # --chunk-words 30
    # =========================================================================

    parser.add_argument(

        "--chunk-words",

        type=int,

        default=30,

        help=(
            "Approximate transcript words "
            "per production generation chunk. "
            "Speaker changes do NOT force a chunk boundary."
        ),
    )


    # =========================================================================
    # GENERATION
    # =========================================================================

    parser.add_argument(

        "--max-new-tokens",

        type=int,

        default=None,
    )


    # =========================================================================
    # OUTPUT
    # =========================================================================

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

        "--save-adjusted",

        action=
            "store_true",
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

    if args.speed <= 0:

        print(
            "--speed must be greater than 0"
        )

        sys.exit(1)


    if not (
        5
        <= args.chunk_words
        <= 100
    ):

        print(
            "--chunk-words must be "
            "between 5 and 100"
        )

        sys.exit(1)


    # =========================================================================
    # INPUT
    # =========================================================================

    text = resolve_main_text(
        args
    )


    # =========================================================================
    # MODE
    # =========================================================================

    if args.seed is not None:

        run_seed_mode(
            args,
            text,
        )

    else:

        run_production_mode(
            args,
            text,
        )


if __name__ == "__main__":

    main()
