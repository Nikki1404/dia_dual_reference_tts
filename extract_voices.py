#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_two_turn_reference(path: Path):
    text = re.sub(r"\s+", " ", path.read_text(encoding="utf-8")).strip()
    matches = list(re.finditer(r"\[(S1|S2)\]\s*", text, flags=re.IGNORECASE))
    turns = []
    for i, match in enumerate(matches):
        speaker = match.group(1).upper()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        speech = text[start:end].strip()
        if speech:
            turns.append((speaker, speech))
    if len(turns) != 2 or turns[0][0] != "S1" or turns[1][0] != "S2":
        raise ValueError("reference.txt must contain exactly: [S1] agent text [S2] customer text")
    return turns


def load_mono(path: Path):
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float32), sr


def rms_frames(audio, sr):
    frame_size = max(1, int(sr * 0.040))
    hop_size = max(1, int(sr * 0.010))
    values, positions = [], []
    for start in range(0, max(1, len(audio) - frame_size), hop_size):
        frame = audio[start:start + frame_size]
        if len(frame) == 0:
            continue
        values.append(np.sqrt(np.mean(frame * frame) + 1e-12))
        positions.append(start + len(frame) // 2)
    return np.asarray(values), np.asarray(positions), hop_size


def find_boundary(audio, sr, s1_text, s2_text):
    s1_words = max(1, len(s1_text.split()))
    s2_words = max(1, len(s2_text.split()))
    expected_sample = int(len(audio) * s1_words / (s1_words + s2_words))

    rms, positions, hop_size = rms_frames(audio, sr)
    if len(rms) == 0:
        return expected_sample

    radius = int(len(audio) * 0.20)
    mask = (
        (positions >= max(0, expected_sample - radius))
        & (positions <= min(len(audio), expected_sample + radius))
    )
    if not np.any(mask):
        return expected_sample

    local_rms = rms[mask]
    local_positions = positions[mask]
    smooth_count = max(1, int(0.120 / (hop_size / sr)))
    kernel = np.ones(smooth_count, dtype=np.float32) / smooth_count
    smoothed = np.convolve(local_rms, kernel, mode="same")
    return int(local_positions[int(np.argmin(smoothed))])


def trim_outer_silence(audio, sr):
    if len(audio) == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return audio
    active = np.flatnonzero(np.abs(audio) >= peak * 0.02)
    if len(active) == 0:
        return audio
    padding = int(sr * 0.040)
    start = max(0, int(active[0]) - padding)
    end = min(len(audio), int(active[-1]) + padding + 1)
    return audio[start:end]


def split_seed_wav(wav_path, s1_text, s2_text):
    audio, sr = load_mono(wav_path)
    boundary = find_boundary(audio, sr, s1_text, s2_text)
    gap = int(sr * 0.030)
    s1 = audio[:max(0, boundary - gap)]
    s2 = audio[min(len(audio), boundary + gap):]
    return (
        trim_outer_silence(s1, sr),
        trim_outer_silence(s2, sr),
        sr,
        boundary / sr,
    )


def main():
    parser = argparse.ArgumentParser(description="Extract S1 agent and S2 customer clips from seed WAVs")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--agent-file", required=True)
    parser.add_argument("--output-dir", default="extracted_voices")
    parser.add_argument("--pattern", default="seed_*.wav")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    customer_dir = output_dir / "customers"
    output_dir.mkdir(parents=True, exist_ok=True)
    customer_dir.mkdir(parents=True, exist_ok=True)

    turns = parse_two_turn_reference(Path(args.reference_text))
    s1_text, s2_text = turns[0][1], turns[1][1]

    print("\n" + "=" * 80)
    print("REFERENCE")
    print("=" * 80)
    print(f"S1 / Agent text   : {s1_text}")
    print(f"S2 / Customer text: {s2_text}")
    print("=" * 80)

    wav_files = sorted(input_dir.glob(args.pattern))
    if not wav_files:
        raise FileNotFoundError(f"No WAVs found in {input_dir} matching {args.pattern}")

    created = 0
    for wav_path in wav_files:
        try:
            _, customer_audio, sr, boundary_s = split_seed_wav(wav_path, s1_text, s2_text)
            output_customer = customer_dir / f"customer_{wav_path.stem}.wav"
            sf.write(str(output_customer), customer_audio, sr, subtype="PCM_16")
            created += 1
            print(f"[CUSTOMER] {wav_path.name} -> {output_customer.name} | boundary={boundary_s:.2f}s")
        except Exception as exc:
            print(f"[FAILED] {wav_path.name}: {exc}")

    supplied = Path(args.agent_file)
    candidates = [supplied, input_dir / supplied, input_dir / supplied.name]
    agent_path = next((c for c in candidates if c.exists()), None)
    if agent_path is None:
        raise FileNotFoundError("Agent WAV not found. Tried:\n" + "\n".join(str(c) for c in candidates))

    agent_audio, _, sr, _ = split_seed_wav(agent_path, s1_text, s2_text)
    final_agent = output_dir / "agent.wav"
    sf.write(str(final_agent), agent_audio, sr, subtype="PCM_16")

    (output_dir / "agent.txt").write_text(s1_text + "\n", encoding="utf-8")
    (output_dir / "customer.txt").write_text(s2_text + "\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Agent output      : {final_agent}")
    print(f"Customers created : {created}")
    print(f"Customer folder   : {customer_dir}")
    print(f"Agent text        : {output_dir / 'agent.txt'}")
    print(f"Customer text     : {output_dir / 'customer.txt'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
