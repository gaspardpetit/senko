"""Compare streaming checkpoints with fresh batch diarization of each prefix.

Usage: python scripts/streaming_parity.py AUDIO.wav --vad pyannote --device cuda
"""

import argparse
import json
import time

import numpy as np
import soundfile as sf

import senko
from senko.streaming import _IncrementalDiarizationEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="16 kHz mono PCM WAV")
    parser.add_argument("--vad", choices=("pyannote", "silero"), default="silero")
    parser.add_argument("--device", choices=("cuda", "coreml", "cpu"), default="cuda")
    parser.add_argument("--accurate", choices=("auto", "yes", "no"), default="auto")
    parser.add_argument("--colors", action="store_true")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the input to exercise long-recording clustering")
    parser.add_argument("--prepend-silence", type=float, default=0.0, help="Add initial silence in seconds")
    parser.add_argument("--checkpoints", type=float, nargs="+", default=[30, 45, 60, 75, 90, 105])
    args = parser.parse_args()

    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    if sample_rate != 16000 or audio.ndim != 1:
        raise ValueError("Expected 16 kHz mono audio")
    if args.repeat < 1:
        raise ValueError("--repeat must be positive")
    if args.repeat > 1:
        audio = np.tile(audio, args.repeat)
    if args.prepend_silence < 0:
        raise ValueError("--prepend-silence must be nonnegative")
    if args.prepend_silence:
        audio = np.concatenate((np.zeros(round(args.prepend_silence * sample_rate), dtype=np.float32), audio))
    diarizer = senko.Diarizer(device=args.device, vad=args.vad, clustering="cpu", warmup=False, quiet=True)
    accurate = {"auto": None, "yes": True, "no": False}[args.accurate]
    stream = _IncrementalDiarizationEngine(diarizer, accurate=accurate, generate_colors=args.colors)

    def check(update, stream_seconds):
        cutoff = round(update["cutoff_seconds"] * sample_rate)
        started = time.perf_counter()
        expected = diarizer.diarize_samples(
            audio[:cutoff], sample_rate=sample_rate, accurate=accurate, generate_colors=args.colors
        )
        batch_seconds = time.perf_counter() - started
        actual = update["result"]

        if expected is None or actual is None:
            assert expected is actual, f"Speech/no-speech mismatch at {cutoff / sample_rate}s"
        else:
            for key in ("vad", "raw_segments", "merged_segments", "raw_speakers_detected", "merged_speakers_detected"):
                assert actual[key] == expected[key], f"{key} mismatch at {cutoff / sample_rate}s"
            if args.colors:
                assert actual["speaker_color_sets"] == expected["speaker_color_sets"]
            assert actual["speaker_centroids"].keys() == expected["speaker_centroids"].keys()
            for key, centroid in expected["speaker_centroids"].items():
                np.testing.assert_allclose(actual["speaker_centroids"][key], centroid, rtol=1e-4, atol=1e-3)

        print(json.dumps({
            "cutoff_seconds": cutoff / sample_rate,
            "stream_seconds": round(stream_seconds, 4),
            "batch_seconds": round(batch_seconds, 4),
            "raw_segments": len(actual["raw_segments"]) if actual else 0,
            "cache": update["cache_stats"],
        }), flush=True)

    previous = 0
    for seconds in args.checkpoints:
        cutoff = min(round(seconds * sample_rate), len(audio))
        if cutoff <= previous:
            continue
        stream.append(audio[previous:cutoff], sample_rate=sample_rate)
        previous = cutoff
        started = time.perf_counter()
        update = stream.update()
        check(update, time.perf_counter() - started)


if __name__ == "__main__":
    main()
