import math
import platform
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


@unittest.skipUnless(platform.system() == "Darwin", "CoreML runs only on macOS")
class CoreMLStreamingIntegrationTests(unittest.TestCase):
    def test_streaming_matches_batch_on_generated_speech(self):
        import soundfile as sf
        from scipy.signal import resample_poly

        import senko

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speech.aiff"
            subprocess.run(
                ["say", "-o", str(path), "Please announce the next flight to Paris. The gate is now open."],
                check=True,
            )
            speech, sample_rate = sf.read(path, dtype="float32")

        if speech.ndim == 2:
            speech = speech.mean(axis=1)
        factor = math.gcd(sample_rate, 16000)
        speech = resample_poly(speech, 16000 // factor, sample_rate // factor).astype(np.float32)
        peak = float(np.max(np.abs(speech)))
        self.assertGreater(peak, 0, "macOS speech synthesis produced silent audio")
        speech *= 0.9 / peak
        repeats = math.ceil((35 * 16000) / len(speech))
        audio = np.concatenate((np.zeros(5 * 16000, dtype=np.float32), np.tile(speech, repeats)))

        diarizer = senko.Diarizer(device="coreml", vad="pyannote", warmup=False, quiet=True)
        from senko.streaming import _IncrementalDiarizationEngine
        stream = _IncrementalDiarizationEngine(diarizer)
        previous = 0
        for seconds in (4.9, 5, 5.1, 9.9, 10, 10.1, 20, 30, 40):
            cutoff = round(seconds * 16000)
            stream.append(audio[previous:cutoff])
            previous = cutoff
            update = stream.update()
            self.assertLessEqual(stream._total_samples - stream._audio_start_samples, 30 * 16000)
            if seconds >= 30:
                self.assertGreater(stream._audio_start_samples, 0)
            actual = update["result"]
            expected = diarizer.diarize_samples(audio[:cutoff])
            self.assertEqual(actual is None, expected is None)
            if seconds >= 20:
                self.assertGreater(update["cache_stats"]["reused_vad_windows"], 0)
            if expected is None:
                continue
            for key in ("vad", "raw_segments", "merged_segments", "raw_speakers_detected", "merged_speakers_detected"):
                self.assertEqual(actual[key], expected[key], f"{key} mismatch at {seconds}s")
            self.assertEqual(actual["speaker_centroids"].keys(), expected["speaker_centroids"].keys())
            for key in expected["speaker_centroids"]:
                with self.subTest(seconds=seconds, speaker=key, cache=update["cache_stats"]):
                    np.testing.assert_allclose(
                        actual["speaker_centroids"][key], expected["speaker_centroids"][key], rtol=1e-4, atol=1e-3
                    )

        self.assertIsNotNone(actual, "Generated speech was not detected")


if __name__ == "__main__":
    unittest.main()
