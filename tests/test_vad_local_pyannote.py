import builtins
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from senko import config
from senko.vad_local_pyannote import LocalSegmentationVADCuda
from senko.vad_local_pyannote.postprocess import build_powerset_mapping, scores_to_segments


EXPECTED_SYNTHETIC_SEGMENTS = [
    (1.0603437500000001, 4.3678437500000005),
    (4.992218750000001, 10.290968750000001),
]


def _synthetic_waveform(duration_seconds: float = 12.0, sample_rate: int = 16000) -> np.ndarray:
    total_samples = int(duration_seconds * sample_rate)
    timeline = np.arange(total_samples, dtype=np.float32) / sample_rate
    waveform = np.zeros_like(timeline)

    def add_voiced_region(start: float, end: float, fundamental: float):
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        t = timeline[start_idx:end_idx] - start
        envelope = np.sin(np.pi * np.clip(t / max(end - start, 1e-6), 0.0, 1.0)) ** 2
        voiced = (
            0.45 * np.sin(2 * np.pi * fundamental * t)
            + 0.22 * np.sin(2 * np.pi * (fundamental * 2.1) * t)
            + 0.1 * np.sin(2 * np.pi * (fundamental * 3.2) * t)
        )
        waveform[start_idx:end_idx] += envelope * voiced

    add_voiced_region(1.0, 4.2, 145.0)
    add_voiced_region(5.0, 8.4, 180.0)
    add_voiced_region(8.55, 10.0, 165.0)
    return np.ascontiguousarray(np.clip(waveform, -1.0, 1.0), dtype=np.float32)


class PostprocessTests(unittest.TestCase):
    def test_no_speech_scores_return_empty_segments(self):
        scores = np.zeros(32, dtype=np.float32)
        segments = scores_to_segments(scores, frame_start=0.0, frame_duration=0.0619375, frame_step=0.016875)
        self.assertEqual(segments, [])

    def test_short_speech_burst_is_removed(self):
        scores = np.zeros(64, dtype=np.float32)
        scores[10:18] = 1.0
        segments = scores_to_segments(scores, frame_start=0.0, frame_duration=0.0619375, frame_step=0.016875)
        self.assertEqual(segments, [])

    def test_short_silence_gap_is_merged(self):
        scores = np.zeros(128, dtype=np.float32)
        scores[10:30] = 1.0
        scores[34:56] = 1.0
        segments = scores_to_segments(scores, frame_start=0.0, frame_duration=0.0619375, frame_step=0.016875)
        self.assertEqual(len(segments), 1)

    def test_overlap_powerset_class_counts_as_speech(self):
        mapping = build_powerset_mapping(3, 2)
        self.assertEqual(mapping.shape, (7, 3))
        self.assertEqual(mapping[4].tolist(), [1.0, 1.0, 0.0])


class ChunkingTests(unittest.TestCase):
    def test_short_audio_is_padded_to_full_window(self):
        import torch

        backend = LocalSegmentationVADCuda.__new__(LocalSegmentationVADCuda)
        backend.torch = torch
        backend.batch_size = 8
        backend.window_size = 6
        backend.step_size = 4

        waveform = np.arange(4, dtype=np.float32)
        batches = list(backend._iter_chunks(waveform))

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].shape, (1, 1, 6))
        self.assertEqual(batches[0][0, 0].tolist(), [0.0, 1.0, 2.0, 3.0, 0.0, 0.0])

    def test_tail_chunk_is_anchored_to_end_for_non_aligned_length(self):
        import torch

        backend = LocalSegmentationVADCuda.__new__(LocalSegmentationVADCuda)
        backend.torch = torch
        backend.batch_size = 8
        backend.window_size = 6
        backend.step_size = 4

        waveform = np.arange(15, dtype=np.float32)
        batches = list(backend._iter_chunks(waveform))

        self.assertEqual(len(batches), 2)
        regular = batches[0]
        tail = batches[1]

        self.assertEqual(regular.shape, (3, 1, 6))
        self.assertEqual(regular[:, 0, 0].tolist(), [0.0, 4.0, 8.0])
        self.assertEqual(tail.shape, (1, 1, 6))
        self.assertEqual(tail[0, 0].tolist(), [9.0, 10.0, 11.0, 12.0, 13.0, 14.0])

    def test_exact_alignment_does_not_emit_duplicate_tail_chunk(self):
        import torch

        backend = LocalSegmentationVADCuda.__new__(LocalSegmentationVADCuda)
        backend.torch = torch
        backend.batch_size = 8
        backend.window_size = 6
        backend.step_size = 4

        waveform = np.arange(14, dtype=np.float32)
        batches = list(backend._iter_chunks(waveform))

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].shape, (3, 1, 6))
        self.assertEqual(batches[0][:, 0, 0].tolist(), [0.0, 4.0, 8.0])


class LocalPyannoteCudaRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for local pyannote regression tests.")

        cls.torch = torch
        cls.model_paths = config.resolve_model_paths(required_fields=config.RUNTIME_PYANNOTE_CUDA_MODEL_FIELDS)
        cls.local_backend = LocalSegmentationVADCuda(
            checkpoint_path=cls.model_paths.pyannote_segmentation_senko_model_path,
            torch_device=torch.device("cuda"),
        )

    def _assert_segments_match_fixture(self, segments):
        self.assertEqual(len(segments), len(EXPECTED_SYNTHETIC_SEGMENTS))
        for (actual_start, actual_end), (expected_start, expected_end) in zip(segments, EXPECTED_SYNTHETIC_SEGMENTS):
            self.assertAlmostEqual(actual_start, expected_start, places=6)
            self.assertAlmostEqual(actual_end, expected_end, places=6)

    def test_cuda_backend_matches_regression_fixture_for_ndarray_input(self):
        segments = self.local_backend.process(_synthetic_waveform())
        self._assert_segments_match_fixture(segments)

    def test_cuda_backend_matches_regression_fixture_for_wav_input(self):
        waveform = _synthetic_waveform()
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "sample.wav"
            sf.write(wav_path, waveform, 16000, subtype="PCM_16")
            segments = self.local_backend.process(str(wav_path))
        self._assert_segments_match_fixture(segments)

    def test_cuda_backend_does_not_import_pyannote_at_runtime(self):
        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "pyannote" or name.startswith("pyannote."):
                raise AssertionError("Local VAD runtime attempted to import pyannote.")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=guarded_import):
            backend = LocalSegmentationVADCuda(
                checkpoint_path=self.model_paths.pyannote_segmentation_senko_model_path,
                torch_device=self.torch.device("cuda"),
            )
            segments = backend.process(_synthetic_waveform())

        self._assert_segments_match_fixture(segments)


if __name__ == "__main__":
    unittest.main()
