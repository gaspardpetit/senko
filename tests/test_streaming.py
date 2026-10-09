import threading
import unittest

import numpy as np

from senko.streaming import DiarizationSession, _IncrementalDiarizationEngine


class _AudioOnlyDiarizer:
    def _normalize_audio_samples(self, samples, sample_rate):
        if sample_rate != 16000:
            raise ValueError("Expected 16 kHz audio")
        return np.asarray(samples, dtype=np.float32)


class SchedulingTests(unittest.TestCase):
    def test_coreml_tail_stays_bounded_and_uses_absolute_offsets(self):
        class Backend:
            def __init__(self):
                self.offsets = []

            def reset_incremental(self):
                pass

            def process_incremental(self, audio, *, sample_offset=0):
                self.offsets.append((sample_offset, len(audio)))
                return []

        class Diarizer(_AudioOnlyDiarizer):
            device = "coreml"
            vad_model_type = "pyannote"
            vad_backend = Backend()

        stream = _IncrementalDiarizationEngine(Diarizer())
        chunk = np.zeros(15 * 16000, dtype=np.float32)
        for _ in range(1000):
            stream.append(chunk)
            stream.update()
            self.assertLessEqual(stream._total_samples - stream._audio_start_samples, 30 * 16000)
        self.assertGreater(stream._audio_start_samples, 0)
        self.assertTrue(all(offset + length == index * len(chunk)
                            for index, (offset, length) in enumerate(stream.diarizer.vad_backend.offsets, 1)))

    def test_session_returns_independent_snapshots(self):
        session = DiarizationSession(_AudioOnlyDiarizer())
        calls = []

        def process(audio):
            calls.append(len(audio))
            return {"merged_segments": [{"end": len(audio) / 16000}], "centroid": np.array([len(audio)])}, {}

        session._stream._diarize_prefix = process
        self.assertIsNone(session.get_diarization())
        first = session.add_samples(np.zeros(16000, dtype=np.float32))
        first["merged_segments"][0]["end"] = -1
        first["centroid"][0] = -1
        self.assertEqual(session.get_diarization()["merged_segments"][0]["end"], 1)
        self.assertEqual(session.get_diarization()["centroid"][0], 16000)
        second = session.add_samples(np.zeros(16000, dtype=np.float32))
        self.assertEqual(second["merged_segments"][0]["end"], 2)
        self.assertEqual(calls, [16000, 32000])

    def test_feature_touching_prefix_end_is_recomputed(self):
        class Diarizer(_AudioOnlyDiarizer):
            device = "cpu"
            vad_model_type = "test"

            def _perform_vad(self, audio):
                return [(0.0, 12.0)]

            def _generate_subsegments(self, vad, accurate):
                return [(0.0, 12.0)]

            def _extract_fbank_features(self, audio, segments, *, sample_offset=0):
                return np.array([len(audio)], dtype=np.float32), [1], [0], 1

            def _generate_embeddings(self, features, frames, offsets, dim):
                return np.array([[features[0]]], dtype=np.float32)

            def _perform_clustering(self, embeddings, segments):
                segment = [{"speaker": "SPEAKER_01", "start": 0.0, "end": 12.0}]
                return segment, segment, {"SPEAKER_01": embeddings[0]}

        stream = _IncrementalDiarizationEngine(Diarizer())
        stream.append(np.zeros(10 * 16000, dtype=np.float32))
        first = stream.update()
        stream.append(np.zeros(16000, dtype=np.float32))
        second = stream.update()
        self.assertEqual(first["cache_stats"]["new_features"], 1)
        self.assertEqual(second["cache_stats"]["new_features"], 1)
        self.assertEqual(second["cache_stats"]["new_embedding_batches"], 1)
        self.assertNotEqual(
            first["result"]["speaker_centroids"]["SPEAKER_01"][0],
            second["result"]["speaker_centroids"]["SPEAKER_01"][0],
        )

    def test_coreml_stream_uses_incremental_vad_and_resets_cache(self):
        class Backend:
            def __init__(self):
                self.resets = 0
                self.calls = []

            def reset_incremental(self):
                self.resets += 1

            def process_incremental(self, audio, *, sample_offset=0):
                self.calls.append(len(audio))
                return []

        class Diarizer(_AudioOnlyDiarizer):
            device = "coreml"
            vad_model_type = "pyannote"
            vad_backend = Backend()

        diarizer = Diarizer()
        stream = _IncrementalDiarizationEngine(diarizer)
        self.assertEqual(diarizer.vad_backend.resets, 1)
        stream.append(np.zeros(16000, dtype=np.float32))
        first = stream.update()
        stream.append(np.zeros(10 * 16000, dtype=np.float32))
        second = stream.update()
        self.assertIsNone(first["result"])
        self.assertEqual(second["cache_stats"]["new_vad_windows"], 1)
        self.assertEqual(diarizer.vad_backend.calls, [16000, 11 * 16000])

    def test_append_during_processing_is_included_in_next_update(self):
        stream = _IncrementalDiarizationEngine(_AudioOnlyDiarizer())
        started = threading.Event()
        release = threading.Event()

        def process(audio):
            started.set()
            self.assertTrue(release.wait(timeout=5))
            return {"samples": len(audio)}, {}

        stream._diarize_prefix = process
        stream.append(np.zeros(16000, dtype=np.float32))
        updates = []
        worker = threading.Thread(target=lambda: updates.append(stream.update()))
        worker.start()
        self.assertTrue(started.wait(timeout=5))
        stream.append(np.zeros(2 * 16000, dtype=np.float32))
        release.set()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(updates[0]["cutoff_seconds"], 1)
        self.assertEqual(stream.update()["cutoff_seconds"], 3)



if __name__ == "__main__":
    unittest.main()
