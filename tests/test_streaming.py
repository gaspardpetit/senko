import threading
import unittest

import numpy as np

from senko.streaming import DiarizationStream


class _AudioOnlyDiarizer:
    def _normalize_audio_samples(self, samples, sample_rate):
        if sample_rate != 16000:
            raise ValueError("Expected 16 kHz audio")
        return np.asarray(samples, dtype=np.float32)


class SchedulingTests(unittest.TestCase):
    def test_coreml_stream_uses_incremental_vad_and_resets_cache(self):
        class Backend:
            def __init__(self):
                self.resets = 0
                self.calls = []

            def reset_incremental(self):
                self.resets += 1

            def process_incremental(self, audio):
                self.calls.append(len(audio))
                return []

        class Diarizer(_AudioOnlyDiarizer):
            device = "coreml"
            vad_model_type = "pyannote"
            vad_backend = Backend()

        diarizer = Diarizer()
        stream = DiarizationStream(diarizer, initial_window_seconds=1, minimum_increment_seconds=1)
        self.assertEqual(diarizer.vad_backend.resets, 1)
        stream.append(np.zeros(16000, dtype=np.float32))
        first = stream.update()
        stream.append(np.zeros(10 * 16000, dtype=np.float32))
        second = stream.update()
        self.assertIsNone(first["result"])
        self.assertEqual(second["cache_stats"]["new_vad_windows"], 1)
        self.assertEqual(diarizer.vad_backend.calls, [16000, 11 * 16000])

    def test_initial_window_and_backlog_coalescing(self):
        stream = DiarizationStream(_AudioOnlyDiarizer(), initial_window_seconds=30, minimum_increment_seconds=15)
        processed = []

        def process(audio):
            processed.append(len(audio))
            return {"samples": len(audio)}, {}

        stream._diarize_prefix = process
        stream.append(np.zeros(20 * 16000, dtype=np.float32))
        self.assertIsNone(stream.update())
        stream.append(np.zeros(10 * 16000, dtype=np.float32))
        self.assertEqual(stream.update()["cutoff_seconds"], 30)
        stream.append(np.zeros(5 * 16000, dtype=np.float32))
        self.assertIsNone(stream.update())
        stream.append(np.zeros(25 * 16000, dtype=np.float32))
        update = stream.update()
        self.assertEqual(update["cutoff_seconds"], 60)
        self.assertEqual(update["cache_stats"]["new_audio_seconds"], 30)
        self.assertEqual(processed, [30 * 16000, 60 * 16000])

    def test_append_during_processing_is_included_in_next_update(self):
        stream = DiarizationStream(_AudioOnlyDiarizer(), initial_window_seconds=1, minimum_increment_seconds=1)
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

    def test_background_worker_coalesces_backlog_and_flushes_tail(self):
        stream = DiarizationStream(_AudioOnlyDiarizer(), initial_window_seconds=1, minimum_increment_seconds=1)
        started = threading.Event()
        release = threading.Event()
        caught_up = threading.Event()
        published = []

        def process(audio):
            if len(audio) == 16000:
                started.set()
                self.assertTrue(release.wait(timeout=5))
            return {"samples": len(audio)}, {}

        def receive(update):
            published.append(update["cutoff_seconds"])
            if update["cutoff_seconds"] == 3:
                caught_up.set()

        stream._diarize_prefix = process
        stream.start(receive)
        stream.append(np.zeros(16000, dtype=np.float32))
        self.assertTrue(started.wait(timeout=5))
        stream.append(np.zeros(16000, dtype=np.float32))
        stream.append(np.zeros(16000, dtype=np.float32))
        release.set()
        self.assertTrue(caught_up.wait(timeout=5))
        stream.append(np.zeros(8000, dtype=np.float32))
        stream.close()
        self.assertEqual(published, [1, 3, 3.5])


if __name__ == "__main__":
    unittest.main()
