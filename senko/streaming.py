"""Batch-equivalent diarization of a growing recording with a retained audio tail."""

from __future__ import annotations

import copy
import threading
import time

import numpy as np

from .colors import generate_speaker_colors


class _IncrementalDiarizationEngine:
    """Reuse completed pipeline work when diarizing a growing audio prefix."""

    def __init__(
        self,
        diarizer,
        *,
        accurate: bool | None = None,
        generate_colors: bool = False,
    ):
        self.diarizer = diarizer
        self.accurate = accurate
        self.generate_colors = generate_colors

        self._audio_buffer = np.empty(0, dtype=np.float32)
        self._audio_start_samples = 0
        self._total_samples = 0
        self._last_cutoff = 0
        self._audio_lock = threading.Lock()
        self._processing_lock = threading.Lock()
        self._vad_regular_scores: list[np.ndarray] = []
        self._coreml_complete_chunks = 0
        self._silero_probs: list[float] = []
        self._silero_model = None
        self._features: dict[tuple[float, float], np.ndarray] = {}
        self._embedding_batches: dict[tuple[tuple[float, float], ...], np.ndarray] = {}
        if getattr(diarizer, "device", None) == "coreml" and diarizer.vad_model_type == "pyannote":
            diarizer.vad_backend.reset_incremental()

    @property
    def available_seconds(self) -> float:
        with self._audio_lock:
            return self._total_samples / 16000

    def append(self, samples, *, sample_rate: int = 16000) -> float:
        """Append mono samples and return the latest available audio time."""
        audio = self.diarizer._normalize_audio_samples(samples, sample_rate)
        with self._audio_lock:
            new_total = self._total_samples + len(audio)
            retained = self._total_samples - self._audio_start_samples
            needed = retained + len(audio)
            if needed > len(self._audio_buffer):
                capacity = max(needed, 2 * len(self._audio_buffer), 16000)
                buffer = np.empty(capacity, dtype=np.float32)
                buffer[:retained] = self._audio_buffer[:retained]
                self._audio_buffer = buffer
            self._audio_buffer[retained:needed] = audio
            self._total_samples = new_total
            return self._total_samples / 16000

    def update(self):
        """Process all pending audio and return its full-prefix result."""
        with self._processing_lock:
            with self._audio_lock:
                cutoff = self._total_samples
                previous_cutoff = self._last_cutoff
                if cutoff == self._last_cutoff:
                    return None
                audio = self._audio_buffer[:cutoff - self._audio_start_samples]

            started = time.perf_counter()
            result, cache_stats = self._diarize_prefix(audio)
            with self._audio_lock:
                self._last_cutoff = cutoff
                self._discard_finalized_audio(cutoff)
            cache_stats["update_seconds"] = time.perf_counter() - started
            cache_stats["new_audio_seconds"] = (cutoff - previous_cutoff) / 16000
        return {"cutoff_seconds": cutoff / 16000, "result": result, "cache_stats": cache_stats}

    def _discard_finalized_audio(self, cutoff: int):
        """Release finalized samples while retaining VAD and feature boundaries."""
        d = self.diarizer
        if getattr(d, "vad_model_type", None) == "pyannote" and getattr(d, "device", None) == "cuda":
            window = d.vad_backend.window_size
            next_regular = len(self._vad_regular_scores) * d.vad_backend.step_size
            discard_before = min(next_regular, cutoff - 2 * window)
        elif getattr(d, "vad_model_type", None) == "pyannote" and getattr(d, "device", None) == "coreml":
            window = 160000
            discard_before = min(self._coreml_complete_chunks * window, cutoff - 2 * window)
            discard_before = discard_before // window * window
        elif getattr(d, "vad_model_type", None) == "silero" and self._silero_probs:
            window = 512
            discard_before = min(len(self._silero_probs) * window, cutoff - 2 * 16000)
            discard_before = discard_before // window * window
        else:
            return
        discard_before = max(self._audio_start_samples, discard_before)
        if discard_before == self._audio_start_samples:
            return
        retained = self._total_samples - discard_before
        buffer = np.empty(max(retained, 16000), dtype=np.float32)
        buffer[:retained] = self._audio_buffer[discard_before - self._audio_start_samples:self._total_samples - self._audio_start_samples]
        self._audio_buffer = buffer
        self._audio_start_samples = discard_before

    def _diarize_prefix(self, audio: np.ndarray):
        d = self.diarizer
        d._timing_stats = {}
        started = time.perf_counter()
        vad_cache = {"new_vad_windows": None, "reused_vad_windows": None}
        if d.vad_model_type == "pyannote" and d.device == "cuda":
            from .diarizer import set_fp32_precision

            set_fp32_precision("ieee")
            vad_started = time.perf_counter()
            previous_windows = len(self._vad_regular_scores)
            vad_segments = d.vad_backend.process_incremental(
                audio, self._vad_regular_scores, sample_offset=self._audio_start_samples,
            )
            vad_cache = {
                "new_vad_windows": len(self._vad_regular_scores) - previous_windows,
                "reused_vad_windows": previous_windows,
            }
            d._timing_stats["vad_time"] = round(time.perf_counter() - vad_started, 2)
        elif d.vad_model_type == "pyannote" and d.device == "coreml":
            vad_started = time.perf_counter()
            vad_segments = d.vad_backend.process_incremental(audio, sample_offset=self._audio_start_samples)
            complete_chunks = (self._audio_start_samples + len(audio)) // 160000
            vad_cache = {
                "new_vad_windows": complete_chunks - self._coreml_complete_chunks,
                "reused_vad_windows": self._coreml_complete_chunks,
            }
            self._coreml_complete_chunks = complete_chunks
            d._timing_stats["vad_time"] = round(time.perf_counter() - vad_started, 2)
        elif d.vad_model_type == "silero":
            try:
                from silero_vad import get_speech_timestamps_from_probs
            except ImportError:
                vad_started = time.perf_counter()
                previous_windows = len(self._silero_probs)
                vad_segments = self._silero_vad_legacy(audio)
                vad_cache = {
                    "new_vad_windows": len(self._silero_probs) - previous_windows,
                    "reused_vad_windows": previous_windows,
                }
                d._timing_stats["vad_time"] = round(time.perf_counter() - vad_started, 2)
            else:
                vad_started = time.perf_counter()
                previous_windows = len(self._silero_probs)
                vad_segments = self._silero_vad(audio, get_speech_timestamps_from_probs)
                vad_cache = {
                    "new_vad_windows": len(self._silero_probs) - previous_windows,
                    "reused_vad_windows": previous_windows,
                }
                d._timing_stats["vad_time"] = round(time.perf_counter() - vad_started, 2)
        else:
            vad_segments = d._perform_vad(audio)

        if not vad_segments:
            self._features.clear()
            self._embedding_batches.clear()
            return None, {**vad_cache, "new_features": 0, "reused_features": 0, "new_embedding_batches": 0, "reused_embedding_batches": 0}

        subsegments = d._generate_subsegments(vad_segments, self.accurate)
        # VAD can extend a subsegment beyond the available prefix. The fbank
        # extractor truncates it at EOF, so that feature changes when more
        # audio arrives even if the subsegment's timestamps stay identical.
        old_batches = self._embedding_batches
        batch_keys = [tuple(subsegments[start:start + 64]) for start in range(0, len(subsegments), 64)]
        needed_segments = [segment for keys in batch_keys if keys not in old_batches for segment in keys]
        features_by_segment = {segment: self._features[segment] for segment in needed_segments if segment in self._features}
        new_segments = [segment for segment in needed_segments if segment not in features_by_segment]
        if new_segments:
            features, frames, offsets, dim = d._extract_fbank_features(
                audio, new_segments, sample_offset=self._audio_start_samples,
            )
            for segment, count, offset in zip(new_segments, frames, offsets):
                start = int(offset)
                end = start + int(count) * dim
                features_by_segment[segment] = features[start:end].copy().reshape(int(count), dim)
        else:
            d._timing_stats["fbank_time"] = 0.0

        cutoff_seconds = (self._audio_start_samples + len(audio)) / 16000
        # Only the final batch can change when speech near the growing edge
        # extends. Older batches already have reusable embeddings.
        final_batch = subsegments[-64:]
        self._features = {
            segment: features_by_segment.get(segment, self._features.get(segment))
            for segment in final_batch
            if segment[1] <= cutoff_seconds and (segment in features_by_segment or segment in self._features)
        }
        new_batches = {}
        batch_embeddings = []
        new_batch_count = 0
        for keys in batch_keys:
            if keys in old_batches:
                batch = old_batches[keys]
            else:
                items = [features_by_segment[key] for key in keys]
                frames = np.asarray([item.shape[0] for item in items], dtype=np.int32)
                offsets = np.cumsum(np.r_[0, frames[:-1]], dtype=np.int64) * items[0].shape[1]
                flat = np.concatenate([item.ravel() for item in items])
                batch = d._generate_embeddings(flat, frames, offsets, items[0].shape[1])
                new_batch_count += 1
            if all(key[1] <= cutoff_seconds for key in keys):
                new_batches[keys] = batch
            batch_embeddings.append(batch)
        self._embedding_batches = new_batches
        if not new_batch_count:
            d._timing_stats["embeddings_time"] = 0.0

        embeddings = np.vstack(batch_embeddings)
        raw_segments, merged_segments, centroids = d._perform_clustering(embeddings, subsegments)
        d._timing_stats["total_time"] = round(time.perf_counter() - started, 2)
        result = {
            "raw_segments": raw_segments,
            "raw_speakers_detected": len({item["speaker"] for item in raw_segments}),
            "merged_speakers_detected": len({item["speaker"] for item in merged_segments}),
            "merged_segments": merged_segments,
            "speaker_centroids": centroids,
            "timing_stats": d._timing_stats.copy(),
            "vad": vad_segments,
        }
        if self.generate_colors:
            result["speaker_color_sets"] = {
                str(index): generate_speaker_colors(merged_segments, index)
                for index in range(10)
            }
        return result, {
            **vad_cache,
            "new_features": len(new_segments),
            "reused_features": len(needed_segments) - len(new_segments),
            "new_embedding_batches": new_batch_count,
            "reused_embedding_batches": len(subsegments[::64]) - new_batch_count,
        }

    def _silero_vad(self, audio, timestamps_from_probs):
        import torch

        d = self.diarizer
        if self._silero_model is None:
            self._silero_model = copy.deepcopy(d.vad_model_silero)
            self._silero_model.reset_states()

        window_size = 512
        total_samples = self._audio_start_samples + len(audio)
        full_count, remainder = divmod(total_samples, window_size)
        wav = torch.from_numpy(audio)
        d._set_torch_num_threads(1)
        try:
            with torch.no_grad():
                for index in range(len(self._silero_probs), full_count):
                    start = index * window_size - self._audio_start_samples
                    chunk = wav[start:start + window_size]
                    self._silero_probs.append(self._silero_model(chunk, 16000).item())
                probs = list(self._silero_probs)
                if remainder:
                    tail_model = copy.deepcopy(self._silero_model)
                    tail_start = full_count * window_size - self._audio_start_samples
                    tail = torch.nn.functional.pad(wav[tail_start:], (0, window_size - remainder))
                    probs.append(tail_model(tail, 16000).item())
            timestamps = timestamps_from_probs(
                probs,
                sampling_rate=16000,
                threshold=0.55,
                min_speech_duration_ms=250,
                min_silence_duration_ms=100,
                return_seconds=False,
                audio_length_samples=total_samples,
                step=1,
            )
            return [(float(item["start"]) / 16000, float(item["end"]) / 16000) for item in timestamps]
        finally:
            d._set_torch_num_threads()

    def _silero_vad_legacy(self, audio):
        """Reuse probabilities with Silero versions lacking from-probs postprocessing."""
        import torch

        d = self.diarizer
        if self._silero_model is None:
            self._silero_model = copy.deepcopy(d.vad_model_silero)
            self._silero_model.reset_states()
        total_samples = self._audio_start_samples + len(audio)
        complete_count = total_samples // 512
        cached_probs = self._silero_probs
        persistent_model = self._silero_model
        tail = torch.from_numpy(audio)

        class CachedModel:
            def __init__(self):
                self.index = 0

            def reset_states(self):
                self.index = 0

            def __call__(self, chunk, sampling_rate):
                index = self.index
                self.index += 1
                if index < len(cached_probs):
                    return torch.tensor(cached_probs[index])
                start = index * 512 - self_offset
                chunk = tail[start:start + 512]
                if len(chunk) < 512:
                    chunk = torch.nn.functional.pad(chunk, (0, 512 - len(chunk)))
                if index < complete_count:
                    probability = persistent_model(chunk, sampling_rate).item()
                    cached_probs.append(probability)
                else:
                    probability = copy.deepcopy(persistent_model)(chunk, sampling_rate).item()
                return torch.tensor(probability)

        self_offset = self._audio_start_samples
        d._set_torch_num_threads(1)
        try:
            timestamps = d.get_speech_timestamps_silero(
                torch.zeros(1).expand(total_samples), CachedModel(), threshold=0.55,
                min_speech_duration_ms=250, min_silence_duration_ms=100, return_seconds=False,
            )
            return [(float(item["start"]) / 16000, float(item["end"]) / 16000) for item in timestamps]
        finally:
            d._set_torch_num_threads()


class DiarizationSession:
    """Synchronous diarization of a growing recording.

    Each call to ``add_samples`` processes the complete available prefix while
    reusing completed work. Returned results are independent snapshots.
    """

    def __init__(self, diarizer, *, accurate: bool | None = None, generate_colors: bool = False):
        self._stream = _IncrementalDiarizationEngine(diarizer, accurate=accurate, generate_colors=generate_colors)
        self._session_lock = threading.Lock()
        self._latest_result = None

    def add_samples(self, samples, *, sample_rate: int = 16000):
        """Append audio, process it, and return a copy of the diarization result."""
        with self._session_lock:
            self._stream.append(samples, sample_rate=sample_rate)
            update = self._stream.update()
            if update is not None:
                self._latest_result = update["result"]
            return copy.deepcopy(self._latest_result)

    def get_diarization(self):
        """Return a copy of the most recently computed result, or ``None``."""
        with self._session_lock:
            return copy.deepcopy(self._latest_result)
