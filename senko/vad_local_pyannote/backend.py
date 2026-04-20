from dataclasses import dataclass

import numpy as np

from ..vad_coreml import VADProcessorCoreML
from .audio import load_audio_source
from .postprocess import aggregate_sliding_scores, build_powerset_mapping, powerset_logits_to_speech, scores_to_segments


@dataclass(frozen=True)
class VADParameters:
    onset: float = 0.5
    offset: float = 0.5
    min_duration_on: float = 0.25
    min_duration_off: float = 0.1


class LocalSegmentationVADCuda:
    def __init__(self, checkpoint_path, torch_device, batch_size: int = 32, parameters: VADParameters | None = None):
        import torch
        from .checkpoint import build_model_from_checkpoint

        self.torch = torch
        self.parameters = parameters or VADParameters()
        self.model, payload = build_model_from_checkpoint(checkpoint_path, map_location=torch_device)
        self.model.to(torch_device)
        self.model.eval()
        self.device = torch_device
        self.batch_size = batch_size

        inference_cfg = payload["inference"]
        model_cfg = payload["model"]
        self.chunk_duration = inference_cfg["duration"]
        self.chunk_step = inference_cfg["step"]
        self.warm_up = tuple(inference_cfg.get("warm_up", (0.0, 0.0)))
        self.frame_start = model_cfg.get("frame_start", self.model.frame_start)
        self.frame_duration = model_cfg.get("frame_duration", self.model.frame_duration)
        self.frame_step = model_cfg.get("frame_step", self.model.frame_step)
        self.sample_rate = model_cfg["sample_rate"]
        self.window_size = int(self.chunk_duration * self.sample_rate)
        self.step_size = round(self.chunk_step * self.sample_rate)
        self.mapping = build_powerset_mapping(
            model_cfg["num_regular_classes"],
            model_cfg["powerset_max_classes"],
        ).to(self.device)

    def _iter_chunks(self, waveform: np.ndarray):
        torch = self.torch
        tensor = torch.from_numpy(waveform).unsqueeze(0)
        _, num_samples = tensor.shape
        regular_chunk_count = 0

        if num_samples >= self.window_size:
            chunks = tensor.unfold(1, self.window_size, self.step_size).permute(1, 0, 2)
            regular_chunk_count = chunks.shape[0]
            for index in range(0, chunks.shape[0], self.batch_size):
                yield chunks[index : index + self.batch_size]

        has_last_chunk = (num_samples < self.window_size) or ((num_samples - self.window_size) % self.step_size > 0)
        if has_last_chunk:
            if num_samples <= self.window_size:
                last_chunk = tensor
                if num_samples < self.window_size:
                    last_chunk = torch.nn.functional.pad(last_chunk, (0, self.window_size - num_samples))
            else:
                last_start = num_samples - self.window_size
                if regular_chunk_count > 0:
                    last_regular_start = (regular_chunk_count - 1) * self.step_size
                    if last_start == last_regular_start:
                        return
                last_chunk = tensor[:, last_start:last_start + self.window_size]
            yield last_chunk.unsqueeze(0)

    def process(self, audio_source) -> list[tuple[float, float]]:
        waveform = load_audio_source(audio_source)
        if waveform.size == 0:
            return []

        batches = []
        with self.torch.inference_mode():
            for batch in self._iter_chunks(waveform):
                logits = self.model(batch.to(self.device))
                batches.append(powerset_logits_to_speech(logits, self.mapping))

        if not batches:
            return []

        speech_scores = np.vstack(batches)
        aggregated = aggregate_sliding_scores(
            speech_scores,
            chunk_duration=self.chunk_duration,
            chunk_step=self.chunk_step,
            frame_start=self.frame_start,
            frame_duration=self.frame_duration,
            frame_step=self.frame_step,
            total_duration=len(waveform) / self.sample_rate,
            warm_up=self.warm_up,
        )

        return scores_to_segments(
            aggregated[:, 0],
            frame_start=self.frame_start,
            frame_duration=self.frame_duration,
            frame_step=self.frame_step,
            onset=self.parameters.onset,
            offset=self.parameters.offset,
            min_duration_on=self.parameters.min_duration_on,
            min_duration_off=self.parameters.min_duration_off,
        )


class LocalSegmentationVADCoreML:
    def __init__(self, lib_path, model_path, parameters: VADParameters | None = None):
        params = parameters or VADParameters()
        self.processor = VADProcessorCoreML(
            lib_path=lib_path,
            model_path=model_path,
            min_duration_on=params.min_duration_on,
            min_duration_off=params.min_duration_off,
        )

    def process(self, audio_source) -> list[tuple[float, float]]:
        return self.processor.process_audio(audio_source)
