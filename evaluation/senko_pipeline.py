import os
import tempfile
from typing import Any, Callable, Dict

import senko
import soundfile as sf
try:
    from pydantic import Field
except ModuleNotFoundError:
    def Field(default=None, **kwargs):  # pragma: no cover - fallback for import-time compatibility
        return default

try:
    from openbench.dataset import DiarizationSample
    from openbench.pipeline.base import Pipeline, PipelineType, register_pipeline
    from openbench.pipeline.diarization.common import DiarizationOutput, DiarizationPipelineConfig
    from openbench.pipeline_prediction import DiarizationAnnotation

    OPENBENCH_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    OPENBENCH_IMPORT_ERROR = exc

    class DiarizationSample:  # pragma: no cover - fallback for import-time compatibility
        waveform: Any
        sample_rate: int

    class DiarizationPipelineConfig:  # pragma: no cover - fallback for import-time compatibility
        pass

    class Pipeline:  # pragma: no cover - fallback for import-time compatibility
        def __init__(self, config):
            self.config = config

    class PipelineType:  # pragma: no cover - fallback for import-time compatibility
        DIARIZATION = "diarization"

    def register_pipeline(cls):  # pragma: no cover - fallback for import-time compatibility
        return cls

    class DiarizationAnnotation(dict):  # pragma: no cover - fallback for import-time compatibility
        pass

    class DiarizationOutput:  # pragma: no cover - fallback for import-time compatibility
        def __init__(self, prediction):
            self.prediction = prediction


def _segment_payload(segments):
    return [
        {
            "start": segment["start"],
            "end": segment["end"],
            "speaker": segment["speaker"],
        }
        for segment in segments
    ]


def _build_diarization_annotation(annotation_cls, segments):
    payload = _segment_payload(segments)

    if isinstance(annotation_cls, type) and issubclass(annotation_cls, dict):
        annotation = annotation_cls()
        for segment in payload:
            annotation[(segment["start"], segment["end"])] = segment["speaker"]
        return annotation

    if hasattr(annotation_cls, "from_segments"):
        return annotation_cls.from_segments(payload)

    if hasattr(annotation_cls, "model_validate"):
        for candidate in (payload, {"segments": payload}, {"prediction": payload}):
            try:
                return annotation_cls.model_validate(candidate)
            except Exception:
                pass

    for constructor_payload in ({"segments": payload}, {"prediction": payload}):
        try:
            return annotation_cls(**constructor_payload)
        except TypeError:
            pass

    annotation = annotation_cls()
    if hasattr(annotation, "segments") and isinstance(annotation.segments, list):
        annotation.segments.extend(payload)
        return annotation

    if hasattr(annotation, "__setitem__"):
        for segment in payload:
            annotation[(segment["start"], segment["end"])] = segment["speaker"]
        return annotation

    raise TypeError(
        f"Unsupported diarization annotation type: {annotation_cls!r}. "
        "Could not populate it from Senko segments."
    )


class SenkoPipelineConfig(DiarizationPipelineConfig):
    model_dir: str | None = Field(
        default=None,
        description="Optional custom model root. Senko resolves models from here first, then falls back to bundled models."
    )
    device: str = Field(
        default="auto",
        description="Device to use for VAD & embeddings stage (auto, cuda, coreml, cpu)"
    )
    vad: str = Field(
        default="auto",
        description="Voice Activity Detection model to use (auto, pyannote, silero)"
    )
    clustering: str = Field(
        default="auto",
        description="Clustering location when device == cuda (auto, gpu, cpu)"
    )
    warmup: bool = Field(
        default=True,
        description="Warm up CAM++ embedding model and clustering objects during initialization"
    )
    quiet: bool = Field(
        default=True,
        description="Suppress progress updates and all other output to stdout"
    )
    num_worker_processes: int | None = Field(
        default=None,
        description="Number of worker processes to use for parallel processing"
    )


@register_pipeline
class SenkoPipeline(Pipeline):
    _config_class = SenkoPipelineConfig
    pipeline_type = PipelineType.DIARIZATION

    def __init__(self, config: SenkoPipelineConfig):
        if OPENBENCH_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "OpenBench is required to use evaluation.senko_pipeline. "
                "Install the OpenBench evaluation dependencies first."
            ) from OPENBENCH_IMPORT_ERROR

        super().__init__(config)
        self.temp_files = []

    def __del__(self):
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass

    def build_pipeline(self) -> Callable[[Dict[str, Any]], DiarizationAnnotation]:
        self.diarizer = senko.Diarizer(
            model_dir=self.config.model_dir,
            device=self.config.device,
            vad=self.config.vad,
            clustering=self.config.clustering,
            warmup=self.config.warmup,
            quiet=self.config.quiet
        )

        def call_pipeline(inputs: Dict[str, Any]) -> DiarizationAnnotation:
            wav_path = inputs["wav_path"]
            try:
                result = self.diarizer.diarize(wav_path, generate_colors=False)
                if result is None:
                    return _build_diarization_annotation(DiarizationAnnotation, [])

                return _build_diarization_annotation(DiarizationAnnotation, result["merged_segments"])

            except senko.AudioFormatError as e:
                raise ValueError(f"Audio format error: {e}")

            finally:
                if wav_path in self.temp_files:
                    try:
                        os.remove(wav_path)
                        self.temp_files.remove(wav_path)
                    except Exception:
                        pass

        return call_pipeline

    def parse_input(self, input_sample: DiarizationSample) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            self.temp_files.append(tmp_path)

            waveform = input_sample.waveform
            sample_rate = input_sample.sample_rate

            if sample_rate != 16000:
                import librosa
                waveform = librosa.resample(
                    waveform,
                    orig_sr=sample_rate,
                    target_sr=16000
                )
                sample_rate = 16000

            sf.write(tmp_path, waveform, sample_rate, subtype="PCM_16")

        return {"wav_path": tmp_path}

    def parse_output(self, output: DiarizationAnnotation) -> DiarizationOutput:
        return DiarizationOutput(prediction=output)
