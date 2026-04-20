import os
import senko
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict
import numpy as np
import soundfile as sf
from pyannote.core import Annotation, Segment
from pydantic import Field
from openbench.dataset import DiarizationSample
from openbench.pipeline.base import Pipeline, PipelineType, register_pipeline
from openbench.pipeline.diarization.common import DiarizationOutput, DiarizationPipelineConfig
from openbench.pipeline_prediction import DiarizationAnnotation

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
    # Override the base class field to fix the type
    num_worker_processes: int | None = Field(
        default=None,
        description="Number of worker processes to use for parallel processing"
    )

@register_pipeline
class SenkoPipeline(Pipeline):
    _config_class = SenkoPipelineConfig
    pipeline_type = PipelineType.DIARIZATION

    def __init__(self, config: SenkoPipelineConfig):
        super().__init__(config)
        self.temp_files = []  # Track temp files for cleanup

    def __del__(self):
        # Clean up temporary files
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
                # Run diarization
                result = self.diarizer.diarize(wav_path, generate_colors=False)

                # Handle case where no speakers are detected
                if result is None:
                    # Return empty annotation
                    return DiarizationAnnotation()

                # Convert Senko output to Pyannote Annotation
                annotation = DiarizationAnnotation()
                segments = result["merged_segments"]
                for segment in segments:
                    start = segment["start"]
                    end = segment["end"]
                    speaker = segment["speaker"]
                    seg = Segment(start, end)
                    annotation[seg] = speaker

                return annotation

            except senko.AudioFormatError as e:
                # This shouldn't happen as we ensure correct format in parse_input but handle it just in case
                raise ValueError(f"Audio format error: {e}")

            finally:
                # Clean up temp file immediately after use
                if wav_path in self.temp_files:
                    try:
                        os.remove(wav_path)
                        self.temp_files.remove(wav_path)
                    except Exception:
                        pass

        return call_pipeline

    def parse_input(self, input_sample: DiarizationSample) -> Dict[str, Any]:
        # Parse DiarizationSample to Senko's expected input format.
        # Senko requires file paths as input (not numpy arrays), so we create a temporary WAV file from the audio sample.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            self.temp_files.append(tmp_path)  # Track for cleanup

            # Ensure audio is 16kHz mono 16-bit WAV as Senko requires
            waveform = input_sample.waveform
            sample_rate = input_sample.sample_rate

            # Resample if needed (Senko requires exactly 16kHz)
            if sample_rate != 16000:
                import librosa
                waveform = librosa.resample(
                    waveform,
                    orig_sr=sample_rate,
                    target_sr=16000
                )
                sample_rate = 16000

            # Save as 16-bit WAV (Senko's required format)
            sf.write(tmp_path, waveform, sample_rate, subtype='PCM_16')

        return {"wav_path": tmp_path}

    def parse_output(self, output: DiarizationAnnotation) -> DiarizationOutput:
        """Parse Senko output to DiarizationOutput format."""
        # Note: prediction_time will be automatically set by the base Pipeline class based on the time taken by the pipeline call
        return DiarizationOutput(prediction=output)
