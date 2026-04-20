from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio_source(audio_source) -> np.ndarray:
    if isinstance(audio_source, np.ndarray):
        if audio_source.ndim != 1:
            raise ValueError("In-memory audio must be a 1-D mono array.")
        return np.ascontiguousarray(audio_source, dtype=np.float32)

    samples, sample_rate = sf.read(str(Path(audio_source)), dtype="float32", always_2d=False)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16kHz audio for local pyannote VAD, got {sample_rate}Hz.")
    if samples.ndim != 1:
        raise ValueError("Local pyannote VAD expects mono audio.")
    return np.ascontiguousarray(samples, dtype=np.float32)
