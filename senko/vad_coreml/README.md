# Pyannote `segmentation-3.0` CoreML
This directory contains code for interfacing with Senko's bundled `segmentation-3.0` CoreML VAD model.

The model provenance is Pyannote's `segmentation-3.0`, but Senko does not require any Pyannote Python package at runtime.

The converted model as well as all the interfacing code is taken from the excellent [FluidAudio](https://github.com/FluidInference/FluidAudio) project by Fluid Inference:
- CoreML model: [https://huggingface.co/FluidInference/speaker-diarization-coreml](https://huggingface.co/FluidInference/speaker-diarization-coreml)
- Interfacing code: [https://github.com/FluidInference/FluidAudio](https://github.com/FluidInference/FluidAudio)

I have simply made the Swift code accessible from Python through a compiled/shared lib, and lightly rearranged it to isolate out only the VAD function.
