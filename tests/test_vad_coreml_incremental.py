import unittest

import numpy as np

from senko import config


@unittest.skipUnless(config.DARWIN, "CoreML runs only on macOS")
class CoreMLIncrementalVADTests(unittest.TestCase):
    def test_growing_prefix_matches_batch_across_chunk_boundaries(self):
        from senko.vad_local_pyannote import LocalSegmentationVADCoreML

        model_paths = config.resolve_model_paths(
            required_fields=config.RUNTIME_PYANNOTE_COREML_MODEL_FIELDS
        )
        backend = LocalSegmentationVADCoreML(
            lib_path=config.get_vad_coreml_lib_path(),
            model_path=str(model_paths.pyannote_segmentation_coreml_model_path),
        )
        rng = np.random.default_rng(8)
        audio = np.zeros(35 * 16000, dtype=np.float32)
        audio[8 * 16000:22 * 16000] = rng.normal(0, 0.07, 14 * 16000).astype(np.float32)

        for seconds in (9.5, 10, 10.1, 19.9, 20, 20.1, 30, 35):
            prefix = audio[:round(seconds * 16000)]
            self.assertEqual(
                backend.process_incremental(prefix),
                backend.process(prefix),
                f"CoreML VAD mismatch at {seconds}s",
            )

        backend.reset_incremental()
        self.assertEqual(backend.process_incremental(audio[:20 * 16000]), backend.process(audio[:20 * 16000]))


if __name__ == "__main__":
    unittest.main()
