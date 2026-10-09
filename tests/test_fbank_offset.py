import ctypes
import unittest

import numpy as np

from senko.config import get_fbank_lib_path


class FbankOffsetTests(unittest.TestCase):
    def test_large_absolute_timestamp_uses_same_samples_after_compaction(self):
        class Features(ctypes.Structure):
            _fields_ = [
                ("data", ctypes.POINTER(ctypes.c_float)),
                ("frames_per_subsegment", ctypes.POINTER(ctypes.c_size_t)),
                ("subsegment_offsets", ctypes.POINTER(ctypes.c_size_t)),
                ("num_subsegments", ctypes.c_size_t),
                ("total_frames", ctypes.c_size_t),
                ("feature_dim", ctypes.c_size_t),
            ]

        lib = ctypes.CDLL(get_fbank_lib_path())
        lib.create_fbank_extractor.restype = ctypes.c_void_p
        lib.destroy_fbank_extractor.argtypes = [ctypes.c_void_p]
        lib.extract_fbank_features_from_memory_offset.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ]
        lib.extract_fbank_features_from_memory_offset.restype = Features
        lib.free_fbank_features.argtypes = [ctypes.POINTER(Features)]

        audio = np.random.default_rng(42).uniform(-0.2, 0.2, 20 * 16000).astype(np.float32)
        start = 86400.123456
        segments = np.asarray([start, start + 1.5], dtype=np.float32)
        initial_offset = 86400 * 16000 - 10 * 16000
        absolute_start = int(np.float32(segments[0] * np.float32(16000)))
        extractor = lib.create_fbank_extractor()

        def extract(samples, offset):
            result = lib.extract_fbank_features_from_memory_offset(
                extractor, samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                len(samples), offset, segments.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1,
            )
            try:
                return np.ctypeslib.as_array(result.data, shape=(result.total_frames * result.feature_dim,)).copy()
            finally:
                lib.free_fbank_features(ctypes.byref(result))

        try:
            full_tail = extract(audio, initial_offset)
            compacted_tail = extract(audio[absolute_start - initial_offset:], absolute_start)
        finally:
            lib.destroy_fbank_extractor(extractor)
        np.testing.assert_array_equal(full_tail, compacted_tail)


if __name__ == "__main__":
    unittest.main()
