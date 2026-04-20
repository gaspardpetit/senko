import unittest

from evaluation.senko_pipeline import _build_diarization_annotation


SEGMENTS = [
    {"start": 0.25, "end": 1.0, "speaker": "SPEAKER_01"},
    {"start": 1.5, "end": 2.25, "speaker": "SPEAKER_02"},
]


class ModelValidateAnnotation:
    def __init__(self, payload):
        self.payload = payload

    @classmethod
    def model_validate(cls, payload):
        if isinstance(payload, list):
            return cls({"root": payload})
        if "segments" in payload:
            return cls(payload)
        raise ValueError("unsupported")


class SegmentsAttributeAnnotation:
    def __init__(self):
        self.segments = []


class MappingAnnotation(dict):
    pass


class EvaluationAnnotationTests(unittest.TestCase):
    def test_build_annotation_prefers_model_validate(self):
        annotation = _build_diarization_annotation(ModelValidateAnnotation, SEGMENTS)
        self.assertEqual(annotation.payload["root"][0]["speaker"], "SPEAKER_01")

    def test_build_annotation_populates_segments_attribute(self):
        annotation = _build_diarization_annotation(SegmentsAttributeAnnotation, SEGMENTS)
        self.assertEqual(annotation.segments, SEGMENTS)

    def test_build_annotation_falls_back_to_mapping(self):
        annotation = _build_diarization_annotation(MappingAnnotation, SEGMENTS)
        self.assertEqual(annotation[(0.25, 1.0)], "SPEAKER_01")
        self.assertEqual(annotation[(1.5, 2.25)], "SPEAKER_02")


if __name__ == "__main__":
    unittest.main()
