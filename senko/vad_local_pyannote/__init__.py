__all__ = ["LocalSegmentationVADCuda", "LocalSegmentationVADCoreML", "VADParameters"]


def __getattr__(name):
    if name in __all__:
        from .backend import LocalSegmentationVADCuda, LocalSegmentationVADCoreML, VADParameters

        exports = {
            "LocalSegmentationVADCuda": LocalSegmentationVADCuda,
            "LocalSegmentationVADCoreML": LocalSegmentationVADCoreML,
            "VADParameters": VADParameters,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
