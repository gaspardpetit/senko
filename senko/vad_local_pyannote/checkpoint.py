from pathlib import Path

from .model import LocalPyanNet

CHECKPOINT_FORMAT_VERSION = 1


def load_normalized_checkpoint(path: str | Path, map_location=None) -> dict:
    import torch

    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected normalized checkpoint payload type: {type(payload).__name__}")

    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported Senko VAD checkpoint format: {payload.get('format_version')!r}. "
            f"Expected {CHECKPOINT_FORMAT_VERSION}."
        )

    for required_key in ("model", "inference", "state_dict"):
        if required_key not in payload:
            raise ValueError(f"Normalized Senko VAD checkpoint is missing '{required_key}'.")

    return payload


def build_model_from_checkpoint(path: str | Path, map_location=None) -> tuple[LocalPyanNet, dict]:
    payload = load_normalized_checkpoint(path, map_location=map_location)
    model_cfg = payload["model"]
    model = LocalPyanNet(
        output_dim=model_cfg["output_dim"],
        sincnet=model_cfg.get("sincnet"),
        lstm=model_cfg.get("lstm"),
        linear=model_cfg.get("linear"),
        sample_rate=model_cfg["sample_rate"],
        num_channels=model_cfg["num_channels"],
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload
