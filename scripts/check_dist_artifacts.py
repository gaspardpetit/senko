from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path


MAX_ARTIFACT_SIZE_MB = 95.0


def _assert_size(path: Path) -> None:
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"ARTIFACT {path.name} {size_mb:.2f} MiB")
    if size_mb > MAX_ARTIFACT_SIZE_MB:
        raise SystemExit(
            f"{path.name} is {size_mb:.2f} MiB, which exceeds the release threshold of {MAX_ARTIFACT_SIZE_MB:.2f} MiB."
        )


def _check_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())

    required_common = {
        "senko/models/README.md",
        "senko/models/pyannote_segmentation_3.0/senko_vad.pt",
        "senko/models/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt",
    }
    missing = sorted(name for name in required_common if name not in names)
    if missing:
        raise SystemExit(f"{path.name} is missing required packaged assets: {missing}")

    if "win_" in path.name:
        required_native = "senko/fbank_extractor.pyd"
        optional_extra = None
    elif "macosx" in path.name:
        required_native = "senko/libfbank_extractor.dylib"
        optional_extra = "senko/libvad_coreml.dylib"
    else:
        required_native = "senko/libfbank_extractor.so"
        optional_extra = None

    if required_native not in names:
        raise SystemExit(f"{path.name} is missing native library {required_native}")

    if optional_extra is not None and optional_extra not in names:
        raise SystemExit(f"{path.name} is missing platform-specific asset {optional_extra}")

    dist_info_entries = [name for name in names if ".dist-info/licenses/" in name]
    for license_name in ("LICENSE", "THIRD_PARTY_LICENSES"):
        if not any(entry.endswith(f"/{license_name}") for entry in dist_info_entries):
            raise SystemExit(f"{path.name} is missing bundled license file {license_name}")


def _check_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as tf:
        names = set(tf.getnames())

    top_level = path.name.removesuffix(".tar.gz")
    required = {
        f"{top_level}/pyproject.toml",
        f"{top_level}/README.md",
        f"{top_level}/models/pyannote_segmentation_3.0/senko_vad.pt",
        f"{top_level}/LICENSE",
        f"{top_level}/THIRD_PARTY_LICENSES",
    }
    missing = sorted(name for name in required if name not in names)
    if missing:
        raise SystemExit(f"{path.name} is missing required source files: {missing}")


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        raise SystemExit("Usage: python scripts/check_dist_artifacts.py <dist-or-wheelhouse-dir> [more paths...]")

    candidates: list[Path] = []
    for raw_path in argv[1:]:
        path = Path(raw_path)
        if path.is_dir():
            candidates.extend(sorted(child for child in path.iterdir() if child.suffix == ".whl" or child.name.endswith(".tar.gz")))
        else:
            candidates.append(path)

    if not candidates:
        raise SystemExit("No distribution artifacts found to inspect.")

    for path in candidates:
        _assert_size(path)
        if path.suffix == ".whl":
            _check_wheel(path)
        elif path.name.endswith(".tar.gz"):
            _check_sdist(path)
        else:
            raise SystemExit(f"Unsupported artifact type: {path}")

    print(f"CHECKED {len(candidates)} artifact(s)")


if __name__ == "__main__":
    main(sys.argv)
