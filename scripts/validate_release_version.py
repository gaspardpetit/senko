from __future__ import annotations

import email
import re
import sys
import tarfile
import zipfile
from pathlib import Path


TAG_PATTERN = re.compile(r"^v(?P<version>\d+\.\d+\.\d+)$")
WHEEL_NAME_PATTERN = re.compile(r"^senko-(?P<version>[^-]+)-")
SDIST_NAME_PATTERN = re.compile(r"^senko-(?P<version>.+)\.tar\.gz$")


def _parse_tag_version(raw_tag: str) -> str:
    match = TAG_PATTERN.fullmatch(raw_tag)
    if match is None:
        raise SystemExit(
            f"Release tags must match strict semver format vX.Y.Z; got {raw_tag!r}."
        )
    return match.group("version")


def _read_wheel_version(path: Path) -> str:
    filename_match = WHEEL_NAME_PATTERN.match(path.name)
    if filename_match is None:
        raise SystemExit(f"Unexpected wheel filename format: {path.name}")

    with zipfile.ZipFile(path) as zf:
        metadata_name = next((name for name in zf.namelist() if name.endswith(".dist-info/METADATA")), None)
        if metadata_name is None:
            raise SystemExit(f"{path.name} is missing dist-info/METADATA.")
        metadata = email.message_from_bytes(zf.read(metadata_name))

    metadata_version = metadata.get("Version")
    if metadata_version is None:
        raise SystemExit(f"{path.name} METADATA is missing Version.")

    filename_version = filename_match.group("version")
    if filename_version != metadata_version:
        raise SystemExit(
            f"{path.name} has mismatched versions: filename={filename_version}, metadata={metadata_version}."
        )
    return metadata_version


def _read_sdist_version(path: Path) -> str:
    filename_match = SDIST_NAME_PATTERN.match(path.name)
    if filename_match is None:
        raise SystemExit(f"Unexpected sdist filename format: {path.name}")

    with tarfile.open(path, "r:gz") as tf:
        pkg_info_member = next((member for member in tf.getmembers() if member.name.endswith("/PKG-INFO")), None)
        if pkg_info_member is None:
            raise SystemExit(f"{path.name} is missing PKG-INFO.")
        pkg_info_bytes = tf.extractfile(pkg_info_member).read()
        metadata = email.message_from_bytes(pkg_info_bytes)

    metadata_version = metadata.get("Version")
    if metadata_version is None:
        raise SystemExit(f"{path.name} PKG-INFO is missing Version.")

    filename_version = filename_match.group("version")
    if filename_version != metadata_version:
        raise SystemExit(
            f"{path.name} has mismatched versions: filename={filename_version}, metadata={metadata_version}."
        )
    return metadata_version


def _artifact_version(path: Path) -> str:
    if path.suffix == ".whl":
        return _read_wheel_version(path)
    if path.name.endswith(".tar.gz"):
        return _read_sdist_version(path)
    raise SystemExit(f"Unsupported artifact type: {path}")


def main(argv: list[str]) -> None:
    if len(argv) < 3:
        raise SystemExit(
            "Usage: python scripts/validate_release_version.py <tag> <dist-or-artifact> [more paths...]"
        )

    expected_version = _parse_tag_version(argv[1])
    artifacts: list[Path] = []
    for raw_path in argv[2:]:
        path = Path(raw_path)
        if path.is_dir():
            artifacts.extend(
                sorted(
                    child
                    for child in path.iterdir()
                    if child.suffix == ".whl" or child.name.endswith(".tar.gz")
                )
            )
        else:
            artifacts.append(path)

    if not artifacts:
        raise SystemExit("No distribution artifacts found to validate.")

    for artifact in artifacts:
        artifact_version = _artifact_version(artifact)
        if artifact_version != expected_version:
            raise SystemExit(
                f"{artifact.name} has version {artifact_version}, expected release version {expected_version}."
            )
        print(f"RELEASE_VERSION_OK {artifact.name} {artifact_version}")


if __name__ == "__main__":
    main(sys.argv)
