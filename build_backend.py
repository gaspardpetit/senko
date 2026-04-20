from __future__ import annotations

import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from scikit_build_core import build as _backend


def _find_vsdevcmd() -> str | None:
    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if vswhere.exists():
        try:
            result = subprocess.run(
                [
                    str(vswhere),
                    "-latest",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-find",
                    r"Common7\Tools\VsDevCmd.bat",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            pass
        else:
            path = result.stdout.strip()
            if path:
                return path

    fallbacks = [
        Path(r"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\17\Community\Common7\Tools\VsDevCmd.bat"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\Tools\VsDevCmd.bat"),
    ]
    for path in fallbacks:
        if path.exists():
            return str(path)
    return None


@lru_cache(maxsize=1)
def _msvc_environment() -> dict[str, str]:
    vsdevcmd = _find_vsdevcmd()
    if not vsdevcmd:
        raise RuntimeError("Could not find VsDevCmd.bat for an installed Visual Studio C++ toolchain.")

    with tempfile.NamedTemporaryFile("w", suffix=".cmd", delete=False, encoding="ascii") as script:
        script.write("@echo off\n")
        script.write(f'call "{vsdevcmd}" -arch=x64 -host_arch=x64 >nul || exit /b 1\n')
        script.write("set\n")
        script_path = script.name

    try:
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", script_path],
            capture_output=True,
            text=True,
        )
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to activate the Visual Studio build environment.\n"
            f"command: {result.args!r}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    env: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key] = value
    return env


def _ensure_windows_msvc() -> None:
    if os.name != "nt":
        return
    if os.environ.get("VSCMD_VER"):
        return

    env = _msvc_environment()
    os.environ.update(env)
    os.environ.setdefault("CC", "cl")
    os.environ.setdefault("CXX", "cl")


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _ensure_windows_msvc()
    return _backend.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _ensure_windows_msvc()
    return _backend.build_editable(wheel_directory, config_settings, metadata_directory)


def get_requires_for_build_wheel(config_settings=None):
    return _backend.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_editable(config_settings=None):
    return _backend.get_requires_for_build_editable(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    return _backend.get_requires_for_build_sdist(config_settings)


def build_sdist(sdist_directory, config_settings=None):
    return _backend.build_sdist(sdist_directory, config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _ensure_windows_msvc()
    return _backend.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    _ensure_windows_msvc()
    return _backend.prepare_metadata_for_build_editable(metadata_directory, config_settings)
