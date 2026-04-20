import json
import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
import importlib.resources
from importlib.metadata import PackageNotFoundError, version

DARWIN = platform.system() == 'Darwin'
WINDOWS = platform.system() == 'Windows'
if DARWIN:
    FBANK_LIB_FILENAME = 'libfbank_extractor.dylib'
elif WINDOWS:
    FBANK_LIB_FILENAME = 'fbank_extractor.pyd'
else:
    FBANK_LIB_FILENAME = 'libfbank_extractor.so'

PACKAGE_DIR = Path(str(importlib.resources.files('senko')))
IS_DEV_MODE = not (PACKAGE_DIR / FBANK_LIB_FILENAME).exists()

if IS_DEV_MODE:
    project_root = Path(__file__).parent.parent
    BUILD_DIR = project_root / 'build'
    if not (BUILD_DIR / FBANK_LIB_FILENAME).exists():
        raise FileNotFoundError(
            f"Could not find '{FBANK_LIB_FILENAME}' in development build directory "
            f"('{BUILD_DIR}'). Please run 'uv pip install -e .' successfully."
        )

    DEFAULT_MODELS_DIR = project_root / 'models'
    FBANK_LIB_PATH = str(BUILD_DIR / FBANK_LIB_FILENAME)
    VAD_COREML_LIB_PATH = str(BUILD_DIR / 'libvad_coreml.dylib')
else:
    DEFAULT_MODELS_DIR = PACKAGE_DIR / 'models'
    FBANK_LIB_PATH = str(PACKAGE_DIR / FBANK_LIB_FILENAME)
    VAD_COREML_LIB_PATH = str(PACKAGE_DIR / 'libvad_coreml.dylib')

CLUSTER = PACKAGE_DIR / 'cluster'
SPECTRAL_YAML = str(CLUSTER / 'conf' / 'spectral.yaml')
UMAP_HDBSCAN_YAML = str(CLUSTER / 'conf' / 'umap_hdbscan.yaml')

if not DEFAULT_MODELS_DIR.exists():
    raise FileNotFoundError(
        f"The 'models' directory was not found at the expected location: {DEFAULT_MODELS_DIR}"
    )

SENKO_MODEL_DIR_ENV_VAR = 'SENKO_MODEL_DIR'
COREML_CACHE_DIRNAME = 'cached'
COREML_EMBEDDINGS_CACHE_RELATIVE_PATH = Path('coreml') / 'camplusplus_batch16.mlmodelc'
COREML_EMBEDDINGS_CACHE_METADATA_NAME = 'metadata.json'

MODEL_RELATIVE_PATHS = {
    'pyannote_segmentation_pt_model_path': Path('pyannote_segmentation_3.0') / 'pytorch_model.bin',
    'pyannote_segmentation_coreml_model_path': Path('pyannote_segmentation.mlmodelc'),
    'embeddings_jit_cuda_model_path': Path('camplusplus_traced_cuda_optimized.pt'),
    'embeddings_pt_model_path': Path('speech_campplus_sv_zh_en_16k-common_advanced') / 'campplus_cn_en_common.pt',
    'embeddings_coreml_path': Path('camplusplus_batch16.mlpackage'),
}


@dataclass(frozen=True)
class ModelPaths:
    configured_model_dir: Path | None
    default_model_dir: Path
    pyannote_segmentation_pt_model_path: Path
    pyannote_segmentation_coreml_model_path: Path
    embeddings_jit_cuda_model_path: Path
    embeddings_pt_model_path: Path
    embeddings_coreml_path: Path

    @property
    def cache_base_dir(self) -> Path:
        return (self.configured_model_dir or self.default_model_dir) / COREML_CACHE_DIRNAME

    @property
    def has_custom_model_dir(self) -> bool:
        return self.configured_model_dir is not None


def get_default_model_dir() -> Path:
    return DEFAULT_MODELS_DIR


def get_senko_version() -> str:
    try:
        return version('senko')
    except PackageNotFoundError:
        return '0.1.0'


def resolve_configured_model_dir(model_dir=None) -> Path | None:
    configured = model_dir if model_dir is not None else os.getenv(SENKO_MODEL_DIR_ENV_VAR)
    if configured in (None, ''):
        return None
    return Path(configured).expanduser().resolve()


def _resolve_model_asset(configured_model_dir: Path | None, relative_path: Path) -> Path:
    if configured_model_dir is not None:
        candidate = configured_model_dir / relative_path
        if candidate.exists():
            return candidate

    fallback = DEFAULT_MODELS_DIR / relative_path
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Required Senko model asset '{relative_path.as_posix()}' was not found in "
        f"{configured_model_dir if configured_model_dir is not None else 'the configured model dir'} "
        f"or the bundled model directory {DEFAULT_MODELS_DIR}."
    )


def resolve_model_paths(model_dir=None) -> ModelPaths:
    configured_model_dir = resolve_configured_model_dir(model_dir)

    resolved = {
        field_name: _resolve_model_asset(configured_model_dir, relative_path)
        for field_name, relative_path in MODEL_RELATIVE_PATHS.items()
    }

    return ModelPaths(
        configured_model_dir=configured_model_dir,
        default_model_dir=DEFAULT_MODELS_DIR,
        **resolved,
    )


def get_coreml_embeddings_cache_dir(model_paths: ModelPaths) -> Path:
    return model_paths.cache_base_dir / COREML_EMBEDDINGS_CACHE_RELATIVE_PATH


def get_coreml_embeddings_cache_metadata_path(model_paths: ModelPaths) -> Path:
    return get_coreml_embeddings_cache_dir(model_paths).parent / COREML_EMBEDDINGS_CACHE_METADATA_NAME


def ensure_writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / '.senko-write-test'
        with open(probe, 'w', encoding='utf-8'):
            pass
        probe.unlink()
        return True
    except OSError:
        return False


def remove_path(path: Path) -> None:
    if not path.exists():
        return

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def fingerprint_path(path: Path) -> dict:
    if path.is_dir():
        entries = []
        for child in sorted(path.rglob('*')):
            if child.is_dir():
                continue
            stat = child.stat()
            entries.append({
                'path': child.relative_to(path).as_posix(),
                'size': stat.st_size,
                'mtime_ns': stat.st_mtime_ns,
            })
        return {
            'type': 'directory',
            'entries': entries,
        }

    stat = path.stat()
    return {
        'type': 'file',
        'size': stat.st_size,
        'mtime_ns': stat.st_mtime_ns,
    }


def make_coreml_cache_metadata(source_model_path: Path, coremltools_version: str) -> dict:
    return {
        'senko_version': get_senko_version(),
        'macos_version': platform.mac_ver()[0],
        'coremltools_version': coremltools_version,
        'source_model_path': str(source_model_path.resolve()),
        'source_model_fingerprint': fingerprint_path(source_model_path),
    }


def read_json_file(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json_file(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)


def is_coreml_embeddings_cache_valid(model_paths: ModelPaths, coremltools_version: str) -> bool:
    compiled_model_dir = get_coreml_embeddings_cache_dir(model_paths)
    metadata_path = get_coreml_embeddings_cache_metadata_path(model_paths)
    if not compiled_model_dir.exists() or not metadata_path.exists():
        return False

    try:
        metadata = read_json_file(metadata_path)
    except (OSError, json.JSONDecodeError):
        return False

    return metadata == make_coreml_cache_metadata(model_paths.embeddings_coreml_path, coremltools_version)


def update_coreml_embeddings_cache_metadata(model_paths: ModelPaths, coremltools_version: str) -> None:
    write_json_file(
        get_coreml_embeddings_cache_metadata_path(model_paths),
        make_coreml_cache_metadata(model_paths.embeddings_coreml_path, coremltools_version),
    )


def reset_coreml_embeddings_cache(model_paths: ModelPaths) -> None:
    remove_path(get_coreml_embeddings_cache_dir(model_paths))
    remove_path(get_coreml_embeddings_cache_metadata_path(model_paths))


# Numba compilation caching directory (~/.cache/senko/numba_cache)
cache_dir = Path.home() / '.cache' / 'senko' / 'numba_cache'
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ['NUMBA_CACHE_DIR'] = str(cache_dir)

# Patch numba.njit to enable disk caching for all decorated functions
import numba
_original_njit = numba.njit


def cached_njit(*args, **kwargs):
    kwargs['cache'] = True
    return _original_njit(*args, **kwargs)


numba.njit = cached_njit
