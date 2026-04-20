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
    FBANK_LIB_FILENAMES = ('libfbank_extractor.dylib',)
elif WINDOWS:
    FBANK_LIB_FILENAMES = ('fbank_extractor.pyd', 'libfbank_extractor.pyd')
else:
    FBANK_LIB_FILENAMES = ('libfbank_extractor.so',)

FBANK_LIB_FILENAME = FBANK_LIB_FILENAMES[0]

PACKAGE_DIR = Path(str(importlib.resources.files('senko')))


def _find_existing_path(root: Path, candidate_names: tuple[str, ...]) -> Path | None:
    for candidate_name in candidate_names:
        candidate = root / candidate_name
        if candidate.exists():
            return candidate
    return None


def get_fbank_lib_path() -> str:
    if IS_DEV_MODE:
        build_fbank_lib = _find_existing_path(BUILD_DIR, FBANK_LIB_FILENAMES)
        if build_fbank_lib is None:
            raise FileNotFoundError(
                f"Could not find any of {FBANK_LIB_FILENAMES} in development build directory "
                f"('{BUILD_DIR}'). Please run 'uv pip install -e .' successfully."
            )
        return str(build_fbank_lib)

    return str(_package_fbank_lib)


def get_vad_coreml_lib_path() -> str:
    if IS_DEV_MODE:
        return str(BUILD_DIR / 'libvad_coreml.dylib')
    return str(PACKAGE_DIR / 'libvad_coreml.dylib')


_package_fbank_lib = _find_existing_path(PACKAGE_DIR, FBANK_LIB_FILENAMES)
IS_DEV_MODE = _package_fbank_lib is None

if IS_DEV_MODE:
    project_root = Path(__file__).parent.parent
    BUILD_DIR = project_root / 'build'
    DEFAULT_MODELS_DIR = project_root / 'models'
else:
    DEFAULT_MODELS_DIR = PACKAGE_DIR / 'models'

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
    'pyannote_segmentation_senko_model_path': Path('pyannote_segmentation_3.0') / 'senko_vad.pt',
    'pyannote_segmentation_coreml_model_path': Path('pyannote_segmentation.mlmodelc'),
    'embeddings_jit_cuda_model_path': Path('camplusplus_traced_cuda_optimized.pt'),
    'embeddings_pt_model_path': Path('speech_campplus_sv_zh_en_16k-common_advanced') / 'campplus_cn_en_common.pt',
    'embeddings_coreml_path': Path('camplusplus_batch16.mlpackage'),
}

PYANNOTE_VAD_MODEL_FIELDS = (
    'pyannote_segmentation_senko_model_path',
    'pyannote_segmentation_coreml_model_path',
)

RUNTIME_PYANNOTE_CUDA_MODEL_FIELDS = ('pyannote_segmentation_senko_model_path',)
RUNTIME_PYANNOTE_COREML_MODEL_FIELDS = ('pyannote_segmentation_coreml_model_path',)

EMBEDDINGS_MODEL_FIELDS = (
    'embeddings_jit_cuda_model_path',
    'embeddings_pt_model_path',
    'embeddings_coreml_path',
)


@dataclass(frozen=True)
class ModelPaths:
    configured_model_dir: Path | None
    default_model_dir: Path
    pyannote_segmentation_senko_model_path: Path | None
    pyannote_segmentation_coreml_model_path: Path | None
    embeddings_jit_cuda_model_path: Path | None
    embeddings_pt_model_path: Path | None
    embeddings_coreml_path: Path | None

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
            return candidate.resolve()

    fallback = DEFAULT_MODELS_DIR / relative_path
    if fallback.exists():
        return fallback.resolve()

    raise FileNotFoundError(
        f"Required Senko model asset '{relative_path.as_posix()}' was not found in "
        f"{configured_model_dir if configured_model_dir is not None else 'the configured model dir'} "
        f"or the bundled model directory {DEFAULT_MODELS_DIR}."
    )


def resolve_model_paths(model_dir=None, required_fields=None) -> ModelPaths:
    configured_model_dir = resolve_configured_model_dir(model_dir)

    if required_fields is None:
        required_fields = tuple(MODEL_RELATIVE_PATHS.keys())

    unknown_fields = set(required_fields) - set(MODEL_RELATIVE_PATHS.keys())
    if unknown_fields:
        raise ValueError(f"Unknown Senko model fields requested: {sorted(unknown_fields)}")

    resolved = {}
    for field_name, relative_path in MODEL_RELATIVE_PATHS.items():
        if field_name in required_fields:
            resolved[field_name] = _resolve_model_asset(configured_model_dir, relative_path)
        else:
            resolved[field_name] = None

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
