import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from senko import config


def _create_model_assets(root: Path):
    for relative_path in config.MODEL_RELATIVE_PATHS.values():
        asset_path = root / relative_path
        if asset_path.suffix in {'.mlpackage', '.mlmodelc'}:
            asset_path.mkdir(parents=True, exist_ok=True)
            (asset_path / 'marker.txt').write_text('test', encoding='utf-8')
        else:
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            asset_path.write_bytes(b'test')


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.default_dir = self.root / 'default-models'
        self.custom_dir = self.root / 'custom-models'
        self.env_dir = self.root / 'env-models'
        _create_model_assets(self.default_dir)
        _create_model_assets(self.custom_dir)
        _create_model_assets(self.env_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_explicit_model_dir_overrides_env(self):
        custom_asset = self.custom_dir / config.MODEL_RELATIVE_PATHS['embeddings_pt_model_path']
        custom_asset.write_bytes(b'custom')

        env_asset = self.env_dir / config.MODEL_RELATIVE_PATHS['embeddings_pt_model_path']
        env_asset.write_bytes(b'env')

        with patch.object(config, 'DEFAULT_MODELS_DIR', self.default_dir):
            with patch.dict(os.environ, {config.SENKO_MODEL_DIR_ENV_VAR: str(self.env_dir)}, clear=False):
                model_paths = config.resolve_model_paths(self.custom_dir)

        self.assertEqual(model_paths.configured_model_dir, self.custom_dir.resolve())
        self.assertEqual(model_paths.embeddings_pt_model_path, custom_asset.resolve())

    def test_env_model_dir_overrides_default(self):
        env_asset = self.env_dir / config.MODEL_RELATIVE_PATHS['embeddings_jit_cuda_model_path']
        env_asset.write_bytes(b'env')

        with patch.object(config, 'DEFAULT_MODELS_DIR', self.default_dir):
            with patch.dict(os.environ, {config.SENKO_MODEL_DIR_ENV_VAR: str(self.env_dir)}, clear=False):
                model_paths = config.resolve_model_paths()

        self.assertEqual(model_paths.configured_model_dir, self.env_dir.resolve())
        self.assertEqual(model_paths.embeddings_jit_cuda_model_path, env_asset.resolve())

    def test_missing_custom_asset_falls_back_to_default(self):
        missing_relative_path = config.MODEL_RELATIVE_PATHS['pyannote_segmentation_senko_model_path']
        (self.custom_dir / missing_relative_path).unlink()

        with patch.object(config, 'DEFAULT_MODELS_DIR', self.default_dir):
            model_paths = config.resolve_model_paths(self.custom_dir)

        self.assertEqual(
            model_paths.pyannote_segmentation_senko_model_path,
            (self.default_dir / missing_relative_path).resolve(),
        )

    def test_missing_asset_in_custom_and_default_raises(self):
        missing_relative_path = config.MODEL_RELATIVE_PATHS['embeddings_coreml_path']
        marker = self.default_dir / missing_relative_path / 'marker.txt'
        marker.unlink()
        (self.default_dir / missing_relative_path).rmdir()
        marker = self.custom_dir / missing_relative_path / 'marker.txt'
        marker.unlink()
        (self.custom_dir / missing_relative_path).rmdir()

        with patch.object(config, 'DEFAULT_MODELS_DIR', self.default_dir):
            with self.assertRaises(FileNotFoundError):
                config.resolve_model_paths(self.custom_dir)

    def test_lazy_model_resolution_skips_unrequested_pyannote_assets(self):
        missing_relative_path = config.MODEL_RELATIVE_PATHS['pyannote_segmentation_senko_model_path']
        (self.default_dir / missing_relative_path).unlink()
        (self.custom_dir / missing_relative_path).unlink()

        with patch.object(config, 'DEFAULT_MODELS_DIR', self.default_dir):
            model_paths = config.resolve_model_paths(
                self.custom_dir,
                required_fields=config.EMBEDDINGS_MODEL_FIELDS,
            )

        self.assertIsNone(model_paths.pyannote_segmentation_senko_model_path)
        self.assertEqual(
            model_paths.embeddings_pt_model_path,
            (self.custom_dir / config.MODEL_RELATIVE_PATHS['embeddings_pt_model_path']).resolve(),
        )

    def test_coreml_cache_metadata_invalidates_on_source_change(self):
        with patch.object(config, 'DEFAULT_MODELS_DIR', self.default_dir):
            model_paths = config.resolve_model_paths(self.custom_dir)

        compiled_cache_dir = config.get_coreml_embeddings_cache_dir(model_paths)
        compiled_cache_dir.mkdir(parents=True, exist_ok=True)
        (compiled_cache_dir / 'model.bin').write_bytes(b'compiled')

        with patch('senko.config.get_senko_version', return_value='1.2.3'):
            with patch('senko.config.platform.mac_ver', return_value=('14.5', ('', '', ''), '')):
                config.update_coreml_embeddings_cache_metadata(model_paths, '8.1')
                self.assertTrue(config.is_coreml_embeddings_cache_valid(model_paths, '8.1'))

                source_model_file = self.custom_dir / config.MODEL_RELATIVE_PATHS['embeddings_coreml_path'] / 'marker.txt'
                source_model_file.write_text('changed', encoding='utf-8')
                self.assertFalse(config.is_coreml_embeddings_cache_valid(model_paths, '8.1'))


if __name__ == '__main__':
    unittest.main()
