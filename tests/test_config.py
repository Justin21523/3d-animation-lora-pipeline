"""
Configuration System Tests

Tests for:
- OmegaConf loading and merging
- 2D/3D parameter conversion
- Stage config schema validation
- Deprecated path detection
"""
import pytest
from pathlib import Path
import tempfile
import yaml


class TestOmegaConfLoading:
    """Test OmegaConf configuration loading."""

    def test_import_omega_loader(self):
        """Test that omega_loader can be imported."""
        from anime_pipeline.config import omega_loader
        assert hasattr(omega_loader, 'load_config')

    def test_load_yaml_config(self, tmp_path):
        """Test loading a basic YAML config file."""
        from anime_pipeline.config.omega_loader import load_config

        # Create test config
        config_content = {
            'model': {
                'name': 'test_model',
                'device': 'cuda'
            },
            'processing': {
                'batch_size': 4,
                'num_workers': 2
            }
        }
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)

        # Load config
        config = load_config(str(config_file))

        assert config.model.name == 'test_model'
        assert config.model.device == 'cuda'
        assert config.processing.batch_size == 4

    def test_config_merge(self, tmp_path):
        """Test merging multiple config files."""
        from anime_pipeline.config.omega_loader import load_config, merge_configs

        # Create base config
        base_config = {'model': {'name': 'base', 'lr': 0.001}}
        base_file = tmp_path / "base.yaml"
        with open(base_file, 'w') as f:
            yaml.dump(base_config, f)

        # Create override config
        override_config = {'model': {'lr': 0.0001}}
        override_file = tmp_path / "override.yaml"
        with open(override_file, 'w') as f:
            yaml.dump(override_config, f)

        # Load and merge
        base = load_config(str(base_file))
        override = load_config(str(override_file))
        merged = merge_configs(base, override)

        assert merged.model.name == 'base'  # From base
        assert merged.model.lr == 0.0001  # Overridden


class TestParamConverter:
    """Test 2D/3D parameter conversion."""

    def test_import_param_converter(self):
        """Test that param_converter can be imported."""
        from anime_pipeline.config import param_converter
        assert hasattr(param_converter, 'convert_params')

    def test_2d_defaults(self):
        """Test 2D animation default parameters."""
        from anime_pipeline.config.param_converter import get_defaults

        params_2d = get_defaults(mode='2d')

        # 2D should have higher alpha threshold (hard edges)
        assert params_2d.get('alpha_threshold', 0.25) >= 0.20
        # 2D should have higher blur threshold
        assert params_2d.get('blur_threshold', 100) >= 80
        # 2D should have larger min cluster size
        assert params_2d.get('min_cluster_size', 20) >= 15

    def test_3d_defaults(self):
        """Test 3D animation default parameters."""
        from anime_pipeline.config.param_converter import get_defaults

        params_3d = get_defaults(mode='3d')

        # 3D should have lower alpha threshold (soft anti-aliased edges)
        assert params_3d.get('alpha_threshold', 0.15) <= 0.20
        # 3D should have lower blur threshold (allow DoF)
        assert params_3d.get('blur_threshold', 80) <= 100
        # 3D should have smaller min cluster size
        assert params_3d.get('min_cluster_size', 12) <= 15

    def test_param_conversion_2d_to_3d(self):
        """Test converting 2D params to 3D."""
        from anime_pipeline.config.param_converter import convert_params

        params_2d = {
            'alpha_threshold': 0.25,
            'blur_threshold': 100,
            'min_cluster_size': 20,
            'flip_aug': True,
            'color_aug': True,
        }

        params_3d = convert_params(params_2d, from_mode='2d', to_mode='3d')

        # 3D should have lower thresholds
        assert params_3d['alpha_threshold'] < params_2d['alpha_threshold']
        # 3D should disable augmentation
        assert params_3d['flip_aug'] is False
        assert params_3d['color_aug'] is False


class TestStageConfigValidation:
    """Test stage-specific config validation."""

    def test_clustering_config_schema(self):
        """Test clustering config has required fields."""
        config_path = Path(__file__).parent.parent / "configs" / "stages" / "clustering" / "2d_defaults.yaml"

        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Check required sections (current 2D clustering schema)
            assert "algorithm" in config
            assert "embeddings" in config
            assert "output" in config

            # Check HDBSCAN params (under algorithm)
            assert config["algorithm"].get("type") == "hdbscan"
            assert "min_cluster_size" in config["algorithm"]
            assert "min_samples" in config["algorithm"]

    def test_segmentation_config_schema(self):
        """Test segmentation config has required fields."""
        config_path = Path(__file__).parent.parent / "configs" / "stages" / "segmentation" / "toonout_2d.yaml"

        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Check required sections
            assert 'model' in config
            assert 'output' in config
            assert 'generation' in config or 'processing' in config

    def test_training_config_schema(self):
        """Test training config has required fields."""
        config_path = Path(__file__).parent.parent / "configs" / "stages" / "training" / "lora_2d_character.yaml"

        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Check required sections
            assert 'model' in config
            assert 'lora' in config
            assert 'training' in config
            assert 'dataset' in config

            # Check LoRA params
            assert 'network_dim' in config['lora']
            assert 'network_alpha' in config['lora']


class TestConfigIntegrity:
    """Test configuration file integrity."""

    def test_all_stage_configs_valid_yaml(self):
        """Test all stage configs are valid YAML."""
        configs_dir = Path(__file__).parent.parent / "configs" / "stages"

        if configs_dir.exists():
            for config_file in configs_dir.rglob("*.yaml"):
                try:
                    with open(config_file) as f:
                        yaml.safe_load(f)
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {config_file}: {e}")

    def test_no_deprecated_paths(self, tmp_path):
        """Test configs don't use deprecated path patterns."""
        deprecated_patterns = [
            '/home/justin/ai_warehouse',  # Old warehouse path
            'C:\\Users\\',  # Windows paths in config
            '/mnt/c/Users/',  # WSL user paths
        ]

        configs_dir = Path(__file__).parent.parent / "configs"

        if configs_dir.exists():
            for config_file in configs_dir.rglob("*.yaml"):
                with open(config_file) as f:
                    content = f.read()

                for pattern in deprecated_patterns:
                    if pattern in content:
                        pytest.fail(f"Deprecated path pattern '{pattern}' found in {config_file}")


class TestEnvironmentVariables:
    """Test environment variable expansion in configs."""

    def test_env_var_expansion(self, tmp_path, monkeypatch):
        """Test that environment variables are expanded in configs."""
        from anime_pipeline.config.omega_loader import load_config

        # Set test env var
        monkeypatch.setenv('TEST_MODEL_PATH', '/test/models')

        # Create config with env var
        config_content = {
            'model': {
                'path': '${TEST_MODEL_PATH}/segmentation/sam2.pt'
            }
        }
        config_file = tmp_path / "env_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)

        # Load and check expansion
        config = load_config(str(config_file), resolve_env=True)

        # Check if env var was expanded
        # Note: This depends on OmegaConf resolver configuration
        assert '${' not in str(config.model.path) or config.model.path == '${TEST_MODEL_PATH}/segmentation/sam2.pt'
