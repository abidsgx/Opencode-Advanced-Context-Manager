"""Tests for config settings."""

import pytest
import yaml

from config.settings import load_config, save_config, get_setting, update_setting


class TestConfig:
    """Tests for config settings."""

    def test_load_config(self, sample_config_yaml):
        """Config from YAML."""
        config = load_config()

        assert "version" in config
        assert "importance" in config
        assert "pruning" in config

    def test_load_config_missing(self, tmp_path, monkeypatch):
        """Returns default when missing."""
        monkeypatch.setattr("config.settings.get_config_path", lambda: tmp_path / "nonexistent.yaml")

        config = load_config()

        # Should return default config
        assert "version" in config
        assert "importance" in config

    def test_get_setting(self):
        """Specific setting retrieved."""
        config = load_config()
        threshold = get_setting("pruning", "importance_threshold")

        assert threshold is not None
        assert isinstance(threshold, (int, float))

    def test_update_setting(self, tmp_path, monkeypatch):
        """Updated and persisted."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("version: '1.0'\npruning:\n  importance_threshold: 3.0\n")
        monkeypatch.setattr("config.settings.get_config_path", lambda: config_path)

        result = update_setting("pruning", "importance_threshold", "5.0")

        assert result["status"] == "ok"
        assert result["value"] == 5.0

        # Verify persisted
        with open(config_path) as f:
            saved = yaml.safe_load(f)
        assert saved["pruning"]["importance_threshold"] == 5.0
