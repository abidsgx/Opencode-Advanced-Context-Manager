"""Tests for tree-sitter analysis."""

import pytest
import yaml

from core.graph.tree_sitter import (
    count_lines,
    get_enabled_languages,
    get_extension_map,
)


@pytest.fixture(autouse=True)
def _patch_tree_sitter_config(monkeypatch, sample_config):
    """Ensure tree_sitter loads our test config."""
    monkeypatch.setattr("core.graph.tree_sitter.load_config", lambda: sample_config)


class TestTreeSitter:
    """Tests for tree-sitter functions."""

    def test_count_lines(self):
        """Line count correct."""
        assert count_lines("") == 0
        assert count_lines("line1") == 1
        assert count_lines("line1\nline2") == 2
        assert count_lines("line1\nline2\nline3") == 3

    def test_get_enabled_languages(self):
        """Returns enabled languages."""
        enabled = get_enabled_languages()

        assert isinstance(enabled, dict)
        # Python should be enabled by default in config
        assert "python" in enabled

    def test_get_extension_map(self):
        """Extension map correct."""
        ext_map = get_extension_map()

        assert isinstance(ext_map, dict)
        # Python should map .py
        assert ".py" in ext_map
        assert ext_map[".py"] == "python"
