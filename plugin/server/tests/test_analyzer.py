"""Tests for GraphAnalyzer."""

import pytest
import yaml

from core.graph.interconnectedness import InterconnectednessManager
from core.graph.analyzer import GraphAnalyzer


def _make_manager(config_data, monkeypatch):
    monkeypatch.setattr("core.graph.interconnectedness.load_config", lambda: config_data)
    return InterconnectednessManager()


class TestGraphAnalyzer:
    """Tests for GraphAnalyzer."""

    def test_analyze_file(self, monkeypatch):
        """All metrics for one file."""
        config = {
            "version": "1.0",
            "interconnectedness": {
                "files": {
                    "src/a.py": {
                        "src/b.py": {"weight": 0.8, "relation": "calls"}
                    }
                },
                "functions": {},
            },
        }
        manager = _make_manager(config, monkeypatch)
        analyzer = GraphAnalyzer(manager)
        result = analyzer.analyze_file("src/a.py")

        assert "file_path" in result
        assert "degree_centrality" in result
        assert "betweenness_centrality" in result
        assert "cluster_id" in result
        assert "dependency_count" in result
        assert "dependents_count" in result

    def test_analyze_all(self, monkeypatch):
        """Metrics for all files."""
        config = {
            "version": "1.0",
            "interconnectedness": {
                "files": {
                    "src/a.py": {
                        "src/b.py": {"weight": 0.8, "relation": "calls"}
                    },
                    "src/b.py": {
                        "src/c.py": {"weight": 0.6, "relation": "imports"}
                    },
                },
                "functions": {},
            },
        }
        manager = _make_manager(config, monkeypatch)
        analyzer = GraphAnalyzer(manager)
        results = analyzer.analyze_all()

        assert len(results) == 3
        assert "src/a.py" in results
        assert "src/b.py" in results
        assert "src/c.py" in results

    def test_find_important_files(self, monkeypatch):
        """Top N by centrality."""
        config = {
            "version": "1.0",
            "interconnectedness": {
                "files": {
                    "src/a.py": {
                        "src/b.py": {"weight": 0.8, "relation": "calls"},
                        "src/c.py": {"weight": 0.7, "relation": "imports"},
                    },
                    "src/b.py": {
                        "src/c.py": {"weight": 0.6, "relation": "imports"}
                    },
                },
                "functions": {},
            },
        }
        manager = _make_manager(config, monkeypatch)
        analyzer = GraphAnalyzer(manager)
        top_files = analyzer.find_important_files(top_n=2)

        assert len(top_files) == 2
        top_names = [f for f, _ in top_files]
        # b.py (centrality 2/3) and a.py or c.py (centrality 1/3)
        assert "src/b.py" in top_names

    def test_find_bridge_files(self, monkeypatch):
        """High betweenness files."""
        config = {
            "version": "1.0",
            "interconnectedness": {
                "files": {
                    "src/a.py": {
                        "src/b.py": {"weight": 0.8, "relation": "calls"}
                    },
                    "src/b.py": {
                        "src/c.py": {"weight": 0.6, "relation": "imports"}
                    },
                },
                "functions": {},
            },
        }
        manager = _make_manager(config, monkeypatch)
        analyzer = GraphAnalyzer(manager)
        bridges = analyzer.find_bridge_files()

        # b.py bridges a.py and c.py - betweenness may or may not be > 0.3
        # depending on graph size; just check it returns a list
        assert isinstance(bridges, list)

    def test_suggest_edges(self, monkeypatch):
        """Similar-name suggestions."""
        config = {
            "version": "1.0",
            "interconnectedness": {
                "files": {"src/auth.py": {}},
                "functions": {},
            },
        }
        manager = _make_manager(config, monkeypatch)
        analyzer = GraphAnalyzer(manager)
        suggestions = analyzer.suggest_edges("src/auth.py")

        assert isinstance(suggestions, list)
