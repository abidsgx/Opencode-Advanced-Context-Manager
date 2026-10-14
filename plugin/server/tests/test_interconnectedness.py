"""Tests for InterconnectednessManager."""

import pytest
import yaml

from core.graph.interconnectedness import InterconnectednessManager


class TestInterconnectednessManager:
    """Tests for InterconnectednessManager."""

    def test_add_edge(self, tmp_path, monkeypatch):
        """Edge stored in matrix."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("version: '1.0'\ninterconnectedness:\n  files: {}\n  functions: {}\n")
        monkeypatch.setattr("core.graph.interconnectedness.load_config", lambda: yaml.safe_load(config_path.read_text()))

        manager = InterconnectednessManager()

        manager.add_edge("src/a.py", "src/b.py", 0.8, "calls")

        connections = manager.get_connections("src/a.py")
        assert "src/b.py" in connections
        assert connections["src/b.py"]["weight"] == 0.8
        assert connections["src/b.py"]["relation"] == "calls"

    def test_remove_edge(self, tmp_path, monkeypatch):
        """Edge removed."""
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
        monkeypatch.setattr("core.graph.interconnectedness.load_config", lambda: config)

        manager = InterconnectednessManager()

        manager.remove_edge("src/a.py", "src/b.py")

        connections = manager.get_connections("src/a.py")
        assert "src/b.py" not in connections

    def test_get_connections(self, tmp_path, monkeypatch):
        """Returns edges for file."""
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
        monkeypatch.setattr("core.graph.interconnectedness.load_config", lambda: config)

        manager = InterconnectednessManager()

        connections = manager.get_connections("src/a.py")

        assert "src/b.py" in connections
        assert connections["src/b.py"]["weight"] == 0.8

    def test_get_blast_radius(self, tmp_path, monkeypatch):
        """Files above threshold returned."""
        config = {
            "version": "1.0",
            "interconnectedness": {
                "files": {
                    "src/a.py": {
                        "src/b.py": {"weight": 0.9, "relation": "calls"},
                        "src/c.py": {"weight": 0.2, "relation": "imports"},
                    }
                },
                "functions": {},
            },
        }
        monkeypatch.setattr("core.graph.interconnectedness.load_config", lambda: config)

        manager = InterconnectednessManager()

        affected = manager.get_blast_radius("src/a.py", threshold=0.5)

        assert len(affected) == 1
        assert affected[0][0] == "src/b.py"
        assert affected[0][1] == 0.9

    def test_blast_radius_empty(self, monkeypatch):
        """No connections -> empty list."""
        config = {
            "version": "1.0",
            "interconnectedness": {"files": {}, "functions": {}},
        }
        monkeypatch.setattr("core.graph.interconnectedness.load_config", lambda: config)

        manager = InterconnectednessManager()

        affected = manager.get_blast_radius("src/nonexistent.py")

        assert len(affected) == 0

    def test_compute_centrality(self, monkeypatch):
        """Scores normalized."""
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
        monkeypatch.setattr("core.graph.interconnectedness.load_config", lambda: config)

        manager = InterconnectednessManager()

        centrality = manager.compute_centrality()

        assert "src/a.py" in centrality
        assert "src/b.py" in centrality
        assert "src/c.py" in centrality
        assert all(0 <= c <= 1 for c in centrality.values())

    def test_get_clusters(self, monkeypatch):
        """Connected components found."""
        config = {
            "version": "1.0",
            "interconnectedness": {
                "files": {
                    "src/a.py": {
                        "src/b.py": {"weight": 0.8, "relation": "calls"}
                    },
                    "src/c.py": {
                        "src/d.py": {"weight": 0.6, "relation": "imports"}
                    },
                },
                "functions": {},
            },
        }
        monkeypatch.setattr("core.graph.interconnectedness.load_config", lambda: config)

        manager = InterconnectednessManager()

        clusters = manager.get_clusters()

        # a->b form one component, c->d form another
        assert clusters["src/a.py"] == clusters["src/b.py"]
        assert clusters["src/c.py"] == clusters["src/d.py"]
        assert clusters["src/a.py"] != clusters["src/c.py"]

    def test_update_co_edits(self, monkeypatch):
        """Co-edit patterns create edges."""
        config = {
            "version": "1.0",
            "interconnectedness": {"files": {}, "functions": {}},
        }
        monkeypatch.setattr("core.graph.interconnectedness.load_config", lambda: config)

        manager = InterconnectednessManager()

        co_edits = {("src/a.py", "src/b.py"): 5, ("src/a.py", "src/c.py"): 1}
        manager.update_from_co_edits(co_edits, min_count=2)

        connections = manager.get_connections("src/a.py")
        assert "src/b.py" in connections
        assert "src/c.py" not in connections  # Below min_count

    def test_get_matrix(self, monkeypatch):
        """Full matrix returned."""
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
        monkeypatch.setattr("core.graph.interconnectedness.load_config", lambda: config)

        manager = InterconnectednessManager()

        matrix = manager.get_matrix()

        assert "files" in matrix
        assert "functions" in matrix
        assert "src/a.py" in matrix["files"]
