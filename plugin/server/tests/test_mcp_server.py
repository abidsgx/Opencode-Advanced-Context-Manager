"""Tests for MCP server tools (unit-level, no process communication)."""

import pytest
from unittest.mock import MagicMock, patch

from mcp_server import (
    score_importance,
    get_interconnectedness,
    record_edit,
    get_pending_feedback,
    get_session_summary,
    get_languages,
    submit_labels,
)


class TestMcpServerTools:
    """Test MCP server tools as function calls."""

    def test_score_importance(self):
        with patch("core.importance.scorer.ImportanceScorer") as MockScorer:
            scorer = MockScorer.return_value
            scorer.score_session.return_value = {
                "files": {"src/auth/login.py": 8.5},
                "functions": {"src/auth/login.py::authenticate": 9.0},
            }
            result = score_importance("test_session", "")
            assert "files" in result
            assert "functions" in result

    def test_get_interconnectedness(self):
        with patch("core.graph.interconnectedness.InterconnectednessManager") as MockManager:
            manager = MockManager.return_value
            manager.get_matrix.return_value = {
                "files": {"src/a.py": {"src/b.py": {"weight": 0.8, "relation": "calls"}}},
                "functions": {},
            }
            result = get_interconnectedness("test_session", "")
            assert "files" in result

    def test_record_edit(self):
        with patch("core.session.store.SessionStore") as MockStore:
            store = MockStore.return_value
            store.record_edit.return_value = {"status": "ok", "edit": {"file": "src/test.py"}}
            result = record_edit("test_session", "src/test.py", "func", "Fix bug", "Fixes bug")
            assert result["status"] == "ok"

    def test_get_pending_feedback(self):
        with patch("core.importance.active_learning.ActiveLearner") as MockAL:
            al = MockAL.return_value
            al.get_pending_samples.return_value = {"samples": [], "count": 0}
            result = get_pending_feedback()
            assert "samples" in result

    def test_get_session_summary(self):
        with patch("core.session.store.SessionStore") as MockStore:
            store = MockStore.return_value
            store.get_summary.return_value = {
                "session_id": "test",
                "turn_count": 5,
                "edit_count": 3,
            }
            result = get_session_summary("test")
            assert "session_id" in result

    def test_get_languages(self):
        with patch("core.graph.tree_sitter.load_config") as mock_config:
            mock_config.return_value = {
                "languages": {
                    "python": {"enabled": True, "package": "tree-sitter-python", "extensions": [".py"]},
                }
            }
            result = get_languages()
            assert "enabled" in result
            assert "python" in result["enabled"]
            assert "extension_map" in result

    def test_submit_labels(self):
        with patch("core.importance.active_learning.ActiveLearner") as MockAL:
            al = MockAL.return_value
            al.submit_labels.return_value = {"status": "ok", "count": 2}
            result = submit_labels([{"sample_id": "s1", "score": 8.0}])
            assert result["status"] == "ok"
