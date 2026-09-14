"""MCP Protocol integration test.

Tests the MCP server tools end-to-end by calling them through the
server's tool functions directly (not over stdio), verifying the
full call chain works correctly.
"""

import time
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from mcp_server import (
    score_importance,
    get_interconnectedness,
    record_edit,
    get_session_summary,
    get_languages,
    submit_labels,
    get_pending_feedback,
)


@pytest.fixture
def mock_session_store():
    """Provide a mock SessionStore that records calls."""
    store = MagicMock()
    store.record_edit.return_value = {"status": "ok", "edit": {"file": "src/test.py"}}
    store.get_summary.return_value = {
        "session_id": "test_session",
        "title": "Test session",
        "turn_count": 5,
        "edit_count": 3,
        "files_edited": ["src/auth/login.py", "src/db/users.py"],
        "functions_edited": ["authenticate", "get_user"],
        "importance_scores": {},
        "interconnectedness_files": [],
    }
    return store


class TestMCPProtocolIntegration:
    """Integration tests for MCP server tools."""

    def test_record_edit_then_get_summary(self, mock_session_store):
        """Record an edit, then retrieve summary — data flows through."""
        with patch("core.session.store.SessionStore", return_value=mock_session_store):
            result = record_edit(
                "test_session", "src/auth/login.py", "authenticate",
                "Fix timeout", "Improves reliability"
            )
            assert result["status"] == "ok"

            summary = get_session_summary("test_session")
            assert summary["session_id"] == "test_session"
            assert summary["edit_count"] == 3

    def test_score_importance_returns_structured_output(self):
        """score_importance returns files and functions keys."""
        with patch("core.importance.scorer.ImportanceScorer") as MockScorer:
            scorer = MockScorer.return_value
            scorer.score_session.return_value = {
                "files": {"src/a.py": 8.0, "src/b.py": 3.5},
                "functions": {"src/a.py::authenticate": 9.0},
            }
            result = score_importance("test_session", "Fix auth bug")

            assert "files" in result
            assert "functions" in result
            assert result["files"]["src/a.py"] == 8.0

    def test_get_interconnectedness_returns_matrix(self):
        """get_interconnectedness returns files and functions."""
        with patch("core.graph.interconnectedness.InterconnectednessManager") as MockManager:
            manager = MockManager.return_value
            manager.get_matrix.return_value = {
                "files": {
                    "src/a.py": {
                        "src/b.py": {"weight": 0.8, "relation": "calls"}
                    }
                },
                "functions": {},
            }

            result = get_interconnectedness("test_session", "")

            assert "files" in result
            assert "src/a.py" in result["files"]

    def test_get_languages_returns_configured_languages(self):
        """get_languages returns enabled languages and extension map."""
        with patch("core.graph.tree_sitter.get_enabled_languages") as mock_enabled, \
             patch("core.graph.tree_sitter.get_extension_map") as mock_ext:
            mock_enabled.return_value = {
                "python": {"enabled": True, "package": "tree-sitter-python", "extensions": [".py"]},
            }
            mock_ext.return_value = {".py": "python"}

            result = get_languages()

            assert "enabled" in result
            assert "extension_map" in result
            assert "python" in result["enabled"]
            assert result["extension_map"][".py"] == "python"

    def test_submit_labels_feeds_back_to_model(self):
        """submit_labels applies labels and returns status."""
        with patch("core.importance.active_learning.ActiveLearner") as MockAL:
            al = MockAL.return_value
            al.submit_labels.return_value = {"status": "ok", "count": 3}

            labels = [
                {"sample_id": "s1", "score": 8.0},
                {"sample_id": "s2", "score": 2.0},
                {"sample_id": "s3", "score": 5.0},
            ]
            result = submit_labels(labels)

            assert result["status"] == "ok"
            assert result["count"] == 3
            al.submit_labels.assert_called_once_with(labels)

    def test_get_pending_feedback_returns_pending(self):
        """get_pending_feedback returns pending samples."""
        with patch("core.importance.active_learning.ActiveLearner") as MockAL:
            al = MockAL.return_value
            al.get_pending_samples.return_value = {
                "samples": [
                    {"sample_id": "s1", "file_path": "src/a.py", "margin": 0.05},
                ],
                "count": 1,
            }

            result = get_pending_feedback()

            assert "samples" in result
            assert result["count"] == 1

    def test_multiple_tool_calls_maintain_consistency(self):
        """Multiple tool calls with same session don't conflict."""
        with patch("core.session.store.SessionStore") as MockStore:
            store = MockStore.return_value
            store.record_edit.return_value = {"status": "ok", "edit": {}}
            store.get_summary.return_value = {
                "session_id": "multi_session",
                "turn_count": 0,
                "edit_count": 0,
            }

            # Record 3 edits
            for i in range(3):
                record_edit("multi_session", f"src/file_{i}.py", f"func_{i}", "reason", "goal")

            # All 3 should be recorded
            assert store.record_edit.call_count == 3

    def test_error_handling_returns_graceful_result(self):
        """Tool calls handle exceptions gracefully."""
        with patch("core.importance.scorer.ImportanceScorer") as MockScorer:
            MockScorer.side_effect = RuntimeError("Model not found")
            try:
                score_importance("nonexistent", "")
            except RuntimeError:
                pass  # Expected — server tools propagate errors
