"""Tests for ImportanceScorer (integration)."""

import pytest
from unittest.mock import patch

from core.importance.scorer import ImportanceScorer


class TestImportanceScorer:
    """Tests for ImportanceScorer."""

    def test_score_session(self, sample_session_data, sample_graph_data):
        """Returns files and functions scores."""
        scorer = ImportanceScorer()
        scorer._get_graph_data = lambda x=None: sample_graph_data

        with patch.object(scorer, "_load_session", return_value=sample_session_data):
            result = scorer.score_session("test_session")

        assert "files" in result
        assert "functions" in result

    def test_score_session_empty(self):
        """Empty session -> empty."""
        scorer = ImportanceScorer()
        result = scorer.score_session("nonexistent_session")

        assert result["files"] == {}
        assert result["functions"] == {}

    def test_score_file(self, sample_session_data, sample_graph_data):
        """Single file scored."""
        scorer = ImportanceScorer()

        score = scorer.score_file("src/auth/login.py", sample_session_data, sample_graph_data)

        assert 0 <= score <= 10

    def test_hybrid_scoring(self, sample_session_data, sample_graph_data):
        """Heuristic + ML combined when model available."""
        scorer = ImportanceScorer()
        scorer._get_graph_data = lambda x=None: sample_graph_data

        with patch.object(scorer, "_load_session", return_value=sample_session_data):
            result = scorer.score_session("test_session")

        assert len(result["files"]) > 0

    def test_heuristic_only(self, sample_session_data, sample_graph_data):
        """Heuristic only when no ML."""
        scorer = ImportanceScorer()
        scorer.model.initialized = False
        scorer._get_graph_data = lambda x=None: sample_graph_data

        with patch.object(scorer, "_load_session", return_value=sample_session_data):
            result = scorer.score_session("test_session")

        assert len(result["files"]) > 0
