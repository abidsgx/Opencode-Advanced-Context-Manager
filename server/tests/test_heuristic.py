"""Tests for HeuristicScorer."""

import time
import numpy as np
import pytest

from core.importance.heuristic import HeuristicScorer


class TestHeuristicScorer:
    """Tests for HeuristicScorer."""

    def test_score_file_basic(self, sample_session_data, sample_graph_data):
        """Single file with known edits -> score in (0, 10)."""
        scorer = HeuristicScorer()
        metrics = scorer._compute_metrics(sample_session_data)

        score = scorer.score_file("src/auth/login.py", metrics, sample_graph_data)

        assert 0 <= score <= 10
        assert score > 0  # Should have some score due to edits

    def test_score_file_goal_alignment(self, sample_session_data, sample_graph_data):
        """File in goal_files -> score boosted."""
        scorer = HeuristicScorer()
        metrics = scorer._compute_metrics(sample_session_data)

        score = scorer.score_file(
            "src/auth/login.py",
            metrics,
            sample_graph_data,
            goal_files=["src/auth/login.py"],
        )

        assert score >= 8.0

    def test_score_file_error_association(self, sample_session_data, sample_graph_data):
        """File in error_files -> score boosted."""
        scorer = HeuristicScorer()
        metrics = scorer._compute_metrics(sample_session_data)

        score_without = scorer.score_file("src/auth/login.py", metrics, sample_graph_data)
        score_with = scorer.score_file(
            "src/auth/login.py",
            metrics,
            sample_graph_data,
            error_files=["src/auth/login.py"],
        )

        assert score_with > score_without

    def test_score_file_recency(self, sample_graph_data):
        """Recent edit -> higher score than old edit."""
        scorer = HeuristicScorer()

        recent_session = {
            "turns": [
                {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "tool",
                            "toolName": "file_edit",
                            "toolInput": {"path": "src/recent.py"},
                            "toolOutput": "modified",
                            "timestamp": time.time() - 10,
                        }
                    ],
                }
            ]
        }

        old_session = {
            "turns": [
                {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "tool",
                            "toolName": "file_edit",
                            "toolInput": {"path": "src/old.py"},
                            "toolOutput": "modified",
                            "timestamp": time.time() - 86400,
                        }
                    ],
                }
            ]
        }

        recent_metrics = scorer._compute_metrics(recent_session)
        old_metrics = scorer._compute_metrics(old_session)

        graph_data = {"degree_centrality": {}}

        recent_score = scorer.score_file("src/recent.py", recent_metrics, graph_data)
        old_score = scorer.score_file("src/old.py", old_metrics, graph_data)

        assert recent_score > old_score

    def test_score_file_no_edits(self, sample_graph_data):
        """File with 0 edits -> low score."""
        scorer = HeuristicScorer()
        empty_session = {"turns": []}
        metrics = scorer._compute_metrics(empty_session)

        score = scorer.score_file("src/nonexistent.py", metrics, sample_graph_data)

        assert score < 5.0

    def test_score_all_multiple_files(self, sample_session_data, sample_graph_data):
        """Ranking correct: more edits > fewer."""
        scorer = HeuristicScorer()

        result = scorer.score_all(sample_session_data, sample_graph_data)

        assert "files" in result
        assert "functions" in result
        assert len(result["files"]) > 0

        # login.py has more edits than users.py
        login_score = result["files"].get("src/auth/login.py", 0)
        users_score = result["files"].get("src/db/users.py", 0)

        assert login_score > users_score

    def test_score_function_basic(self, sample_session_data, sample_graph_data):
        """Function scoring works."""
        scorer = HeuristicScorer()
        metrics = scorer._compute_metrics(sample_session_data)

        score = scorer.score_function(
            "authenticate", "src/auth/login.py", metrics, sample_graph_data
        )

        assert 0 <= score <= 10

    def test_score_function_in_goal(self, sample_session_data, sample_graph_data):
        """Goal function -> 10.0."""
        scorer = HeuristicScorer()
        metrics = scorer._compute_metrics(sample_session_data)

        score = scorer.score_function(
            "authenticate",
            "src/auth/login.py",
            metrics,
            sample_graph_data,
            goal_functions=["src/auth/login.py::authenticate"],
        )

        assert score == 10.0

    def test_custom_weights(self, sample_session_data, sample_graph_data):
        """Different weights -> different scores."""
        scorer_default = HeuristicScorer()
        scorer_custom = HeuristicScorer(
            weights={
                "edit_frequency": 0.50,
                "dependency_centrality": 0.10,
                "goal_alignment": 0.10,
                "error_association": 0.10,
                "recency": 0.20,
            }
        )

        metrics_default = scorer_default._compute_metrics(sample_session_data)
        metrics_custom = scorer_custom._compute_metrics(sample_session_data)

        score_default = scorer_default.score_file(
            "src/auth/login.py", metrics_default, sample_graph_data
        )
        score_custom = scorer_custom.score_file(
            "src/auth/login.py", metrics_custom, sample_graph_data
        )

        # Scores should be different with different weights
        assert score_default != score_custom

    def test_score_range_clamped(self, sample_session_data, sample_graph_data):
        """All scores in [0, 10]."""
        scorer = HeuristicScorer()

        result = scorer.score_all(sample_session_data, sample_graph_data)

        for file_path, score in result["files"].items():
            assert 0 <= score <= 10, f"Score for {file_path} out of range: {score}"

        for func, score in result["functions"].items():
            assert 0 <= score <= 10, f"Score for {func} out of range: {score}"
