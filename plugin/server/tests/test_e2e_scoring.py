"""End-to-end scoring integration test.

Tests the full pipeline: session data -> feature extraction -> heuristic + ML scoring -> output.
"""

import time
import numpy as np
import pytest

from core.importance.heuristic import HeuristicScorer
from features.base import compute_session_metrics, build_all_features
from core.graph.interconnectedness import InterconnectednessManager


class TestE2EScoring:
    """Full pipeline scoring tests."""

    def _make_session(self, edits):
        """Build a session dict from a list of (file, func, timestamp_offset_sec) tuples."""
        turns = []
        for file_path, func, offset in edits:
            turns.append({
                "role": "assistant",
                "parts": [{
                    "type": "tool",
                    "toolName": "file_edit",
                    "toolInput": {"path": file_path, "function": func},
                    "toolOutput": f"modified {file_path}",
                    "timestamp": time.time() - offset,
                }],
            })
        return {
            "_id": "e2e_session",
            "title": "E2E test session",
            "startedAt": time.time() - 3600,
            "updatedAt": time.time(),
            "turns": turns,
            "editImpact": [],
        }

    def test_heuristic_scores_ranked_correctly(self):
        """Files with more edits score higher."""
        session = self._make_session([
            ("src/auth/login.py", "authenticate", 300),
            ("src/auth/login.py", "validate_token", 200),
            ("src/auth/login.py", "hash_password", 100),
            ("src/db/users.py", "get_user", 50),
        ])

        graph_data = {
            "degree_centrality": {"src/auth/login.py": 0.8, "src/db/users.py": 0.3},
            "betweenness_centrality": {},
            "clusters": {},
            "dependencies": {},
            "dependents": {},
        }

        scorer = HeuristicScorer()
        result = scorer.score_all(session, graph_data)

        assert "src/auth/login.py" in result["files"]
        assert "src/db/users.py" in result["files"]
        assert result["files"]["src/auth/login.py"] > result["files"]["src/db/users.py"]

    def test_feature_extraction_matches_session_edits(self):
        """Extracted features correspond to session edits."""
        session = self._make_session([
            ("src/auth/login.py", "authenticate", 300),
            ("src/db/users.py", "get_user", 100),
        ])

        graph_data = {
            "degree_centrality": {"src/auth/login.py": 0.5, "src/db/users.py": 0.3},
            "betweenness_centrality": {},
            "clusters": {},
            "dependencies": {},
            "dependents": {},
            "file_lines": {},
            "lines_changed": {},
            "call_depth": {},
        }

        features = build_all_features(session, graph_data)

        assert "src/auth/login.py" in features
        assert "src/db/users.py" in features

        metrics = compute_session_metrics(session)
        # login.py edited once, users.py edited once
        assert metrics.edits_per_file["src/auth/login.py"] == 1
        assert metrics.edits_per_file["src/db/users.py"] == 1
        assert metrics.total_edits == 2

    def test_scoring_produces_valid_scores(self):
        """All scores in [0, 10]."""
        session = self._make_session([
            ("src/auth/login.py", "authenticate", 300),
            ("src/db/users.py", "get_user", 100),
            ("src/utils/helpers.py", None, 50),
        ])

        graph_data = {
            "degree_centrality": {"src/auth/login.py": 0.8, "src/db/users.py": 0.4, "src/utils/helpers.py": 0.1},
            "betweenness_centrality": {},
            "clusters": {},
            "dependencies": {},
            "dependents": {},
        }

        scorer = HeuristicScorer()
        result = scorer.score_all(session, graph_data)

        for file_path, score in result["files"].items():
            assert 0 <= score <= 10, f"Score for {file_path} out of range: {score}"

    def test_goal_files_get_higher_scores(self):
        """Goal-aligned files score higher than non-goal files."""
        session = self._make_session([
            ("src/auth/login.py", "authenticate", 300),
            ("src/db/users.py", "get_user", 300),
        ])

        graph_data = {"degree_centrality": {}}

        scorer = HeuristicScorer()
        metrics = scorer._compute_metrics(session)

        score_goal = scorer.score_file(
            "src/auth/login.py", metrics, graph_data,
            goal_files=["src/auth/login.py"]
        )
        score_no_goal = scorer.score_file(
            "src/db/users.py", metrics, graph_data,
            goal_files=["src/auth/login.py"]
        )

        assert score_goal > score_no_goal

    def test_interconnectedness_influences_scoring(self):
        """Files with more connections get higher centrality."""
        manager = InterconnectednessManager()
        manager.add_edge("src/a.py", "src/b.py", 0.8, "calls")
        manager.add_edge("src/a.py", "src/c.py", 0.7, "imports")
        manager.add_edge("src/a.py", "src/d.py", 0.6, "calls")

        centrality = manager.compute_centrality()

        assert centrality["src/a.py"] > centrality["src/b.py"]
        assert centrality["src/a.py"] > centrality["src/c.py"]

    def test_synthetic_data_feeds_into_scoring(self):
        """Generated synthetic data can be used for model training."""
        from training.synthetic import SyntheticGenerator

        gen = SyntheticGenerator()
        samples = gen.generate_from_rules(50)

        assert len(samples) == 50
        X = np.array([s["features"] for s in samples])
        y = np.array([s["label"] for s in samples])

        assert X.shape == (50, 19)
        assert set(y).issubset({0, 1})

    def test_online_learner_processes_session_data(self):
        """OnlineLearner can process features from a real session."""
        from training.online_learner import OnlineLearner

        session = self._make_session([
            ("src/auth/login.py", "authenticate", 300),
            ("src/db/users.py", "get_user", 100),
        ])

        graph_data = {
            "degree_centrality": {"src/auth/login.py": 0.5, "src/db/users.py": 0.3},
            "betweenness_centrality": {},
            "clusters": {},
            "dependencies": {},
            "dependents": {},
            "file_lines": {},
            "lines_changed": {},
            "call_depth": {},
        }

        features = build_all_features(session, graph_data)

        learner = OnlineLearner()
        learner.load_or_init(n_features=19)

        for file_path, feat_vec in features.items():
            learner.add_sample(feat_vec, 1, timestamp=time.time())

        assert len(learner.buffer) == len(features)

    def test_multi_edits_inflate_heuristic_score(self):
        """Repeated edits to same file increase its score."""
        session_single = self._make_session([
            ("src/auth/login.py", "authenticate", 300),
        ])
        session_triple = self._make_session([
            ("src/auth/login.py", "authenticate", 300),
            ("src/auth/login.py", "validate_token", 200),
            ("src/auth/login.py", "hash_password", 100),
        ])

        graph_data = {"degree_centrality": {}}
        scorer = HeuristicScorer()

        score_single = scorer.score_all(session_single, graph_data)["files"]["src/auth/login.py"]
        score_triple = scorer.score_all(session_triple, graph_data)["files"]["src/auth/login.py"]

        assert score_triple >= score_single
