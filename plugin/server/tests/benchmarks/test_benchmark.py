"""Benchmark tests for performance-critical operations.

Measures execution time for key operations to catch regressions.
Run with: pytest tests/benchmarks/ -v -s
"""

import time
import numpy as np
import pytest

from core.importance.heuristic import HeuristicScorer
from features.base import build_all_features, compute_session_metrics
from training.synthetic import SyntheticGenerator
from training.online_learner import OnlineLearner


def _make_large_session(n_files=20, edits_per_file=5):
    """Create a session with many files and repeated edits per file."""
    import time as _time
    turns = []
    for j in range(edits_per_file):
        for i in range(n_files):
            turns.append({
                "role": "assistant",
                "parts": [{
                    "type": "tool",
                    "toolName": "file_edit",
                    "toolInput": {"path": f"src/module_{i}.py", "function": f"func_{j}"},
                    "toolOutput": f"modified module_{i}.py",
                    "timestamp": _time.time() - (edits_per_file - j) * n_files * 10 - i,
                }],
            })
    return {
        "_id": "bench_session",
        "title": "Benchmark session",
        "startedAt": _time.time() - 3600,
        "updatedAt": _time.time(),
        "turns": turns,
        "editImpact": [],
    }


def _make_graph_data(n_files=20):
    """Create graph data for many files."""
    centrality = {f"src/module_{i}.py": np.random.uniform(0.1, 0.9) for i in range(n_files)}
    return {
        "degree_centrality": centrality,
        "betweenness_centrality": {k: v * 0.5 for k, v in centrality.items()},
        "clusters": {k: i % 3 for i, k in enumerate(centrality)},
        "dependencies": {k: [] for k in centrality},
        "dependents": {k: [] for k in centrality},
        "file_lines": {k: np.random.randint(50, 500) for k in centrality},
        "lines_changed": {k: np.random.randint(1, 50) for k in centrality},
        "call_depth": {k: np.random.randint(1, 5) for k in centrality},
    }


class TestBenchmarks:
    """Performance benchmarks."""

    def test_heuristic_scoring_100_files(self):
        """Heuristic scoring of 100 unique files completes under 1s."""
        session = _make_large_session(n_files=100, edits_per_file=1)
        graph_data = _make_graph_data(n_files=100)

        scorer = HeuristicScorer()

        start = time.perf_counter()
        result = scorer.score_all(session, graph_data)
        elapsed = time.perf_counter() - start

        assert len(result["files"]) == 100
        assert elapsed < 1.0, f"Heuristic scoring took {elapsed:.3f}s (>1s)"

    def test_feature_extraction_50_files(self):
        """Feature extraction for 50 files completes under 2s."""
        session = _make_large_session(n_files=50, edits_per_file=2)
        graph_data = _make_graph_data(n_files=50)

        start = time.perf_counter()
        features = build_all_features(session, graph_data)
        elapsed = time.perf_counter() - start

        assert len(features) > 0
        assert elapsed < 2.0, f"Feature extraction took {elapsed:.3f}s (>2s)"

    def test_synthetic_generation_1000_samples(self):
        """Generating 1000 synthetic samples completes under 1s."""
        gen = SyntheticGenerator()

        start = time.perf_counter()
        samples = gen.generate_from_rules(1000)
        elapsed = time.perf_counter() - start

        assert len(samples) == 1000
        assert elapsed < 1.0, f"Synthetic generation took {elapsed:.3f}s (>1s)"

    def test_online_learner_fit_500_samples(self):
        """Online learner processing 500 samples completes under 2s."""
        learner = OnlineLearner()
        learner.load_or_init(n_features=19)

        X = np.random.randn(500, 19)
        y = np.random.randint(0, 2, 500)

        start = time.perf_counter()
        for i in range(500):
            learner.add_sample(X[i], int(y[i]))
        learner.update()
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"Online learner took {elapsed:.3f}s (>2s)"

    def test_feature_extraction_single_file(self):
        """Feature extraction for a single file is fast (<50ms)."""
        session = _make_large_session(n_files=1, edits_per_file=3)
        graph_data = _make_graph_data(n_files=1)

        start = time.perf_counter()
        features = build_all_features(session, graph_data)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.05, f"Single file extraction took {elapsed*1000:.1f}ms (>50ms)"

    def test_heuristic_scoring_10_files(self):
        """Heuristic scoring of 10 files is fast (<100ms)."""
        session = _make_large_session(n_files=10, edits_per_file=2)
        graph_data = _make_graph_data(n_files=10)

        scorer = HeuristicScorer()

        start = time.perf_counter()
        result = scorer.score_all(session, graph_data)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1, f"Heuristic scoring took {elapsed*1000:.1f}ms (>100ms)"

    def test_session_metrics_computation(self):
        """Session metrics computation for large session is fast (<100ms)."""
        session = _make_large_session(n_files=50, edits_per_file=5)

        start = time.perf_counter()
        metrics = compute_session_metrics(session)
        elapsed = time.perf_counter() - start

        assert len(metrics.edits_per_file) == 50
        assert elapsed < 0.1, f"Metrics computation took {elapsed*1000:.1f}ms (>100ms)"
