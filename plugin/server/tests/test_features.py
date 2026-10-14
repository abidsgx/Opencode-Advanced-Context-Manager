"""Tests for feature extraction."""

import time
import numpy as np
import pytest

from features.base import (
    FeatureVector,
    TemporalExtractor,
    StructuralExtractor,
    GraphExtractor,
    EditPatternExtractor,
    CompositeExtractor,
    build_features,
    build_all_features,
    compute_session_metrics,
    SessionMetrics,
)


class TestTemporalExtractor:
    """Tests for TemporalExtractor."""

    def test_basic(self, sample_session_data, sample_graph_data):
        """Time ratio, frequency, recency correct."""
        metrics = compute_session_metrics(sample_session_data)
        extractor = TemporalExtractor()

        fv = extractor.extract("src/auth/login.py", metrics, sample_graph_data)

        assert fv.time_spent_on_file >= 0
        assert 0 <= fv.time_spent_ratio <= 1
        assert fv.edit_frequency >= 0
        assert 0 <= fv.recency_score <= 1

    def test_no_edits(self, sample_graph_data):
        """File with no edits -> zero features."""
        metrics = SessionMetrics()
        extractor = TemporalExtractor()

        fv = extractor.extract("src/nonexistent.py", metrics, sample_graph_data)

        assert fv.time_spent_on_file == 0
        assert fv.time_spent_ratio == 0
        assert fv.edit_frequency == 0


class TestStructuralExtractor:
    """Tests for StructuralExtractor."""

    def test_basic(self, sample_session_data, sample_graph_data):
        """Lines, functions, call depth correct."""
        metrics = compute_session_metrics(sample_session_data)
        extractor = StructuralExtractor()

        fv = extractor.extract("src/auth/login.py", metrics, sample_graph_data)

        assert fv.lines_changed >= 0
        assert fv.lines_changed_ratio >= 0
        assert fv.functions_modified >= 0
        assert fv.call_depth >= 0

    def test_functions_count(self, sample_session_data, sample_graph_data):
        """Functions modified count correct."""
        metrics = compute_session_metrics(sample_session_data)
        extractor = StructuralExtractor()

        fv = extractor.extract("src/auth/login.py", metrics, sample_graph_data)

        # login.py has 2 functions modified
        assert fv.functions_modified == 2


class TestGraphExtractor:
    """Tests for GraphExtractor."""

    def test_basic(self, sample_session_data, sample_graph_data):
        """Centrality, betweenness, cluster correct."""
        metrics = compute_session_metrics(sample_session_data)
        extractor = GraphExtractor()

        fv = extractor.extract("src/auth/login.py", metrics, sample_graph_data)

        assert 0 <= fv.degree_centrality <= 1
        assert 0 <= fv.betweenness_centrality <= 1
        assert fv.cluster_id >= 0
        assert fv.dependency_count >= 0
        assert fv.dependents_count >= 0
        assert 0 <= fv.cluster_isolation <= 1

    def test_centrality_values(self, sample_session_data, sample_graph_data):
        """Centrality values match graph data."""
        metrics = compute_session_metrics(sample_session_data)
        extractor = GraphExtractor()

        fv = extractor.extract("src/auth/login.py", metrics, sample_graph_data)

        assert fv.degree_centrality == pytest.approx(0.8, abs=0.01)
        assert fv.betweenness_centrality == pytest.approx(0.6, abs=0.01)


class TestEditPatternExtractor:
    """Tests for EditPatternExtractor."""

    def test_basic(self, sample_session_data, sample_graph_data):
        """Reverts, iterations counted."""
        metrics = compute_session_metrics(sample_session_data)
        extractor = EditPatternExtractor()

        fv = extractor.extract("src/auth/login.py", metrics, sample_graph_data)

        assert fv.revert_count >= 0
        assert fv.iteration_count >= 0
        assert fv.cross_file_edits >= 0

    def test_revert_detected(self, sample_session_data, sample_graph_data):
        """Revert pattern detected."""
        metrics = compute_session_metrics(sample_session_data)
        extractor = EditPatternExtractor()

        fv = extractor.extract("src/auth/login.py", metrics, sample_graph_data)

        # login.py has a revert pattern (edited, then another file, then edited again)
        assert fv.revert_count > 0


class TestCompositeExtractor:
    """Tests for CompositeExtractor."""

    def test_basic(self, sample_session_data, sample_graph_data):
        """Products computed correctly."""
        metrics = compute_session_metrics(sample_session_data)

        temporal = TemporalExtractor()
        structural = StructuralExtractor()
        graph = GraphExtractor()
        edit_pattern = EditPatternExtractor()
        composite = CompositeExtractor(temporal, structural, graph, edit_pattern)

        fv = composite.extract("src/auth/login.py", metrics, sample_graph_data)

        # Check composite features
        assert fv.dependency_x_time == pytest.approx(
            fv.dependency_count * fv.time_spent_ratio, abs=0.01
        )
        assert fv.centrality_x_frequency == pytest.approx(
            fv.degree_centrality * fv.edit_frequency, abs=0.01
        )


class TestBuildFeatures:
    """Tests for build_features pipeline."""

    def test_full_pipeline(self, sample_session_data, sample_graph_data):
        """All extractors -> 19-dim vector."""
        metrics = compute_session_metrics(sample_session_data)
        features = build_features("src/auth/login.py", metrics, sample_graph_data)

        assert isinstance(features, np.ndarray)
        assert features.shape == (19,)

    def test_all_features(self, sample_session_data, sample_graph_data):
        """Multiple files -> dict of arrays."""
        result = build_all_features(sample_session_data, sample_graph_data)

        assert isinstance(result, dict)
        assert len(result) > 0

        for file_path, features in result.items():
            assert isinstance(features, np.ndarray)
            assert features.shape == (19,)


class TestComputeSessionMetrics:
    """Tests for compute_session_metrics."""

    def test_reverts_detected(self, sample_session_data):
        """Revert pattern detected."""
        metrics = compute_session_metrics(sample_session_data)

        # login.py should have reverts
        assert "src/auth/login.py" in metrics.revert_counts
        assert metrics.revert_counts["src/auth/login.py"] > 0

    def test_iterations_counted(self, sample_session_data):
        """Iteration count correct."""
        metrics = compute_session_metrics(sample_session_data)

        # login.py has 3 edits -> 1 iteration
        assert "src/auth/login.py" in metrics.iteration_counts
        assert metrics.iteration_counts["src/auth/login.py"] >= 1

    def test_cross_file_pairs(self, sample_session_data):
        """Co-edit pairs detected."""
        metrics = compute_session_metrics(sample_session_data)

        # There should be co-edit pairs
        assert len(metrics.cross_file_pairs) > 0

    def test_edits_per_file(self, sample_session_data):
        """Edit counts correct."""
        metrics = compute_session_metrics(sample_session_data)

        assert metrics.edits_per_file["src/auth/login.py"] == 3
        assert metrics.edits_per_file["src/db/users.py"] == 1


class TestFeatureVector:
    """Tests for FeatureVector dataclass."""

    def test_to_array(self, sample_feature_vector):
        """Array conversion correct shape (19,)."""
        fv = FeatureVector()
        for i, name in enumerate(FeatureVector.feature_names()):
            setattr(fv, name, sample_feature_vector[i])

        arr = fv.to_array()

        assert arr.shape == (19,)
        np.testing.assert_array_equal(arr, sample_feature_vector)

    def test_feature_names(self):
        """19 feature names returned."""
        names = FeatureVector.feature_names()

        assert len(names) == 19
        assert "time_spent_on_file" in names
        assert "dependency_x_time" in names
        assert "cluster_isolation" in names
