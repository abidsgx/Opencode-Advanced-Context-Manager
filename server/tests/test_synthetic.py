"""Tests for SyntheticGenerator."""

import numpy as np
import pytest

from training.synthetic import SyntheticGenerator


class TestSyntheticGenerator:
    """Tests for SyntheticGenerator."""

    def test_generate_from_rules(self):
        """Generates n_samples."""
        gen = SyntheticGenerator()
        samples = gen.generate_from_rules(100)

        assert len(samples) == 100
        assert all("features" in s for s in samples)
        assert all("label" in s for s in samples)
        assert all(s["label"] in (0, 1) for s in samples)

    def test_generate_balance(self):
        """Roughly balanced."""
        gen = SyntheticGenerator()
        samples = gen.generate_from_rules(200)

        labels = [s["label"] for s in samples]
        n_positive = sum(labels)
        n_negative = len(labels) - n_positive

        # Should be roughly balanced (within 30%)
        assert abs(n_positive - n_negative) < len(labels) * 0.3

    def test_generate_from_session(self, sample_session_data, sample_graph_data):
        """Session -> labeled samples."""
        gen = SyntheticGenerator()
        samples = gen.generate_from_session(sample_session_data, sample_graph_data)

        assert len(samples) > 0
        assert all("file_path" in s for s in samples)
        assert all("features" in s for s in samples)
        assert all("label" in s for s in samples)

    def test_balance_dataset(self):
        """Oversamples minority."""
        gen = SyntheticGenerator()

        # Create imbalanced dataset
        X = np.random.randn(100, 19)
        y = np.array([0] * 90 + [1] * 10)

        X_bal, y_bal = gen.balance_dataset(X, y)

        assert len(X_bal) == len(y_bal)
        # Should be balanced now
        n_pos = sum(y_bal == 1)
        n_neg = sum(y_bal == 0)
        assert abs(n_pos - n_neg) < 5  # Small difference due to randomness
