"""Tests for ModelEvaluator."""

import numpy as np
import pytest

from training.evaluator import ModelEvaluator


class TestModelEvaluator:
    """Tests for ModelEvaluator."""

    def test_evaluate_basic(self):
        """Accuracy, precision, recall, F1."""
        evaluator = ModelEvaluator()

        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0])

        result = evaluator.evaluate(y_true, y_pred)

        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert 0 <= result["accuracy"] <= 1
        assert 0 <= result["precision"] <= 1
        assert 0 <= result["recall"] <= 1
        assert 0 <= result["f1"] <= 1

    def test_evaluate_with_prob(self):
        """Log loss computed."""
        evaluator = ModelEvaluator()

        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0])
        y_prob = np.array(
            [
                [0.9, 0.1],
                [0.4, 0.6],
                [0.2, 0.8],
                [0.1, 0.9],
                [0.8, 0.2],
                [0.6, 0.4],
                [0.7, 0.3],
                [0.3, 0.7],
                [0.2, 0.8],
                [0.5, 0.5],
            ]
        )

        result = evaluator.evaluate(y_true, y_pred, y_prob)

        assert "log_loss" in result
        assert result["log_loss"] is not None
        assert result["log_loss"] >= 0

    def test_evaluate_perfect(self):
        """Perfect -> accuracy=1.0."""
        evaluator = ModelEvaluator()

        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])

        result = evaluator.evaluate(y_true, y_pred)

        assert result["accuracy"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_compare_models(self):
        """Best model identified."""
        evaluator = ModelEvaluator()

        results = [
            {"accuracy": 0.8, "f1": 0.75},
            {"accuracy": 0.9, "f1": 0.85},
            {"accuracy": 0.85, "f1": 0.80},
        ]

        comparison = evaluator.compare_models(results)

        assert comparison["n_models"] == 3
        assert comparison["best_accuracy"] == 0.9
        assert comparison["best_f1"] == 0.85
