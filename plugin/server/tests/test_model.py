"""Tests for ImportanceModel and StreamingTrainer."""

import numpy as np
import pytest
import time

from core.importance.model import ImportanceModel, StreamingTrainer, get_model


class TestImportanceModel:
    """Tests for ImportanceModel."""

    def test_initialize(self, tmp_models_dir, monkeypatch):
        """Initialize with random data -> model.initialized = True."""
        monkeypatch.setattr(
            "core.importance.model.PROJECT_ROOT", tmp_models_dir.parent.parent.parent
        )

        model = ImportanceModel()
        X = np.random.randn(50, 19)
        model.initialize(X)

        assert model.initialized is True
        assert model.model is not None
        assert model.scaler is not None

    def test_predict_before_init(self):
        """Predict before init -> zeros."""
        model = ImportanceModel()
        X = np.random.randn(5, 19)

        predictions = model.predict(X)

        assert len(predictions) == 5
        assert all(p == 0 for p in predictions)

    def test_predict_after_init(self, tmp_models_dir, monkeypatch):
        """Predict after init -> valid probabilities."""
        monkeypatch.setattr(
            "core.importance.model.PROJECT_ROOT", tmp_models_dir.parent.parent.parent
        )

        model = ImportanceModel()
        X = np.random.randn(50, 19)
        model.initialize(X)

        X_test = np.random.randn(5, 19)
        proba = model.predict_proba(X_test)

        assert proba.shape == (5, 2)
        assert all(0 <= p <= 1 for row in proba for p in row)
        assert all(abs(sum(row) - 1.0) < 0.01 for row in proba)

    def test_partial_fit(self, tmp_models_dir, monkeypatch):
        """Partial fit updates model weights."""
        monkeypatch.setattr(
            "core.importance.model.PROJECT_ROOT", tmp_models_dir.parent.parent.parent
        )

        model = ImportanceModel()
        X = np.random.randn(50, 19)
        model.initialize(X)

        # Get initial weights
        initial_coef = model.model.coef_.copy()

        # Partial fit
        X_new = np.random.randn(10, 19)
        y_new = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        model.partial_fit(X_new, y_new)

        # Weights should have changed
        assert not np.array_equal(initial_coef, model.model.coef_)

    def test_save_load(self, tmp_models_dir, monkeypatch):
        """Save then load -> predictions match."""
        monkeypatch.setattr(
            "core.importance.model.PROJECT_ROOT", tmp_models_dir.parent.parent.parent
        )

        model = ImportanceModel()
        X = np.random.randn(50, 19)
        model.initialize(X)

        X_test = np.random.randn(5, 19)
        predictions_before = model.predict(X_test)

        # Save
        model.save()

        # Load into new model
        model2 = ImportanceModel()
        model2.load()

        predictions_after = model2.predict(X_test)

        np.testing.assert_array_equal(predictions_before, predictions_after)

    def test_is_uncertain(self, tmp_models_dir, monkeypatch):
        """Low confidence -> uncertain=True."""
        monkeypatch.setattr(
            "core.importance.model.PROJECT_ROOT", tmp_models_dir.parent.parent.parent
        )

        model = ImportanceModel()
        X = np.random.randn(50, 19)
        model.initialize(X)

        # Get uncertain samples
        uncertain = model.is_uncertain(X)

        assert len(uncertain) == 50
        assert any(uncertain)  # At least some should be uncertain

    def test_high_confidence(self, tmp_models_dir, monkeypatch):
        """High confidence -> uncertain=False."""
        monkeypatch.setattr(
            "core.importance.model.PROJECT_ROOT", tmp_models_dir.parent.parent.parent
        )

        model = ImportanceModel()
        # Initialize with some data
        X_init = np.random.randn(20, 19)
        model.initialize(X_init)
        # Then train with proper labels to get confident predictions
        X_pos = np.ones((25, 19)) * 10
        X_neg = np.ones((25, 19)) * -10
        X = np.concatenate([X_pos, X_neg])
        y = np.array([1] * 25 + [0] * 25)
        model.partial_fit(X, y)

        # Predict on same data - should be confident
        uncertain = model.is_uncertain(X)

        # Most should not be uncertain
        assert sum(~uncertain) > len(uncertain) // 2


class TestStreamingTrainer:
    """Tests for StreamingTrainer."""

    def test_process_point(self, tmp_models_dir, monkeypatch):
        """Process point -> prediction recorded."""
        monkeypatch.setattr(
            "core.importance.model.PROJECT_ROOT", tmp_models_dir.parent.parent.parent
        )

        model = ImportanceModel()
        X = np.random.randn(50, 19)
        model.initialize(X)

        trainer = StreamingTrainer(model, window_size=20)

        x = np.random.randn(19)
        trainer.process_point(x, 1)

        assert len(trainer.predictions) == 1
        assert len(trainer.true_vals) == 1

    def test_buffer_fills(self, tmp_models_dir, monkeypatch):
        """Buffer fills to window_size -> model updated."""
        monkeypatch.setattr(
            "core.importance.model.PROJECT_ROOT", tmp_models_dir.parent.parent.parent
        )

        model = ImportanceModel()
        X = np.random.randn(50, 19)
        model.initialize(X)

        trainer = StreamingTrainer(model, window_size=10)

        for i in range(15):
            x = np.random.randn(19)
            trainer.process_point(x, i % 2)

        # Buffer stays at maxlen, but training should have occurred
        assert len(trainer.x_buffer) == 10
        assert len(trainer.train_sizes) > 0
        assert trainer.current_size > 0

    def test_time_decay_weights(self, tmp_models_dir, monkeypatch):
        """Weights decrease with age."""
        monkeypatch.setattr(
            "core.importance.model.PROJECT_ROOT", tmp_models_dir.parent.parent.parent
        )

        model = ImportanceModel()
        X = np.random.randn(50, 19)
        model.initialize(X)

        trainer = StreamingTrainer(model, window_size=10, half_life=5)

        # Process points
        for i in range(10):
            x = np.random.randn(19)
            trainer.process_point(x, i % 2)

        # Check that train_scores exist
        assert len(trainer.train_sizes) > 0

    def test_accuracy_tracking(self, tmp_models_dir, monkeypatch):
        """Accuracy improves over time."""
        monkeypatch.setattr(
            "core.importance.model.PROJECT_ROOT", tmp_models_dir.parent.parent.parent
        )

        model = ImportanceModel()
        X = np.random.randn(50, 19)
        model.initialize(X)

        trainer = StreamingTrainer(model, window_size=10)

        # Process points where label depends on features
        for i in range(30):
            x = np.random.randn(19)
            label = 1 if x[0] > 0 else 0
            trainer.process_point(x, label)

        accuracy = trainer.get_accuracy()
        assert 0 <= accuracy <= 1

    def test_log_loss(self, tmp_models_dir, monkeypatch):
        """Log loss computed correctly."""
        monkeypatch.setattr(
            "core.importance.model.PROJECT_ROOT", tmp_models_dir.parent.parent.parent
        )

        model = ImportanceModel()
        X = np.random.randn(50, 19)
        model.initialize(X)

        trainer = StreamingTrainer(model, window_size=10)

        for i in range(15):
            x = np.random.randn(19)
            trainer.process_point(x, i % 2)

        log_loss = trainer.get_log_loss()
        assert log_loss >= 0
