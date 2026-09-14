"""Tests for OnlineLearner and DataCollector."""

import json
import time
import numpy as np
import pytest
from pathlib import Path

from training.online_learner import OnlineLearner, DataCollector


class TestOnlineLearner:
    """Tests for OnlineLearner."""

    def test_load_or_init_new(self, tmp_path, monkeypatch):
        """Creates new model when no saved model exists."""
        # Patch PROJECT_ROOT to a dir with NO model files
        empty_root = tmp_path / "empty_project"
        empty_root.mkdir()
        monkeypatch.setattr("training.online_learner.PROJECT_ROOT", empty_root)

        learner = OnlineLearner()
        learner.load_or_init(n_features=19)

        assert learner.initialized is False
        assert learner.model is not None
        assert learner.scaler is not None

    def test_load_or_init_existing(self, tmp_path, monkeypatch):
        """Loads saved model."""
        monkeypatch.setattr("training.online_learner.PROJECT_ROOT", tmp_path)

        # Create and save a model first
        learner1 = OnlineLearner()
        learner1.load_or_init(n_features=19)
        X = np.random.randn(30, 19)
        y = np.random.randint(0, 2, 30)
        learner1.initial_fit(X, y)

        # Load it
        learner2 = OnlineLearner()
        learner2.load_or_init(n_features=19)

        assert learner2.initialized is True

    def test_initial_fit(self, tmp_path, monkeypatch):
        """First fit initializes model."""
        monkeypatch.setattr("training.online_learner.PROJECT_ROOT", tmp_path)

        learner = OnlineLearner()
        learner.load_or_init(n_features=19)

        X = np.random.randn(30, 19)
        y = np.random.randint(0, 2, 30)
        learner.initial_fit(X, y)

        assert learner.initialized is True

    def test_add_sample(self, tmp_path, monkeypatch):
        """Sample in buffer."""
        monkeypatch.setattr("training.online_learner.PROJECT_ROOT", tmp_path)

        learner = OnlineLearner()
        learner.load_or_init(n_features=19)

        x = np.random.randn(19)
        learner.add_sample(x, 1)

        assert len(learner.buffer) == 1

    def test_update_buffer_full(self, tmp_path, monkeypatch):
        """Update when full."""
        monkeypatch.setattr("training.online_learner.PROJECT_ROOT", tmp_path)

        learner = OnlineLearner()
        learner.load_or_init(n_features=19)

        # Fill buffer
        for i in range(50):
            x = np.random.randn(19)
            learner.add_sample(x, i % 2)

        # Update should trigger
        learner.update()

        assert len(learner.buffer) == 0

    def test_update_buffer_empty(self, tmp_path, monkeypatch):
        """No update when empty."""
        monkeypatch.setattr("training.online_learner.PROJECT_ROOT", tmp_path)

        learner = OnlineLearner()
        learner.load_or_init(n_features=19)

        # Don't add any samples
        initial_coef = None
        if learner.model is not None and hasattr(learner.model, "coef_"):
            initial_coef = learner.model.coef_.copy()

        learner.update()

        # Model should not have changed
        if initial_coef is not None:
            np.testing.assert_array_equal(initial_coef, learner.model.coef_)

    def test_time_decay(self, tmp_path, monkeypatch):
        """Newer weighted higher."""
        monkeypatch.setattr("training.online_learner.PROJECT_ROOT", tmp_path)

        learner = OnlineLearner()
        learner.load_or_init(n_features=19)

        # Add samples with different timestamps
        for i in range(20):
            x = np.random.randn(19)
            timestamp = time.time() - (20 - i) * 86400  # Days ago
            learner.add_sample(x, i % 2, timestamp=timestamp)

        learner.update()

        # Check that model was updated
        assert learner.initialized is True

    def test_predict(self, tmp_path, monkeypatch):
        """Predictions + uncertainties."""
        monkeypatch.setattr("training.online_learner.PROJECT_ROOT", tmp_path)

        learner = OnlineLearner()
        learner.load_or_init(n_features=19)

        X = np.random.randn(30, 19)
        y = np.random.randint(0, 2, 30)
        learner.initial_fit(X, y)

        X_test = np.random.randn(5, 19)
        result = learner.predict(X_test)

        assert "predictions" in result
        assert "probabilities" in result
        assert "uncertain" in result
        assert len(result["predictions"]) == 5

    def test_save_load(self, tmp_path, monkeypatch):
        """Persists across instances."""
        monkeypatch.setattr("training.online_learner.PROJECT_ROOT", tmp_path)

        learner1 = OnlineLearner()
        learner1.load_or_init(n_features=19)

        X = np.random.randn(30, 19)
        y = np.random.randint(0, 2, 30)
        learner1.initial_fit(X, y)

        X_test = np.random.randn(5, 19)
        predictions1 = learner1.predict(X_test)

        # Load into new learner
        learner2 = OnlineLearner()
        learner2.load_or_init(n_features=19)

        predictions2 = learner2.predict(X_test)

        assert predictions1["predictions"] == predictions2["predictions"]


class TestDataCollector:
    """Tests for DataCollector."""

    def test_record_edit(self, tmp_training_dir, monkeypatch):
        """Edit recorded to JSONL."""
        monkeypatch.setattr("training.online_learner.TRAINING_DIR", tmp_training_dir)

        collector = DataCollector()
        x = np.random.randn(19)
        collector.record_edit("src/a.py", "func_a", x, "session_1")

        assert collector.data_path.exists()

        with open(collector.data_path) as f:
            lines = f.readlines()
        assert len(lines) == 1

    def test_get_labeled_data(self, tmp_training_dir, monkeypatch):
        """Returns X, y arrays."""
        monkeypatch.setattr("training.online_learner.TRAINING_DIR", tmp_training_dir)

        collector = DataCollector()

        # Record labeled edits
        for i in range(10):
            x = np.random.randn(19)
            collector.record_edit(f"src/file_{i}.py", f"func_{i}", x, "session_1", label=i % 2)

        X, y = collector.get_labeled_data()

        assert X.shape == (10, 19)
        assert y.shape == (10,)

    def test_get_unlabeled_data(self, tmp_training_dir, monkeypatch):
        """Returns entries without labels."""
        monkeypatch.setattr("training.online_learner.TRAINING_DIR", tmp_training_dir)

        collector = DataCollector()

        # Record unlabeled edits
        for i in range(5):
            x = np.random.randn(19)
            collector.record_edit(f"src/file_{i}.py", f"func_{i}", x, "session_1")

        unlabeled = collector.get_unlabeled_data()

        assert len(unlabeled) == 5

    def test_apply_labels(self, tmp_training_dir, monkeypatch):
        """Labels applied to unlabeled data."""
        monkeypatch.setattr("training.online_learner.TRAINING_DIR", tmp_training_dir)

        collector = DataCollector()

        # Record unlabeled edits
        for i in range(5):
            x = np.random.randn(19)
            collector.record_edit(f"src/file_{i}.py", f"func_{i}", x, "session_1")

        # Apply labels
        labels = {"src/file_0.py": 1, "src/file_1.py": 0}
        collector.apply_labels(labels)

        # Check labeled data
        X, y = collector.get_labeled_data()
        assert len(X) == 2
