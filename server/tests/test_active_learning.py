"""Tests for ActiveLearner."""

import json
import time
import numpy as np
import pytest

from core.importance.active_learning import ActiveLearner


class TestActiveLearner:
    """Tests for ActiveLearner."""

    def test_identify_uncertain(self, tmp_models_dir, monkeypatch):
        """Finds margin-sampled uncertain points."""
        monkeypatch.setattr(
            "core.importance.active_learning.PENDING_PATH",
            tmp_models_dir / "pending.jsonl",
        )

        # Initialize model
        from core.importance.model import ImportanceModel

        model = ImportanceModel()
        X = np.random.randn(50, 19)
        model.initialize(X)

        learner = ActiveLearner()
        learner.model = model

        X_test = np.random.randn(20, 19)
        files = [f"src/file_{i}.py" for i in range(20)]

        uncertain = learner.identify_uncertain(X_test, files)

        assert len(uncertain) > 0
        assert len(uncertain) <= 10
        assert all("sample_id" in s for s in uncertain)
        assert all("file_path" in s for s in uncertain)
        assert all("probability" in s for s in uncertain)
        assert all("margin" in s for s in uncertain)

    def test_identify_uncertain_no_model(self, tmp_models_dir, monkeypatch):
        """Empty when model not initialized."""
        monkeypatch.setattr(
            "core.importance.active_learning.PENDING_PATH",
            tmp_models_dir / "pending.jsonl",
        )

        learner = ActiveLearner()
        learner.model.initialized = False

        X_test = np.random.randn(10, 19)
        files = [f"src/file_{i}.py" for i in range(10)]

        uncertain = learner.identify_uncertain(X_test, files)

        assert len(uncertain) == 0

    def test_get_pending_empty(self, tmp_models_dir, monkeypatch):
        """No pending file -> empty list."""
        monkeypatch.setattr(
            "core.importance.active_learning.PENDING_PATH",
            tmp_models_dir / "pending.jsonl",
        )

        learner = ActiveLearner()
        result = learner.get_pending_samples()

        assert result["samples"] == []
        assert result["count"] == 0

    def test_get_pending_with_data(self, tmp_models_dir, monkeypatch):
        """Pending file exists -> returns samples."""
        pending_path = tmp_models_dir / "pending.jsonl"
        monkeypatch.setattr("core.importance.active_learning.PENDING_PATH", pending_path)

        # Write pending samples
        samples = [
            {"sample_id": "s1", "file_path": "src/a.py", "features": [0.1] * 19, "margin": 0.1},
            {"sample_id": "s2", "file_path": "src/b.py", "features": [0.2] * 19, "margin": 0.2},
        ]
        with open(pending_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        learner = ActiveLearner()
        result = learner.get_pending_samples()

        assert result["count"] == 2
        assert len(result["samples"]) == 2

    def test_submit_labels(self, tmp_models_dir, monkeypatch):
        """Labels saved to labels.jsonl."""
        pending_path = tmp_models_dir / "pending.jsonl"
        labels_path = tmp_models_dir / "labels.jsonl"
        monkeypatch.setattr("core.importance.active_learning.PENDING_PATH", pending_path)
        monkeypatch.setattr("core.importance.active_learning.LABELS_PATH", labels_path)

        # Write pending samples
        samples = [
            {"sample_id": "s1", "file_path": "src/a.py", "features": [0.1] * 19, "margin": 0.1},
        ]
        with open(pending_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        learner = ActiveLearner()
        labels = [{"sample_id": "s1", "score": 8.5}]
        result = learner.submit_labels(labels)

        assert result["status"] == "ok"
        assert result["labeled"] == 1

        # Check labels file
        with open(labels_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        label = json.loads(lines[0])
        assert label["sample_id"] == "s1"
        assert label["score"] == 8.5

    def test_cleanup_pending(self, tmp_models_dir, monkeypatch):
        """Labeled removed from pending."""
        pending_path = tmp_models_dir / "pending.jsonl"
        labels_path = tmp_models_dir / "labels.jsonl"
        monkeypatch.setattr("core.importance.active_learning.PENDING_PATH", pending_path)
        monkeypatch.setattr("core.importance.active_learning.LABELS_PATH", labels_path)

        # Write pending samples
        samples = [
            {"sample_id": "s1", "file_path": "src/a.py", "features": [0.1] * 19, "margin": 0.1},
            {"sample_id": "s2", "file_path": "src/b.py", "features": [0.2] * 19, "margin": 0.2},
        ]
        with open(pending_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        learner = ActiveLearner()
        labels = [{"sample_id": "s1", "score": 8.5}]
        learner.submit_labels(labels)

        # Check pending file
        with open(pending_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        remaining = json.loads(lines[0])
        assert remaining["sample_id"] == "s2"

    def test_margin_sampling_order(self, tmp_models_dir, monkeypatch):
        """Most uncertain selected first."""
        monkeypatch.setattr(
            "core.importance.active_learning.PENDING_PATH",
            tmp_models_dir / "pending.jsonl",
        )

        from core.importance.model import ImportanceModel

        model = ImportanceModel()
        X = np.random.randn(50, 19)
        model.initialize(X)

        learner = ActiveLearner()
        learner.model = model

        X_test = np.random.randn(20, 19)
        files = [f"src/file_{i}.py" for i in range(20)]

        uncertain = learner.identify_uncertain(X_test, files)

        # Check that margins are sorted (ascending)
        margins = [s["margin"] for s in uncertain]
        assert margins == sorted(margins)
