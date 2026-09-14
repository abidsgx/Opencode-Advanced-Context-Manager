"""Shared test fixtures for ctx-manager-server tests."""

import json
import time
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml


# --- 1. Session data fixtures ---


@pytest.fixture
def sample_session_data():
    """Valid session with 3 edited files, revert pattern, co-edits."""
    return {
        "_id": "test_session_001",
        "title": "Test session",
        "startedAt": time.time() - 3600,
        "updatedAt": time.time(),
        "turns": [
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool",
                        "toolName": "file_edit",
                        "toolInput": {"path": "src/auth/login.py", "function": "authenticate"},
                        "toolOutput": "line1\nline2\nline3",
                        "timestamp": time.time() - 300,
                    }
                ],
            },
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool",
                        "toolName": "file_edit",
                        "toolInput": {"path": "src/auth/login.py", "function": "validate_token"},
                        "toolOutput": "line1\nline2",
                        "timestamp": time.time() - 200,
                    }
                ],
            },
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool",
                        "toolName": "file_edit",
                        "toolInput": {"path": "src/db/users.py"},
                        "toolOutput": "line1",
                        "timestamp": time.time() - 100,
                    }
                ],
            },
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool",
                        "toolName": "file_edit",
                        "toolInput": {"path": "src/auth/login.py", "function": "authenticate"},
                        "toolOutput": "reverted previous change",
                        "timestamp": time.time() - 50,
                    }
                ],
            },
        ],
        "editImpact": [],
        "importanceScores": None,
        "interconnectedness": {},
    }


@pytest.fixture
def sample_session_data_no_edits():
    """Session with no tool edits."""
    return {
        "_id": "test_session_no_edits",
        "title": "No edits session",
        "startedAt": time.time() - 3600,
        "updatedAt": time.time(),
        "turns": [
            {"role": "user", "parts": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "parts": [{"type": "text", "text": "Hi there"}]},
        ],
        "editImpact": [],
    }


@pytest.fixture
def session_data_multiple_files():
    """Session with edits to 5 different files."""
    files = [
        "src/auth/login.py",
        "src/auth/middleware.py",
        "src/db/users.py",
        "src/utils/helpers.py",
        "src/api/routes.py",
    ]
    turns = []
    for i, file_path in enumerate(files):
        turns.append(
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool",
                        "toolName": "file_edit",
                        "toolInput": {"path": file_path, "function": f"func_{i}"},
                        "toolOutput": f"modified {file_path}",
                        "timestamp": time.time() - (len(files) - i) * 60,
                    }
                ],
            }
        )
    return {
        "_id": "test_session_multi",
        "title": "Multi-file session",
        "startedAt": time.time() - 3600,
        "updatedAt": time.time(),
        "turns": turns,
        "editImpact": [],
    }


# --- 2. Graph data fixtures ---


@pytest.fixture
def sample_graph_data():
    """Graph data with centrality, betweenness, clusters, dependencies."""
    return {
        "degree_centrality": {
            "src/auth/login.py": 0.8,
            "src/db/users.py": 0.4,
            "src/auth/middleware.py": 0.6,
        },
        "betweenness_centrality": {
            "src/auth/login.py": 0.6,
            "src/db/users.py": 0.2,
            "src/auth/middleware.py": 0.4,
        },
        "clusters": {"src/auth/login.py": 0, "src/db/users.py": 1, "src/auth/middleware.py": 0},
        "dependencies": {
            "src/auth/login.py": ["src/db/users.py"],
            "src/db/users.py": [],
            "src/auth/middleware.py": ["src/auth/login.py"],
        },
        "dependents": {
            "src/auth/login.py": ["src/auth/middleware.py"],
            "src/db/users.py": ["src/auth/login.py"],
            "src/auth/middleware.py": [],
        },
        "file_lines": {"src/auth/login.py": 200, "src/db/users.py": 150, "src/auth/middleware.py": 100},
        "lines_changed": {"src/auth/login.py": 10, "src/db/users.py": 5, "src/auth/middleware.py": 3},
        "call_depth": {"src/auth/login.py": 3, "src/db/users.py": 1, "src/auth/middleware.py": 2},
    }


@pytest.fixture
def sample_graph_data_empty():
    """Empty graph data."""
    return {
        "degree_centrality": {},
        "betweenness_centrality": {},
        "clusters": {},
        "dependencies": {},
        "dependents": {},
        "file_lines": {},
        "lines_changed": {},
        "call_depth": {},
    }


# --- 3. Feature vector fixtures ---


@pytest.fixture
def sample_feature_vector():
    """19-dim numpy array with realistic values."""
    return np.array(
        [
            5.0,  # time_spent_on_file
            0.3,  # time_spent_ratio
            0.1,  # edit_frequency
            0.9,  # recency_score
            15,  # lines_changed
            0.075,  # lines_changed_ratio
            2,  # functions_modified
            3,  # call_depth
            0.8,  # degree_centrality
            0.6,  # betweenness_centrality
            0,  # cluster_id
            1,  # dependency_count
            2,  # dependents_count
            1,  # revert_count
            2,  # iteration_count
            3,  # cross_file_edits
            0.3,  # dependency_x_time (1 * 0.3)
            0.08,  # centrality_x_frequency (0.8 * 0.1)
            0.5,  # cluster_isolation
        ]
    )


@pytest.fixture
def sample_feature_matrix():
    """Multiple feature vectors as a matrix."""
    np.random.seed(42)
    return np.random.randn(10, 19)


# --- 4. Config fixtures ---


@pytest.fixture
def sample_config():
    """Importance + pruning config dict."""
    return {
        "version": "1.0",
        "agent": {"goals": []},
        "interconnectedness": {
            "files": {},
            "functions": {},
            "auto_detect": {"enabled": True, "methods": ["imports", "calls", "co_edits"]},
        },
        "languages": {
            "python": {
                "enabled": True,
                "package": "tree-sitter-python",
                "extensions": [".py"],
            },
            "javascript": {
                "enabled": True,
                "package": "tree-sitter-javascript",
                "extensions": [".js", ".jsx", ".mjs"],
            },
            "typescript": {
                "enabled": True,
                "package": "tree-sitter-typescript",
                "extensions": [".ts", ".tsx"],
            },
        },
        "importance": {
            "ml": {
                "enabled": True,
                "model_path": "data/models/importance_model.pkl",
                "scaler_path": "data/models/feature_scaler.pkl",
                "confidence_threshold": 0.6,
                "online_learning": {
                    "enabled": True,
                    "decay_rate": 0.023,
                    "min_samples_for_training": 10,
                },
            },
            "heuristic_weights": {
                "edit_frequency": 0.25,
                "dependency_centrality": 0.20,
                "goal_alignment": 0.30,
                "error_association": 0.10,
                "recency": 0.15,
            },
        },
        "pruning": {
            "enabled": True,
            "importance_threshold": 3.0,
            "protected_importance": 7.0,
            "blast_radius_expansion": True,
            "max_compression_ratio": 0.3,
        },
        "session": {
            "extended_recording": True,
            "capture_edit_metadata": True,
        },
    }


@pytest.fixture
def sample_config_yaml(tmp_path):
    """Write sample config to YAML and return path."""
    config = {
        "version": "1.0",
        "importance": {
            "heuristic_weights": {
                "edit_frequency": 0.25,
                "dependency_centrality": 0.20,
                "goal_alignment": 0.30,
                "error_association": 0.10,
                "recency": 0.15,
            }
        },
        "pruning": {"importance_threshold": 3.0, "protected_importance": 7.0},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


# --- 5. Temporary directory fixtures ---


@pytest.fixture
def tmp_models_dir(tmp_path):
    """Temporary models directory for save/load tests."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir


@pytest.fixture
def tmp_sessions_dir(tmp_path):
    """Temporary sessions directory."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    return sessions_dir


@pytest.fixture
def tmp_training_dir(tmp_path):
    """Temporary training directory."""
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    return training_dir


# --- 6. Label fixtures ---


@pytest.fixture
def sample_labels():
    """Sample labels for active learning."""
    return [
        {"sample_id": "sample_001", "score": 8.5},
        {"sample_id": "sample_002", "score": 2.0},
        {"sample_id": "sample_003", "score": 6.0},
    ]


@pytest.fixture
def sample_pending_samples():
    """Sample pending samples for active learning."""
    return [
        {
            "sample_id": "sample_001",
            "file_path": "src/auth/login.py",
            "features": np.random.randn(19).tolist(),
            "probability": [0.45, 0.55],
            "margin": 0.1,
            "timestamp": time.time(),
        },
        {
            "sample_id": "sample_002",
            "file_path": "src/db/users.py",
            "features": np.random.randn(19).tolist(),
            "probability": [0.52, 0.48],
            "margin": 0.04,
            "timestamp": time.time(),
        },
    ]
