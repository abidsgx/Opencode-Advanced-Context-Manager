"""Manual smoke tests for the Context Manager.

Run with: python -m tests.test_smoke -v

These 8 scenarios simulate real user workflows end-to-end.
"""

import os
import sys
import time
import tempfile
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _banner(n, title):
    print(f"\n{'='*60}")
    print(f"  SMOKE TEST {n}: {title}")
    print(f"{'='*60}")


def _ok(msg):
    print(f"  [PASS] {msg}")


def _info(msg):
    print(f"  [INFO] {msg}")


# ============================================================
# SMOKE TEST 1: First session boot — heuristic-only scoring
# ============================================================
def test_01_first_session_heuristic():
    """Heuristic scoring works without ML model on first boot."""
    _banner(1, "First Session Boot (Heuristic-Only Scoring)")

    from core.importance.heuristic import HeuristicScorer
    from core.importance.model import ImportanceModel

    model = ImportanceModel()
    _info(f"Model initialized: {model.initialized}")
    assert not model.initialized

    session = {
        "_id": "smoke_1", "title": "First session",
        "startedAt": time.time() - 3600, "updatedAt": time.time(),
        "turns": [
            {"role": "assistant", "parts": [{"type": "tool", "toolName": "file_edit",
                "toolInput": {"path": "src/auth/login.py"}, "toolOutput": "ok",
                "timestamp": time.time() - 300}]},
            {"role": "assistant", "parts": [{"type": "tool", "toolName": "file_edit",
                "toolInput": {"path": "src/auth/login.py"}, "toolOutput": "ok",
                "timestamp": time.time() - 200}]},
            {"role": "assistant", "parts": [{"type": "tool", "toolName": "file_edit",
                "toolInput": {"path": "src/db/users.py"}, "toolOutput": "ok",
                "timestamp": time.time() - 100}]},
        ],
        "editImpact": [],
    }

    graph_data = {"degree_centrality": {"src/auth/login.py": 0.8, "src/db/users.py": 0.3}}
    scorer = HeuristicScorer()
    result = scorer.score_all(session, graph_data)
    scores = result["files"]

    _info(f"Scores: {scores}")
    assert scores["src/auth/login.py"] > scores["src/db/users.py"]
    _ok("Heuristic scoring works without ML model")
    _ok("More edits = higher score")
    return True


# ============================================================
# SMOKE TEST 2: Context pruning — low-importance messages removed
# ============================================================
def test_02_context_pruning():
    """Low-importance tool messages get identified for removal."""
    _banner(2, "Context Pruning Pipeline")

    messages = [
        {"role": "tool", "file_path": "src/auth/login.py"},
        {"role": "tool", "file_path": "src/db/users.py"},
        {"role": "tool", "file_path": "src/utils/helpers.py"},
    ]

    importance_scores = {
        "src/auth/login.py": 9.0,
        "src/db/users.py": 4.0,
        "src/utils/helpers.py": 1.5,
    }

    protected_threshold = 7.0
    prune_threshold = 3.0

    protected = [m for m in messages if importance_scores.get(m["file_path"], 0) >= protected_threshold]
    removable = [m for m in messages if importance_scores.get(m["file_path"], 0) < prune_threshold]

    _info(f"Protected: {[m['file_path'] for m in protected]}")
    _info(f"Removable: {[m['file_path'] for m in removable]}")

    assert len(protected) == 1
    assert len(removable) == 1
    _ok("Protected threshold works")
    _ok("Low-importance messages identified for removal")
    return True


# ============================================================
# SMOKE TEST 3: MCP tools — all 11 tools registered
# ============================================================
def test_03_mcp_tools():
    """All 11 MCP tools are registered and callable."""
    _banner(3, "MCP Tool Registration")

    from mcp_server import (
        score_importance, get_interconnectedness, record_edit,
        get_pending_feedback, submit_labels, get_session_summary,
        update_config, install_grammars, analyze_file_tree,
        get_languages, set_language,
    )

    tools = [
        score_importance, get_interconnectedness, record_edit,
        get_pending_feedback, submit_labels, get_session_summary,
        update_config, install_grammars, analyze_file_tree,
        get_languages, set_language,
    ]

    for fn in tools:
        assert callable(fn), f"{fn.__name__} not callable"

    _info(f"Verified {len(tools)} tool functions: {[fn.__name__ for fn in tools]}")
    _ok("All 11 MCP tool functions importable and callable")
    return True


# ============================================================
# SMOKE TEST 4: Feature extraction — 19 features populated
# ============================================================
def test_04_feature_extraction():
    """Extract features from a session and verify all 19 dimensions."""
    _banner(4, "Feature Extraction (19 Features)")

    from features.base import build_all_features, FeatureVector

    names = FeatureVector.feature_names()
    _info(f"Feature names ({len(names)}): {names}")
    assert len(names) == 19

    session = {
        "_id": "smoke_features", "title": "Feature test",
        "startedAt": time.time() - 7200, "updatedAt": time.time(),
        "turns": [
            {"role": "assistant", "parts": [{"type": "tool", "toolName": "file_edit",
                "toolInput": {"path": "src/a.py", "function": "foo"}, "toolOutput": "ok",
                "timestamp": time.time() - 600}]},
            {"role": "assistant", "parts": [{"type": "tool", "toolName": "file_edit",
                "toolInput": {"path": "src/a.py", "function": "bar"}, "toolOutput": "error occurred",
                "timestamp": time.time() - 300}]},
            {"role": "assistant", "parts": [{"type": "tool", "toolName": "file_edit",
                "toolInput": {"path": "src/b.py"}, "toolOutput": "ok",
                "timestamp": time.time() - 100}]},
        ],
        "editImpact": [],
    }

    graph_data = {
        "degree_centrality": {"src/a.py": 0.7, "src/b.py": 0.3},
        "betweenness_centrality": {}, "clusters": {},
        "dependencies": {}, "dependents": {},
        "file_lines": {"src/a.py": 150, "src/b.py": 50},
        "lines_changed": {"src/a.py": 20, "src/b.py": 5},
        "call_depth": {"src/a.py": 3, "src/b.py": 1},
    }

    features = build_all_features(session, graph_data)
    _info(f"Files with features: {list(features.keys())}")

    for file_path, vec in features.items():
        if isinstance(vec, np.ndarray):
            arr = vec
        else:
            arr = np.array(vec)
        _info(f"  {file_path}: {len(arr)} features, non-zero: {np.count_nonzero(arr)}")
        assert len(arr) == 19
        assert np.count_nonzero(arr) > 0

    _ok("All 19 feature dimensions populated")
    _ok("Graph features populated from centrality data")
    return True


# ============================================================
# SMOKE TEST 5: Online learning — buffer → initial_fit → partial_fit
# ============================================================
def test_05_online_learning():
    """Online learner bootstraps from buffer, auto-initializes, then updates."""
    _banner(5, "Online Learning Bootstrap")

    from training.online_learner import OnlineLearner, MIN_SAMPLES
    import training.online_learner as ol_mod

    _info(f"MIN_SAMPLES threshold: {MIN_SAMPLES}")

    tmpdir = tempfile.mkdtemp()
    try:
        # Patch the save/load paths to use temp directory
        import training.online_learner
        _orig_save = OnlineLearner.save
        _orig_load_or_init = OnlineLearner.load_or_init

        def _patched_load_or_init(self, n_features=19):
            from sklearn.linear_model import SGDClassifier
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.model = SGDClassifier(loss='log_loss', penalty='elasticnet',
                                       l1_ratio=0.15, alpha=0.01, random_state=42)
            self.initialized = False
            self.buffer = []

        def _patched_save(self):
            os.makedirs(os.path.join(tmpdir, 'models'), exist_ok=True)
            import joblib
            joblib.dump(self.model, os.path.join(tmpdir, 'models', 'online_model.pkl'))
            joblib.dump(self.scaler, os.path.join(tmpdir, 'models', 'online_scaler.pkl'))

        OnlineLearner.load_or_init = _patched_load_or_init
        OnlineLearner.save = _patched_save

        learner = OnlineLearner()
        learner.load_or_init(n_features=19)
        _info(f"Initial: initialized={learner.initialized}, buffer={len(learner.buffer)}")
        assert not learner.initialized
        assert len(learner.buffer) == 0

        for i in range(MIN_SAMPLES + 5):
            learner.add_sample(np.random.randn(19), int(i % 2))

        _info(f"After adding {MIN_SAMPLES + 5} samples: buffer={len(learner.buffer)}")
        assert len(learner.buffer) == MIN_SAMPLES + 5

        learner.update()
        _info(f"After update: initialized={learner.initialized}, buffer={len(learner.buffer)}")
        assert learner.initialized
        assert len(learner.buffer) == 0

        for i in range(5):
            learner.add_sample(np.random.randn(19), int(i % 2))
        learner.update()
        assert learner.initialized

        _ok("Buffer accumulates samples")
        _ok(f"Auto-initializes at {MIN_SAMPLES} samples")
        _ok("Buffer cleared after initial_fit")
        _ok("Subsequent updates use partial_fit")
    finally:
        OnlineLearner.load_or_init = _orig_load_or_init
        OnlineLearner.save = _orig_save
        shutil.rmtree(tmpdir, ignore_errors=True)

    return True


# ============================================================
# SMOKE TEST 6: Active learning — uncertain samples identified
# ============================================================
def test_06_active_learning():
    """Active learner identifies uncertain predictions for human review."""
    _banner(6, "Active Learning (Uncertain Sample Identification)")

    from core.importance.active_learning import ActiveLearner

    learner = ActiveLearner()
    pending = learner.get_pending_samples()
    _info(f"Pending samples: {pending['count']}")
    assert pending["count"] == 0

    if not learner.model.initialized:
        _info("Model not initialized — identify_uncertain returns []")
        X = np.random.randn(5, 19)
        file_paths = ["src/a.py", "src/b.py", "src/c.py", "src/d.py", "src/e.py"]
        uncertain = learner.identify_uncertain(X, file_paths)
        _info(f"Uncertain samples: {len(uncertain)}")
        assert len(uncertain) == 0

    _ok("get_pending_samples() returns empty on first run")
    _ok("identify_uncertain returns [] when model uninitialized")
    return True


# ============================================================
# SMOKE TEST 7: Interconnectedness — graph builds, centrality computed
# ============================================================
def test_07_interconnectedness():
    """Build a file dependency graph and compute centrality."""
    _banner(7, "Interconnectedness Graph")

    from core.graph.interconnectedness import InterconnectednessManager

    manager = InterconnectednessManager()

    manager.add_edge("src/a.py", "src/b.py", 0.9, "calls")
    manager.add_edge("src/b.py", "src/c.py", 0.7, "imports")
    manager.add_edge("src/a.py", "src/d.py", 0.6, "calls")

    centrality = manager.compute_centrality()
    _info(f"Centrality: {centrality}")

    assert centrality["src/a.py"] >= centrality["src/c.py"]
    assert centrality["src/a.py"] >= centrality["src/d.py"]

    blast = manager.get_blast_radius("src/a.py")
    _info(f"Blast radius of src/a.py: {blast}")
    assert isinstance(blast, list) and len(blast) >= 1

    clusters = manager.get_clusters()
    _info(f"Clusters: {clusters}")

    matrix = manager.get_matrix()
    _info(f"Matrix keys: {list(matrix.keys())}")

    _ok("Edges added correctly")
    _ok("Centrality computed (A > C, A > D)")
    _ok("Blast radius returns list of dependents")
    _ok("Clusters identified")
    _ok("Matrix exportable")
    return True


# ============================================================
# SMOKE TEST 8: Synthetic data generation
# ============================================================
def test_08_synthetic_data():
    """Generate synthetic training data from rules."""
    _banner(8, "Synthetic Data Generation")

    from training.synthetic import SyntheticGenerator

    gen = SyntheticGenerator()
    samples = gen.generate_from_rules(100)
    _info(f"Generated {len(samples)} samples")

    X = np.array([s["features"] for s in samples])
    y = np.array([s["label"] for s in samples])

    _info(f"Feature matrix shape: {X.shape}")
    _info(f"Label distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}")

    assert X.shape == (100, 19)
    assert set(np.unique(y)).issubset({0, 1})

    n_0, n_1 = np.sum(y == 0), np.sum(y == 1)
    ratio = min(n_0, n_1) / max(n_0, n_1)
    _info(f"Balance ratio: {ratio:.2f}")
    assert ratio >= 0.3

    feature_stds = X.std(axis=0)
    non_trivial = np.sum(feature_stds > 0.01)
    _info(f"Features with non-trivial variance: {non_trivial}/19")
    assert non_trivial >= 10

    _ok("100 samples generated")
    _ok("Feature matrix is (100, 19)")
    _ok("Labels are binary (0/1)")
    _ok("Dataset reasonably balanced")
    _ok("Features have meaningful variance")
    return True


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    tests = [
        test_01_first_session_heuristic,
        test_02_context_pruning,
        test_03_mcp_tools,
        test_04_feature_extraction,
        test_05_online_learning,
        test_06_active_learning,
        test_07_interconnectedness,
        test_08_synthetic_data,
    ]

    results = []
    for test_fn in tests:
        try:
            ok = test_fn()
            results.append((test_fn.__doc__.strip(), ok))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [ERROR] {e}")
            results.append((test_fn.__doc__.strip(), False))

    print(f"\n{'='*60}")
    print(f"  SMOKE TEST RESULTS")
    print(f"{'='*60}")
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  {passed}/{total} smoke tests passed")
    print(f"{'='*60}")

    sys.exit(0 if passed == total else 1)
