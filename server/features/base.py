import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

# --- 1. Feature vector dataclass ---

@dataclass
class FeatureVector:
    # Temporal features
    time_spent_on_file: float = 0.0
    time_spent_ratio: float = 0.0
    edit_frequency: float = 0.0
    recency_score: float = 0.0

    # Structural features
    lines_changed: int = 0
    lines_changed_ratio: float = 0.0
    functions_modified: int = 0
    call_depth: int = 0

    # Graph features
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    cluster_id: int = 0
    dependency_count: int = 0
    dependents_count: int = 0

    # Edit pattern features
    revert_count: int = 0
    iteration_count: int = 0
    cross_file_edits: int = 0

    # Composite features
    dependency_x_time: float = 0.0
    centrality_x_frequency: float = 0.0
    cluster_isolation: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.time_spent_on_file,
            self.time_spent_ratio,
            self.edit_frequency,
            self.recency_score,
            self.lines_changed,
            self.lines_changed_ratio,
            self.functions_modified,
            self.call_depth,
            self.degree_centrality,
            self.betweenness_centrality,
            self.cluster_id,
            self.dependency_count,
            self.dependents_count,
            self.revert_count,
            self.iteration_count,
            self.cross_file_edits,
            self.dependency_x_time,
            self.centrality_x_frequency,
            self.cluster_isolation,
        ])

    @staticmethod
    def feature_names() -> list:
        return [
            'time_spent_on_file', 'time_spent_ratio', 'edit_frequency', 'recency_score',
            'lines_changed', 'lines_changed_ratio', 'functions_modified', 'call_depth',
            'degree_centrality', 'betweenness_centrality', 'cluster_id',
            'dependency_count', 'dependents_count',
            'revert_count', 'iteration_count', 'cross_file_edits',
            'dependency_x_time', 'centrality_x_frequency', 'cluster_isolation',
        ]


@dataclass
class EditRecord:
    file_path: str
    function_name: str
    timestamp: float
    lines_changed: int
    is_revert: bool
    is_iteration: bool
    co_edited_files: List[str] = field(default_factory=list)


@dataclass
class SessionMetrics:
    total_edits: int = 0
    total_time: float = 0.0
    edits_per_file: Dict[str, int] = field(default_factory=dict)
    time_per_file: Dict[str, float] = field(default_factory=dict)
    functions_per_file: Dict[str, set] = field(default_factory=dict)
    first_edit_time: Dict[str, float] = field(default_factory=dict)
    last_edit_time: Dict[str, float] = field(default_factory=dict)
    revert_counts: Dict[str, int] = field(default_factory=dict)
    iteration_counts: Dict[str, int] = field(default_factory=dict)
    cross_file_pairs: Dict[str, int] = field(default_factory=dict)


# --- 2. Feature extractor base class ---

class FeatureExtractor:
    def extract(self, file_path: str, metrics: SessionMetrics, graph_data: dict) -> FeatureVector:
        raise NotImplementedError


# --- 3. Temporal feature extractor ---

class TemporalExtractor(FeatureExtractor):
    def extract(self, file_path: str, metrics: SessionMetrics, graph_data: dict) -> FeatureVector:
        edits = metrics.edits_per_file.get(file_path, 0)
        time_on_file = metrics.time_per_file.get(file_path, 0.0)
        total_time = max(metrics.total_time, 1.0)

        time_ratio = time_on_file / total_time
        freq = edits / total_time if total_time > 0 else 0.0

        # Recency: exponential decay from last edit
        last_time = metrics.last_edit_time.get(file_path, 0.0)
        minutes_ago = (time.time() - last_time) / 60.0 if last_time > 0 else 9999.0
        recency = np.exp(-0.01 * minutes_ago)

        fv = FeatureVector()
        fv.time_spent_on_file = time_on_file
        fv.time_spent_ratio = time_ratio
        fv.edit_frequency = freq
        fv.recency_score = recency
        return fv


# --- 4. Structural feature extractor ---

class StructuralExtractor(FeatureExtractor):
    def extract(self, file_path: str, metrics: SessionMetrics, graph_data: dict) -> FeatureVector:
        functions = metrics.functions_per_file.get(file_path, set())
        lines = graph_data.get('file_lines', {}).get(file_path, 100)
        lines_changed = graph_data.get('lines_changed', {}).get(file_path, 0)
        call_depth = graph_data.get('call_depth', {}).get(file_path, 0)

        ratio = lines_changed / max(lines, 1)

        fv = FeatureVector()
        fv.lines_changed = lines_changed
        fv.lines_changed_ratio = ratio
        fv.functions_modified = len(functions)
        fv.call_depth = call_depth
        return fv


# --- 5. Graph feature extractor ---

class GraphExtractor(FeatureExtractor):
    def extract(self, file_path: str, metrics: SessionMetrics, graph_data: dict) -> FeatureVector:
        centrality = graph_data.get('degree_centrality', {})
        betweenness = graph_data.get('betweenness_centrality', {})
        clusters = graph_data.get('clusters', {})
        dependencies = graph_data.get('dependencies', {})
        dependents = graph_data.get('dependents', {})

        total_files = max(len(centrality), 1)
        cluster_members = sum(1 for c in clusters.values() if c == clusters.get(file_path, -1))
        isolation = 1.0 - (cluster_members / total_files) if total_files > 0 else 1.0

        fv = FeatureVector()
        fv.degree_centrality = centrality.get(file_path, 0.0)
        fv.betweenness_centrality = betweenness.get(file_path, 0.0)
        fv.cluster_id = clusters.get(file_path, 0)
        fv.dependency_count = len(dependencies.get(file_path, []))
        fv.dependents_count = len(dependents.get(file_path, []))
        fv.cluster_isolation = isolation
        return fv


# --- 6. Edit pattern feature extractor ---

class EditPatternExtractor(FeatureExtractor):
    def extract(self, file_path: str, metrics: SessionMetrics, graph_data: dict) -> FeatureVector:
        reverts = metrics.revert_counts.get(file_path, 0)
        iterations = metrics.iteration_counts.get(file_path, 0)

        # Count cross-file edits: how many other files were edited in the same sessions
        cross = 0
        for pair, count in metrics.cross_file_pairs.items():
            if file_path in pair:
                cross += count

        fv = FeatureVector()
        fv.revert_count = reverts
        fv.iteration_count = iterations
        fv.cross_file_edits = cross
        return fv


# --- 7. Composite feature extractor ---

class CompositeExtractor(FeatureExtractor):
    def __init__(self, temporal: TemporalExtractor, structural: StructuralExtractor,
                 graph: GraphExtractor, edit_pattern: EditPatternExtractor):
        self.temporal = temporal
        self.structural = structural
        self.graph = graph
        self.edit_pattern = edit_pattern

    def extract(self, file_path: str, metrics: SessionMetrics, graph_data: dict) -> FeatureVector:
        t = self.temporal.extract(file_path, metrics, graph_data)
        s = self.structural.extract(file_path, metrics, graph_data)
        g = self.graph.extract(file_path, metrics, graph_data)
        e = self.edit_pattern.extract(file_path, metrics, graph_data)

        fv = FeatureVector()
        # Copy base features
        for name in FeatureVector.feature_names():
            val = getattr(t, name, 0.0)
            if val == 0.0:
                val = getattr(s, name, 0.0)
            if val == 0.0:
                val = getattr(g, name, 0.0)
            if val == 0.0:
                val = getattr(e, name, 0.0)
            setattr(fv, name, val)

        # Compute composites
        fv.dependency_x_time = fv.dependency_count * fv.time_spent_ratio
        fv.centrality_x_frequency = fv.degree_centrality * fv.edit_frequency
        return fv


# --- 8. Build all features for a file ---

def build_features(file_path: str, metrics: SessionMetrics, graph_data: dict) -> np.ndarray:
    temporal = TemporalExtractor()
    structural = StructuralExtractor()
    graph_ext = GraphExtractor()
    edit_pattern = EditPatternExtractor()
    composite = CompositeExtractor(temporal, structural, graph_ext, edit_pattern)

    parts = [
        temporal.extract(file_path, metrics, graph_data),
        structural.extract(file_path, metrics, graph_data),
        graph_ext.extract(file_path, metrics, graph_data),
        edit_pattern.extract(file_path, metrics, graph_data),
        composite.extract(file_path, metrics, graph_data),
    ]

    # Merge: composite overwrites base features, composites are additive
    final = FeatureVector()
    for part in parts:
        for name in FeatureVector.feature_names():
            current = getattr(final, name)
            new_val = getattr(part, name)
            if name in ('dependency_x_time', 'centrality_x_frequency', 'cluster_isolation'):
                setattr(final, name, new_val)
            elif current == 0.0 and new_val != 0.0:
                setattr(final, name, new_val)

    return final.to_array()


def build_all_features(session_data: dict, graph_data: dict) -> dict:
    """Build features for all files in a session.

    Returns dict mapping file_path -> numpy feature array.
    """
    metrics = compute_session_metrics(session_data)

    features = {}
    for file_path in metrics.edits_per_file.keys():
        features[file_path] = build_features(file_path, metrics, graph_data)

    return features


def compute_session_metrics(session_data: dict) -> SessionMetrics:
    """Compute session-level metrics from raw session data.

    Populates all fields including:
    - edits_per_file: count of edits per file
    - time_per_file: total time spent editing each file
    - functions_per_file: set of functions modified per file
    - first_edit_time / last_edit_time: timestamps
    - revert_counts: times a file was reverted then re-edited
    - iteration_counts: times a file was re-edited after editing another file
    - cross_file_pairs: co-edit frequency between file pairs
    """
    metrics = SessionMetrics()
    turns = session_data.get('turns', [])

    # Track edit history for revert/iteration detection
    # Each entry: (file_path, function_name, timestamp)
    edit_history = []

    # Track which files are edited in sequence for cross-file detection
    recent_edits = []  # Last N file paths in order

    for turn in turns:
        for part in turn.get('parts', []):
            if part.get('type') != 'tool':
                continue
            if not part.get('toolName', '').endswith('edit'):
                continue

            tool_input = part.get('toolInput', {})
            file_path = tool_input.get('path', tool_input.get('file', ''))
            if not file_path:
                continue

            ts = part.get('timestamp', time.time())
            if isinstance(ts, str):
                try:
                    ts = float(ts)
                except ValueError:
                    ts = time.time()

            func = tool_input.get('function', '')
            output = part.get('toolOutput', '')

            # Count lines changed from output
            lines_changed = 0
            if output:
                # Look for diff-style indicators: +/- lines, or line ranges
                for line in output.split('\n'):
                    line = line.strip()
                    if line.startswith('+') or line.startswith('-'):
                        lines_changed += 1
                    elif line.startswith('@@'):
                        # Unified diff hunk header like @@ -10,5 +10,7 @@
                        pass
                # Fallback: if no diff markers, count total lines
                if lines_changed == 0:
                    lines_changed = output.count('\n') + 1

            # --- Update basic metrics ---
            metrics.edits_per_file[file_path] = metrics.edits_per_file.get(file_path, 0) + 1
            metrics.total_edits += 1

            if file_path not in metrics.first_edit_time:
                metrics.first_edit_time[file_path] = ts
            metrics.last_edit_time[file_path] = ts

            # Track function
            if func:
                if file_path not in metrics.functions_per_file:
                    metrics.functions_per_file[file_path] = set()
                metrics.functions_per_file[file_path].add(func)

            # --- Detect reverts ---
            # A revert = same file edited, then a different file, then same file again
            # within a short time window (or tool output contains "revert"/"undo"/"rollback")
            output_lower = output.lower() if output else ''
            if any(word in output_lower for word in ('revert', 'undo', 'rollback', 'reset')):
                metrics.revert_counts[file_path] = metrics.revert_counts.get(file_path, 0) + 1
            else:
                # Check pattern: file A -> file B -> file A (re-edit after touching another file)
                if len(edit_history) >= 2:
                    prev_file = edit_history[-1][0]
                    prev_prev_file = edit_history[-2][0]
                    if file_path == prev_prev_file and prev_file != file_path:
                        # Same file re-edited after touching another file = possible revert/rework
                        metrics.revert_counts[file_path] = metrics.revert_counts.get(file_path, 0) + 1

            # --- Detect iterations ---
            # An iteration = file re-edited multiple times in the session (more than 2 edits)
            edit_count = metrics.edits_per_file[file_path]
            if edit_count > 2:
                metrics.iteration_counts[file_path] = edit_count - 2  # iterations beyond first 2 edits

            # --- Track cross-file edits ---
            recent_edits.append(file_path)
            if len(recent_edits) > 10:
                recent_edits = recent_edits[-10:]

            # Count co-edits: pairs of files edited close together
            for other_file in set(recent_edits):
                if other_file == file_path:
                    continue
                pair_key = tuple(sorted([file_path, other_file]))
                pair_str = f"{pair_key[0]}|{pair_key[1]}"
                metrics.cross_file_pairs[pair_str] = metrics.cross_file_pairs.get(pair_str, 0) + 1

            # Record in history
            edit_history.append((file_path, func, ts))

    # Compute time spent per file and total session time
    if edit_history:
        timestamps = [e[2] for e in edit_history]
        min_time = min(timestamps)
        max_time = max(timestamps)
        metrics.total_time = max((max_time - min_time) / 60.0, 1.0)

        # Time per file = sum of time between consecutive edits to that file
        file_times = {}
        file_edit_times = {}
        for file_path, _, ts in edit_history:
            if file_path not in file_edit_times:
                file_edit_times[file_path] = []
            file_edit_times[file_path].append(ts)

        for file_path, times in file_edit_times.items():
            if len(times) >= 2:
                file_times[file_path] = (max(times) - min(times)) / 60.0
            else:
                file_times[file_path] = 0.0

        metrics.time_per_file = file_times
    else:
        metrics.total_time = 1.0

    return metrics


# --- 9. Test it ---

if __name__ == '__main__':
    # Dummy session data for testing
    session = {
        'turns': [
            {'role': 'assistant', 'parts': [
                {'type': 'tool', 'toolName': 'file_edit',
                 'toolInput': {'path': 'src/auth/login.py', 'function': 'authenticate'},
                 'toolOutput': 'line1\nline2\nline3', 'timestamp': time.time() - 300}
            ]},
            {'role': 'assistant', 'parts': [
                {'type': 'tool', 'toolName': 'file_edit',
                 'toolInput': {'path': 'src/auth/login.py', 'function': 'validate_token'},
                 'toolOutput': 'line1\nline2', 'timestamp': time.time() - 100}
            ]},
            {'role': 'assistant', 'parts': [
                {'type': 'tool', 'toolName': 'file_edit',
                 'toolInput': {'path': 'src/db/users.py'},
                 'toolOutput': 'line1', 'timestamp': time.time() - 50}
            ]},
        ]
    }

    graph = {
        'file_lines': {'src/auth/login.py': 200, 'src/db/users.py': 150},
        'lines_changed': {'src/auth/login.py': 5, 'src/db/users.py': 2},
        'call_depth': {'src/auth/login.py': 2, 'src/db/users.py': 1},
        'degree_centrality': {'src/auth/login.py': 0.8, 'src/db/users.py': 0.4},
        'betweenness_centrality': {'src/auth/login.py': 0.6, 'src/db/users.py': 0.2},
        'clusters': {'src/auth/login.py': 0, 'src/db/users.py': 1},
        'dependencies': {'src/auth/login.py': ['src/db/users.py'], 'src/db/users.py': []},
        'dependents': {'src/auth/login.py': [], 'src/db/users.py': ['src/auth/login.py']},
    }

    features = build_all_features(session, graph)

    print("=== Feature Vectors ===")
    for file_path, vec in features.items():
        print(f"\n{file_path}:")
        names = FeatureVector.feature_names()
        for i, name in enumerate(names):
            print(f"  {name}: {vec[i]:.4f}")

    print("\nDone.")
