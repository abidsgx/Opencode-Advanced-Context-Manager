import time
from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

# --- 1. Config ---

def load_config():
    config_path = Path(__file__).parent.parent.parent.parent / 'data' / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


config = load_config()
weights = config.get('importance', {}).get('heuristic_weights', {
    'edit_frequency': 0.25,
    'dependency_centrality': 0.20,
    'goal_alignment': 0.30,
    'error_association': 0.10,
    'recency': 0.15,
})


# --- 2. Heuristic scorer ---

_default_weights = {
    'edit_frequency': 0.25,
    'dependency_centrality': 0.20,
    'goal_alignment': 0.30,
    'error_association': 0.10,
    'recency': 0.15,
}


class HeuristicScorer:
    def __init__(self, weights: dict = None):
        self.weights = weights or _default_weights

    def score_file(self, file_path: str, metrics: dict, graph_data: dict,
                   goal_files: list = None, error_files: list = None) -> float:
        """Score a file 0-10 based on heuristics."""
        goal_files = goal_files or []
        error_files = error_files or []

        # Edit frequency: log-scaled, more edits = higher score
        edits = metrics.get('edits_per_file', {}).get(file_path, 0)
        freq_score = np.log1p(edits) / np.log1p(max(metrics.get('total_edits', 1), 1))
        freq_score = min(freq_score * 10, 10.0)

        # Dependency centrality: from graph
        centrality = graph_data.get('degree_centrality', {}).get(file_path, 0.0)
        centrality_score = centrality * 10.0

        # Goal alignment: is this file in the goal list?
        goal_score = 10.0 if file_path in goal_files else 2.0

        # Error association: was this file involved in errors?
        error_score = 8.0 if file_path in error_files else 2.0

        # Recency: exponential decay
        last_edit = metrics.get('last_edit_time', {}).get(file_path, 0.0)
        minutes_ago = (time.time() - last_edit) / 60.0 if last_edit > 0 else 9999.0
        recency_score = 10.0 * np.exp(-0.01 * minutes_ago)

        # Weighted combination
        score = (
            self.weights.get('edit_frequency', 0.25) * freq_score +
            self.weights.get('dependency_centrality', 0.20) * centrality_score +
            self.weights.get('goal_alignment', 0.30) * goal_score +
            self.weights.get('error_association', 0.10) * error_score +
            self.weights.get('recency', 0.15) * recency_score
        )

        return float(np.clip(score, 0.0, 10.0))

    def score_function(self, func_name: str, file_path: str, metrics: dict,
                       graph_data: dict, goal_functions: list = None) -> float:
        """Score a function 0-10 based on heuristics."""
        goal_functions = goal_functions or []

        # Check if function is in goal list
        qualified = f"{file_path}::{func_name}"
        if qualified in goal_functions:
            return 10.0

        # Functions modified more times get higher scores
        func_edits = metrics.get('function_edits', {}).get(qualified, 0)
        freq_score = np.log1p(func_edits) / np.log1p(max(metrics.get('total_edits', 1), 1))
        freq_score = min(freq_score * 10, 10.0)

        # Functions with more dependents get higher scores
        dependents = graph_data.get('function_dependents', {}).get(qualified, [])
        dep_score = min(len(dependents) * 2.0, 10.0)

        score = 0.5 * freq_score + 0.5 * dep_score
        return float(np.clip(score, 0.0, 10.0))

    def score_all(self, session_data: dict, graph_data: dict,
                  goal_files: list = None, goal_functions: list = None,
                  error_files: list = None) -> dict:
        """Score all files and functions in a session."""
        metrics = self._compute_metrics(session_data)

        file_scores = {}
        for file_path in metrics.get('edits_per_file', {}).keys():
            file_scores[file_path] = self.score_file(
                file_path, metrics, graph_data, goal_files, error_files
            )

        func_scores = {}
        for qualified in metrics.get('function_edits', {}).keys():
            file_path, func_name = qualified.rsplit('::', 1)
            func_scores[qualified] = self.score_function(
                func_name, file_path, metrics, graph_data, goal_functions
            )

        return {
            'files': file_scores,
            'functions': func_scores,
        }

    def _compute_metrics(self, session_data: dict) -> dict:
        """Compute basic metrics from session data."""
        metrics = {
            'total_edits': 0,
            'edits_per_file': {},
            'function_edits': {},
            'first_edit_time': {},
            'last_edit_time': {},
        }

        for turn in session_data.get('turns', []):
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
                    ts = time.time()

                metrics['edits_per_file'][file_path] = metrics['edits_per_file'].get(file_path, 0) + 1
                metrics['total_edits'] += 1

                if file_path not in metrics['first_edit_time']:
                    metrics['first_edit_time'][file_path] = ts
                metrics['last_edit_time'][file_path] = ts

                func = tool_input.get('function', '')
                if func:
                    qualified = f"{file_path}::{func}"
                    metrics['function_edits'][qualified] = metrics['function_edits'].get(qualified, 0) + 1

        return metrics


# --- 3. Test it ---

if __name__ == '__main__':
    session = {
        'turns': [
            {'role': 'assistant', 'parts': [
                {'type': 'tool', 'toolName': 'file_edit',
                 'toolInput': {'path': 'src/auth/login.py', 'function': 'authenticate'},
                 'toolOutput': 'modified', 'timestamp': time.time() - 300}
            ]},
            {'role': 'assistant', 'parts': [
                {'type': 'tool', 'toolName': 'file_edit',
                 'toolInput': {'path': 'src/auth/login.py', 'function': 'validate_token'},
                 'toolOutput': 'modified', 'timestamp': time.time() - 100}
            ]},
            {'role': 'assistant', 'parts': [
                {'type': 'tool', 'toolName': 'file_edit',
                 'toolInput': {'path': 'src/db/users.py'},
                 'toolOutput': 'modified', 'timestamp': time.time() - 50}
            ]},
        ]
    }

    graph = {
        'degree_centrality': {'src/auth/login.py': 0.8, 'src/db/users.py': 0.4},
        'function_dependents': {},
    }

    scorer = HeuristicScorer()
    result = scorer.score_all(session, graph, goal_files=['src/auth/login.py'])

    print("=== Heuristic Scores ===")
    for f, s in result['files'].items():
        print(f"  {f}: {s:.2f}")
    for f, s in result['functions'].items():
        print(f"  {f}: {s:.2f}")

    print("\nDone.")
