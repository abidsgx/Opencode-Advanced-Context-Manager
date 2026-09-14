import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from features.base import FeatureVector, build_all_features
from .heuristic import HeuristicScorer
from .model import ImportanceModel, get_model, predict_importance

# --- 1. Config ---

def load_config():
    config_path = Path(__file__).parent.parent.parent.parent / 'data' / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


config = load_config()


# --- 2. Hybrid scorer ---

class ImportanceScorer:
    def __init__(self):
        self.heuristic = HeuristicScorer()
        self.model = get_model()

    def score_session(self, session_id: str, goal: str = "") -> dict:
        """Score all files/functions in a session.

        Combines heuristic scores with ML predictions.
        """
        # Load session data
        session_data = self._load_session(session_id)
        if not session_data:
            return {'files': {}, 'functions': {}}

        # Get goal files/functions from config
        goal_files = self._get_goal_files(goal)
        goal_functions = self._get_goal_functions(goal)

        # Heuristic scores
        graph_data = self._get_graph_data(session_data)
        heuristic_result = self.heuristic.score_all(
            session_data, graph_data, goal_files, goal_functions
        )

        # ML scores (if model is trained)
        ml_result = self._ml_score(session_data, graph_data)

        # Combine: average heuristic and ML, or just heuristic if ML unavailable
        combined_files = {}
        for file_path in set(list(heuristic_result['files'].keys()) + list(ml_result.get('files', {}).keys())):
            h_score = heuristic_result['files'].get(file_path, 5.0)
            m_score = ml_result.get('files', {}).get(file_path, h_score)

            if self.model.initialized:
                combined = 0.5 * h_score + 0.5 * m_score
            else:
                combined = h_score

            combined_files[file_path] = round(float(np.clip(combined, 0.0, 10.0)), 2)

        combined_functions = {}
        for func in set(list(heuristic_result['functions'].keys()) + list(ml_result.get('functions', {}).keys())):
            h_score = heuristic_result['functions'].get(func, 5.0)
            m_score = ml_result.get('functions', {}).get(func, h_score)

            if self.model.initialized:
                combined = 0.5 * h_score + 0.5 * m_score
            else:
                combined = h_score

            combined_functions[func] = round(float(np.clip(combined, 0.0, 10.0)), 2)

        return {
            'files': combined_files,
            'functions': combined_functions,
        }

    def score_file(self, file_path: str, session_data: dict, graph_data: dict) -> float:
        """Score a single file."""
        heuristic_result = self.heuristic.score_all(session_data, graph_data)
        h_score = heuristic_result['files'].get(file_path, 5.0)

        if self.model.initialized:
            features = build_all_features(session_data, graph_data)
            if file_path in features:
                X = features[file_path].reshape(1, -1)
                ml_result = predict_importance(X)
                m_score = ml_result['probabilities'][0][1] * 10.0
                return round(float(0.5 * h_score + 0.5 * m_score), 2)

        return round(h_score, 2)

    def _ml_score(self, session_data: dict, graph_data: dict) -> dict:
        """Get ML-based scores for all files."""
        if not self.model.initialized:
            return {'files': {}, 'functions': {}}

        features = build_all_features(session_data, graph_data)
        if not features:
            return {'files': {}, 'functions': {}}

        file_paths = list(features.keys())
        X = np.array([features[f] for f in file_paths])

        result = predict_importance(X)

        file_scores = {}
        for i, file_path in enumerate(file_paths):
            prob = result['probabilities'][i][1]
            file_scores[file_path] = prob * 10.0

        return {'files': file_scores, 'functions': {}}

    def _load_session(self, session_id: str) -> dict:
        """Load session data from file."""
        # Try plugin sessions first
        sessions_dir = Path(__file__).parent.parent.parent.parent / '.opencode' / 'sessions'
        session_path = sessions_dir / f"{session_id}.json"

        if session_path.exists():
            with open(session_path, 'r') as f:
                return json.load(f)

        # Try data/sessions
        data_sessions = Path(__file__).parent.parent.parent.parent / 'data' / 'sessions'
        session_path = data_sessions / f"{session_id}.json"

        if session_path.exists():
            with open(session_path, 'r') as f:
                return json.load(f)

        return {}

    def _get_goal_files(self, goal: str) -> list:
        """Get priority files for a goal."""
        agent_config = config.get('agent', {})
        for g in agent_config.get('goals', []):
            if g.get('id') == goal or not goal:
                return g.get('priority_files', [])
        return []

    def _get_goal_functions(self, goal: str) -> list:
        """Get priority functions for a goal."""
        agent_config = config.get('agent', {})
        for g in agent_config.get('goals', []):
            if g.get('id') == goal or not goal:
                return g.get('priority_functions', [])
        return []

    def _get_graph_data(self, session_data: dict = None) -> dict:
        """Get graph data for feature extraction.

        Populates:
        - degree_centrality, betweenness_centrality, clusters, dependencies, dependents
          from interconnectedness graph
        - file_lines, lines_changed, call_depth from tree-sitter analysis
        """
        graph_data = {
            'degree_centrality': {},
            'betweenness_centrality': {},
            'clusters': {},
            'dependencies': {},
            'dependents': {},
            'file_lines': {},
            'lines_changed': {},
            'call_depth': {},
        }

        # 1. Graph metrics from interconnectedness
        try:
            from core.graph.analyzer import GraphAnalyzer
            from core.graph.interconnectedness import InterconnectednessManager

            manager = InterconnectednessManager()
            analyzer = GraphAnalyzer(manager)
            all_metrics = analyzer.analyze_all()

            for file_path, metrics in all_metrics.items():
                graph_data['degree_centrality'][file_path] = metrics['degree_centrality']
                graph_data['betweenness_centrality'][file_path] = metrics['betweenness_centrality']
                graph_data['clusters'][file_path] = metrics['cluster_id']
                graph_data['dependencies'][file_path] = metrics['dependencies']
                graph_data['dependents'][file_path] = metrics['dependents']
        except Exception:
            pass

        # 2. Tree-sitter analysis for file_lines, lines_changed, call_depth
        try:
            from core.graph.tree_sitter import count_lines, get_extension_map, get_parser

            ext_map = get_extension_map()

            # Collect file paths from session data
            files_to_analyze = set()
            if session_data:
                for turn in session_data.get('turns', []):
                    for part in turn.get('parts', []):
                        if part.get('type') != 'tool':
                            continue
                        tool_input = part.get('toolInput', {})
                        file_path = tool_input.get('path', tool_input.get('file', ''))
                        if file_path:
                            files_to_analyze.add(file_path)

            # Also include files from graph metrics
            files_to_analyze.update(graph_data['dependencies'].keys())
            files_to_analyze.update(graph_data['dependents'].keys())

            # Try to analyze each file
            for file_path in files_to_analyze:
                ext = '.' + file_path.rsplit('.', 1)[-1] if '.' in file_path else ''
                if ext not in ext_map:
                    continue

                try:
                    from ..graph.tree_sitter import analyze_file
                    analysis = analyze_file(file_path)

                    graph_data['file_lines'][file_path] = analysis['lines']
                    graph_data['call_depth'][file_path] = analysis['call_depth']

                    # Import relationships from tree-sitter
                    if analysis['imports']:
                        graph_data['dependencies'][file_path] = analysis['imports']
                except Exception:
                    # File might not exist or be unreadable
                    graph_data['file_lines'][file_path] = 0
                    graph_data['call_depth'][file_path] = 0

        except ImportError:
            # tree-sitter not available, use estimates
            pass

        # 3. Lines changed from session data (if available)
        if session_data:
            for turn in session_data.get('turns', []):
                for part in turn.get('parts', []):
                    if part.get('type') != 'tool':
                        continue
                    if not part.get('toolName', '').endswith('edit'):
                        continue

                    tool_input = part.get('toolInput', {})
                    file_path = tool_input.get('path', tool_input.get('file', ''))
                    output = part.get('toolOutput', '')

                    if file_path and output:
                        # Count diff-style changes
                        lines_changed = 0
                        for line in output.split('\n'):
                            line = line.strip()
                            if line.startswith('+') or line.startswith('-'):
                                lines_changed += 1
                        if lines_changed == 0:
                            lines_changed = output.count('\n') + 1

                        graph_data['lines_changed'][file_path] = (
                            graph_data['lines_changed'].get(file_path, 0) + lines_changed
                        )

        return graph_data


# --- 3. Test it ---

if __name__ == '__main__':
    scorer = ImportanceScorer()

    session = {
        'turns': [
            {'role': 'assistant', 'parts': [
                {'type': 'tool', 'toolName': 'file_edit',
                 'toolInput': {'path': 'src/auth/login.py', 'function': 'authenticate'},
                 'toolOutput': 'modified', 'timestamp': time.time() - 300}
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

    result = scorer.heuristic.score_all(session, graph)
    print("=== Heuristic Scores ===")
    for f, s in result['files'].items():
        print(f"  {f}: {s:.2f}")

    print("\nDone.")
