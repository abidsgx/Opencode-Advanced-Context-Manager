import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# --- 1. Data types ---

@dataclass
class Edge:
    source: str
    target: str
    weight: float
    relation: str  # "calls", "imports", "inherits", "co_edits", "tests"
    context: str = ""
    source_type: str = "file"  # "file" or "function"
    target_type: str = "file"

    def to_dict(self):
        return {
            'weight': self.weight,
            'relation': self.relation,
            'context': self.context,
        }


@dataclass
class GraphNode:
    name: str
    node_type: str = "file"  # "file" or "function"
    edges_out: List[Edge] = field(default_factory=list)
    edges_in: List[Edge] = field(default_factory=list)


# --- 2. Config ---

def load_config():
    config_path = Path(__file__).parent.parent.parent.parent / 'data' / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def save_config(config):
    config_path = Path(__file__).parent.parent.parent.parent / 'data' / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


# --- 3. Interconnectedness manager ---

class InterconnectednessManager:
    def __init__(self):
        self.config = load_config()
        self.matrix = self.config.get('interconnectedness', {'files': {}, 'functions': {}})

    def get_matrix(self, session_id: str = "", goal: str = "") -> dict:
        """Return the full interconnectedness matrix."""
        return self.matrix

    def add_edge(self, source: str, target: str, weight: float, relation: str,
                 context: str = "", level: str = "files"):
        """Add an edge to the matrix."""
        if level not in self.matrix:
            self.matrix[level] = {}

        if source not in self.matrix[level]:
            self.matrix[level][source] = {}

        self.matrix[level][source][target] = {
            'weight': weight,
            'relation': relation,
            'context': context,
        }

        self._save()

    def remove_edge(self, source: str, target: str, level: str = "files"):
        """Remove an edge from the matrix."""
        if level in self.matrix and source in self.matrix[level]:
            if target in self.matrix[level][source]:
                del self.matrix[level][source][target]
                self._save()

    def get_connections(self, file_path: str, level: str = "files") -> dict:
        """Get all connections for a file/function."""
        return self.matrix.get(level, {}).get(file_path, {})

    def get_related_files(self, file_path: str) -> list:
        """Get files related to the given file."""
        connections = self.get_connections(file_path, "files")
        return list(connections.keys())

    def get_blast_radius(self, file_path: str, threshold: float = 0.3) -> list:
        """Get files affected by changes to the given file.

        Returns list of (target, weight) tuples sorted by weight descending.
        """
        connections = self.get_connections(file_path, "files")
        affected = [(t, info['weight']) for t, info in connections.items()
                    if info['weight'] >= threshold]
        return sorted(affected, key=lambda x: x[1], reverse=True)

    def update_from_co_edits(self, co_edit_counts: Dict[tuple, int], min_count: int = 2):
        """Update edges based on co-edit patterns.

        co_edit_counts: dict mapping (file_a, file_b) -> co-edit count
        """
        if not co_edit_counts:
            return

        max_count = max(co_edit_counts.values())

        for (file_a, file_b), count in co_edit_counts.items():
            if count < min_count:
                continue

            weight = count / max_count
            self.add_edge(file_a, file_b, weight, "co_edits", level="files")
            self.add_edge(file_b, file_a, weight, "co_edits", level="files")

    def update_from_imports(self, imports: Dict[str, List[str]]):
        """Update edges based on import statements.

        imports: dict mapping file -> list of imported files
        """
        for file_path, imported_files in imports.items():
            for imported in imported_files:
                self.add_edge(file_path, imported, 0.8, "imports", level="files")

    def update_from_calls(self, calls: Dict[str, List[str]]):
        """Update edges based on function calls.

        calls: dict mapping "file::function" -> list of called "file::function"
        """
        for caller, callees in calls.items():
            for callee in callees:
                self.add_edge(caller, callee, 0.9, "calls", level="functions")

    def compute_centrality(self) -> dict:
        """Compute degree centrality for all files."""
        files = set()
        edge_count = {}

        for source, targets in self.matrix.get('files', {}).items():
            files.add(source)
            for target in targets:
                files.add(target)
                edge_count[source] = edge_count.get(source, 0) + 1
                edge_count[target] = edge_count.get(target, 0) + 1

        total = max(len(files), 1)
        centrality = {f: edge_count.get(f, 0) / total for f in files}
        return centrality

    def compute_betweenness(self) -> dict:
        """Simple betweenness approximation based on bridge connections."""
        files = set()
        for source, targets in self.matrix.get('files', {}).items():
            files.add(source)
            files.update(targets.keys())

        # Count how many times a file appears on paths between other files
        betweenness = {f: 0.0 for f in files}

        for source, targets in self.matrix.get('files', {}).items():
            for target in targets:
                # Simple: if source and target are in different clusters,
                # the files between them get betweenness
                for mid in files:
                    if mid != source and mid != target:
                        if (mid in self.matrix.get('files', {}).get(source, {}) and
                                target in self.matrix.get('files', {}).get(mid, {})):
                            betweenness[mid] += 1.0

        # Normalize
        max_b = max(betweenness.values()) if betweenness else 1.0
        if max_b > 0:
            betweenness = {f: v / max_b for f, v in betweenness.items()}

        return betweenness

    def get_clusters(self) -> dict:
        """Simple cluster detection based on connection density."""
        files = set()
        adjacency = {}
        for source, targets in self.matrix.get('files', {}).items():
            files.add(source)
            for target in targets:
                files.add(target)
                adjacency.setdefault(source, set()).add(target)
                adjacency.setdefault(target, set()).add(source)

        # Simple connected components via BFS
        visited = set()
        clusters = {}
        cluster_id = 0

        for file in files:
            if file in visited:
                continue
            queue = [file]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                clusters[current] = cluster_id
                for neighbor in adjacency.get(current, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            cluster_id += 1

        return clusters

    def _save(self):
        """Save matrix back to config."""
        self.config['interconnectedness'] = self.matrix
        save_config(self.config)


# --- 4. Test it ---

if __name__ == '__main__':
    manager = InterconnectednessManager()

    # Add some edges
    manager.add_edge('src/auth/login.py', 'src/auth/middleware.py', 0.9, 'calls', 'login flow')
    manager.add_edge('src/auth/login.py', 'src/db/users.py', 0.7, 'queries', 'user lookup')
    manager.add_edge('src/auth/middleware.py', 'src/auth/login.py', 0.8, 'validates', 'session check')
    manager.add_edge('src/auth/middleware.py', 'src/auth/tokens.py', 0.6, 'uses', 'token verification')

    print("=== Matrix ===")
    print(json.dumps(manager.get_matrix(), indent=2))

    print("\n=== Connections for src/auth/login.py ===")
    print(json.dumps(manager.get_connections('src/auth/login.py'), indent=2))

    print("\n=== Blast Radius ===")
    print(manager.get_blast_radius('src/auth/login.py', threshold=0.5))

    print("\n=== Centrality ===")
    print(json.dumps(manager.compute_centrality(), indent=2))

    print("\n=== Clusters ===")
    print(json.dumps(manager.get_clusters(), indent=2))

    print("\nDone.")
