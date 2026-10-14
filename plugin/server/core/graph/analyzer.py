import json
from typing import Dict, List

import numpy as np

from .interconnectedness import InterconnectednessManager

# --- 1. Graph analyzer ---

class GraphAnalyzer:
    def __init__(self, manager: InterconnectednessManager):
        self.manager = manager

    def analyze_file(self, file_path: str) -> dict:
        """Get comprehensive graph metrics for a file."""
        connections = self.manager.get_connections(file_path, "files")
        centrality = self.manager.compute_centrality()
        betweenness = self.manager.compute_betweenness()
        clusters = self.manager.get_clusters()

        # Dependencies: files this file calls/imports
        dependencies = [t for t, info in connections.items()
                       if info['relation'] in ('calls', 'imports')]

        # Dependents: files that call/import this file
        dependents = []
        for source, targets in self.manager.matrix.get('files', {}).items():
            if file_path in targets:
                dependents.append(source)

        return {
            'file_path': file_path,
            'degree_centrality': centrality.get(file_path, 0.0),
            'betweenness_centrality': betweenness.get(file_path, 0.0),
            'cluster_id': clusters.get(file_path, -1),
            'dependency_count': len(dependencies),
            'dependents_count': len(dependents),
            'dependencies': dependencies,
            'dependents': dependents,
            'out_degree': len(connections),
            'in_degree': len(dependents),
        }

    def analyze_all(self) -> dict:
        """Get metrics for all files in the graph."""
        files = set()
        for source, targets in self.manager.matrix.get('files', {}).items():
            files.add(source)
            files.update(targets.keys())

        results = {}
        for file_path in files:
            results[file_path] = self.analyze_file(file_path)

        return results

    def find_important_files(self, top_n: int = 10) -> list:
        """Find files with highest centrality (most connected)."""
        centrality = self.manager.compute_centrality()
        sorted_files = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_files[:top_n]

    def find_bridge_files(self) -> list:
        """Find files with high betweenness (bridges between clusters)."""
        betweenness = self.manager.compute_betweenness()
        return [(f, b) for f, b in betweenness.items() if b > 0.3]

    def suggest_edges(self, file_path: str) -> list:
        """Suggest potential connections based on patterns."""
        suggestions = []
        connections = self.manager.get_connections(file_path, "files")

        # Files with similar names might be related
        base_name = file_path.rsplit('/', 1)[-1].split('.')[0]
        for source, targets in self.manager.matrix.get('files', {}).items():
            if source == file_path:
                continue
            source_base = source.rsplit('/', 1)[-1].split('.')[0]
            if base_name == source_base and source not in connections:
                suggestions.append({
                    'target': source,
                    'reason': 'same base name',
                    'confidence': 0.5,
                })

        return suggestions


# --- 2. Test it ---

if __name__ == '__main__':
    from interconnectedness import InterconnectednessManager

    manager = InterconnectednessManager()

    # Build a small graph
    manager.add_edge('src/auth/login.py', 'src/auth/middleware.py', 0.9, 'calls')
    manager.add_edge('src/auth/login.py', 'src/db/users.py', 0.7, 'queries')
    manager.add_edge('src/auth/middleware.py', 'src/auth/tokens.py', 0.6, 'uses')
    manager.add_edge('src/db/users.py', 'src/db/schema.py', 0.5, 'defines')

    analyzer = GraphAnalyzer(manager)

    print("=== Analysis for src/auth/login.py ===")
    print(json.dumps(analyzer.analyze_file('src/auth/login.py'), indent=2))

    print("\n=== All Files ===")
    all_metrics = analyzer.analyze_all()
    for f, m in all_metrics.items():
        print(f"  {f}: centrality={m['degree_centrality']:.2f}, betweenness={m['betweenness_centrality']:.2f}")

    print("\n=== Top Files ===")
    for f, c in analyzer.find_important_files(3):
        print(f"  {f}: {c:.2f}")

    print("\nDone.")
