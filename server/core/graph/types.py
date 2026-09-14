from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Edge:
    source: str
    target: str
    weight: float
    relation: str
    context: str = ""
    source_type: str = "file"
    target_type: str = "file"


@dataclass
class GraphNode:
    name: str
    node_type: str = "file"
    edges_out: List[Edge] = field(default_factory=list)
    edges_in: List[Edge] = field(default_factory=list)


@dataclass
class GraphMetrics:
    file_path: str
    degree_centrality: float
    betweenness_centrality: float
    cluster_id: int
    dependency_count: int
    dependents_count: int
    dependencies: List[str]
    dependents: List[str]
