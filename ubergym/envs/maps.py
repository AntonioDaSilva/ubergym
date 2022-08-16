from dataclasses import dataclass
from typing import List, Optional
import networkx as nx
from methodtools import lru_cache
import numpy as np

@dataclass
class Map:
    # we assume that the graph has nodes with nodes 0,1,...,N-1
    graph: nx.DiGraph

    def __post_init__(self):
        self.nodes = list(self.graph)
        self.distance = lru_cache()(self._distance)
        self.shortest_path = lru_cache()(self._shortest_path)

    def _distance(self, src, dst) -> float:
        return nx.algorithms.shortest_path_length(self.graph, src, dst, weight="weight")

    def _shortest_path(self, src, dst) -> List[int]:
        return nx.algorithms.shortest_path(self.graph, src, dst, weight="weight")

    def neighbors(self, src, dst) -> bool:
        return self.graph.has_edge(src, dst)

    def __len__(self) -> int:
        return len(self.nodes)

    def random_node(self) -> int:
        return np.random.randint(0, len(self.nodes))