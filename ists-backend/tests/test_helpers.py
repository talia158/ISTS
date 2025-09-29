"""
tests/test_helpers.py
~~~~~~~~~~~~~~~~~~~~~

Unit-tests for two low-level helpers in backend.py:

1. numbers_to_bitmaps
2. create_clustered_simulation_graph
"""
import math
from typing import List, Set

import networkx as nx
import pytest

from main import numbers_to_bitmaps, create_clustered_simulation_graph


# ---------------------------------------------------------------------------
#  numbers_to_bitmaps
# ---------------------------------------------------------------------------
def decode(bitmap: List[int]) -> Set[int]:
    """
    Reverse of numbers_to_bitmaps for test assertions.
    (Assumes input begins with length header.)
    """
    blocks = bitmap[0]
    out: Set[int] = set()
    for block_idx in range(blocks):
        word = bitmap[1 + block_idx]
        for bit in range(32):
            if word & (1 << bit):
                out.add(block_idx * 32 + bit)
    return out


@pytest.mark.parametrize(
    "src",
    [
        [],
        [0],
        [5, 0, 5, 31],                 # duplicates + boundary
        [10, 42, 99, 1000, 1023],      # spans many 32-bit blocks
    ],
)
def test_numbers_to_bitmaps_roundtrip(src):
    packed = numbers_to_bitmaps(src)
    assert packed[0] == len(packed) - 1             # header = #blocks
    assert decode(packed) == set(src)               # duplicates removed
    # no zero-padding beyond declared blocks
    assert all(b <= 0xFFFFFFFF for b in packed[1:])


# ---------------------------------------------------------------------------
#  create_clustered_simulation_graph
# ---------------------------------------------------------------------------
def build_sample_graph():
    """
    Cluster A : nodes 0,1
    Cluster B : nodes 2,3

    Edges:
        0-1   (intra A)  weight 4
        2-3   (intra B)  weight 5
        0-2   (inter A-B) weight 1.5
        1-2   (inter A-B) weight 2.0
    """
    g = nx.Graph()
    g.add_edge(0, 1, weight=4)
    g.add_edge(2, 3, weight=5)
    g.add_edge(0, 2, weight=1.5)
    g.add_edge(1, 2, weight=2.0)

    # map to clusters
    cl_map = {"0": "A", "1": "A", "2": "B", "3": "B"}
    return g, cl_map


def test_cluster_graph_node_set():
    g, cl_map = build_sample_graph()
    c_graph = create_clustered_simulation_graph(g, cl_map)

    # should contain exactly the two cluster nodes
    assert set(c_graph.nodes()) == {"A", "B"}
    # property exists
    for c in c_graph.nodes():
        assert math.isclose(c_graph.nodes[c]["activation_threshold"], 0.0)


def test_cluster_graph_edge_aggregation():
    g, cl_map = build_sample_graph()
    c_graph = create_clustered_simulation_graph(g, cl_map)

    # only one edge between A and B
    assert c_graph.number_of_edges() == 1
    w = c_graph["A"]["B"]["weight"]
    # expected weight = 1.5 + 2.0  (sum of inter-cluster edges)
    assert pytest.approx(w, rel=1e-6) == 3.5


def test_no_intra_cluster_edges():
    """
    Edges whose endpoints map to the *same* cluster must be ignored.
    """
    g, _ = build_sample_graph()
    # make every node belong to cluster 'X'
    cl_map_same = {str(n): "X" for n in g.nodes()}
    c_graph = create_clustered_simulation_graph(g, cl_map_same)

    # single node, no edges
    assert list(c_graph.nodes()) == ["X"]
    assert c_graph.number_of_edges() == 0
