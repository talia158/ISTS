"""
tests/test_generate_display_data.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Structure tests for backend.generate_display_data.

* cluster-only buffers  (length == maxClusteredNode/EdgePosition)
* full-length buffers   (clusters + UC nodes)
* blacklist content     (root clusters start un-hidden = 0)
* node→cluster mapping  integrity
* bitmap - index bounds safety
"""
import networkx as nx
import pytest

from main import (
    create_clustered_simulation_graph,
    calculate_layout,
    generate_display_data,
    reverse_cluster_map,
)


# ---------------------------------------------------------------------------
#  Minimal fixture: square graph → two clusters
# ---------------------------------------------------------------------------
def make_display_blob():
    g = nx.Graph()
    edges = [(0, 1), (1, 3), (3, 2), (2, 0)]
    for u, v in edges:
        g.add_edge(u, v, weight=1.0)
    for n in g.nodes():
        g.nodes[n]["threshold"] = 0.0

    cluster_map = {"0": "C0", "1": "C0", "2": "C1", "3": "C1"}
    rev = reverse_cluster_map(cluster_map)

    c_graph = create_clustered_simulation_graph(g, cluster_map)
    c_layout = calculate_layout(c_graph, seed=1)

    uc_layout = {
        "0": [c_layout["C0"][0] - 0.1, c_layout["C0"][1]],
        "1": [c_layout["C0"][0] + 0.1, c_layout["C0"][1]],
        "2": [c_layout["C1"][0] - 0.1, c_layout["C1"][1]],
        "3": [c_layout["C1"][0] + 0.1, c_layout["C1"][1]],
    }

    node_ts = {str(n): n for n in g.nodes()}
    edge_act = {}

    blob = generate_display_data(
        g,
        c_graph,
        uc_layout,
        c_layout,
        rev,
        cluster_map,
        node_ts,
        max_frame=10,
        heuristic="debug",
        execution_time=0.0,
        edge_activation_dict=edge_act,
    )
    return blob


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------
def test_array_lengths_match_contract():
    d = make_display_blob()["display_data"]

    n_clusters = d["maxClusteredNodePosition"]
    e_clusters = d["maxClusteredEdgePosition"]
    n_total    = len(d["circlePosList"])
    e_total    = len(d["startPosList"])

    # cluster-only buffers
    assert len(d["nodeScaleList"])      == n_clusters
    assert len(d["clusterSizeList"])    == n_clusters
    assert len(d["blacklistList"])      == n_clusters
    assert len(d["bitmapIndex"])        == e_clusters

    # full-length buffers
    assert len(d["circlePosList"])      == n_total
    assert len(d["nodeToClusterList"])  == n_total
    assert len(d["nodeToActiveTimestampList"]) == n_total
    assert len(d["edgeIDList"])         == e_total
    assert len(d["startPosList"])       == e_total
    assert len(d["edgeColorList"])      == e_total


def test_blacklist_content():
    d = make_display_blob()["display_data"]
    # root clusters are visible (0)
    assert set(d["blacklistList"]) == {0}


def test_node_to_cluster_mapping():
    blob = make_display_blob()
    d   = blob["display_data"]
    idx = blob["graph_data"]["node_label_to_list_index_dict"]
    cmap = blob["graph_data"]["cluster_map"]

    # For every UC node, verify mapping → parent-cluster index
    for uc_label, cl_label in cmap.items():
        if uc_label == cl_label:  # skip actual cluster nodes
            continue
        uc_idx = idx[uc_label]
        cl_idx = idx[cl_label]
        assert d["nodeToClusterList"][uc_idx] == cl_idx


def test_bitmap_index_bounds():
    d = make_display_blob()["display_data"]
    bmp   = d["bitmapList"]
    index = d["bitmapIndex"]

    for pos in index:
        assert 0 <= pos < len(bmp)
        count = bmp[pos]
        assert pos + 1 + count <= len(bmp)


def test_cluster_size_values():
    d = make_display_blob()["display_data"]
    # two clusters, each with 2 children
    assert d["clusterSizeList"] == [2, 2]
