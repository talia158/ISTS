import math
import networkx as nx
import pytest

from main import expand_node, reverse_cluster_map

# ---------------------------------------------------------------------------
# helper to build a tiny root‑level context
# ---------------------------------------------------------------------------

def root_context():
    """Return a dict with all the arguments expand_node expects."""
    g = nx.Graph()
    g.add_weighted_edges_from([
        (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0),  # square (cluster A)
        (4, 5, 1.0),                                           # small cluster B
    ])

    cluster_map = {
        "0": "A", "1": "A", "2": "A", "3": "A",
        "4": "B", "5": "B",
    }
    rev_map = reverse_cluster_map(cluster_map)

    # "global" lists mimicking the root display payload order
    node_label_list = ["A", "B", "0", "1", "2", "3", "4", "5"]
    node_label_to_idx = {lbl: i for i, lbl in enumerate(node_label_list)}

    # timestamps: UC nodes fire at their int value; clusters -1
    node_ts_list = [-1, -1, 0, 1, 2, 3, 0, 0]

    # minimal global_layout / scale entries (clusters only)
    global_layout = {"A": [0.0, 0.0], "B": [10.0, 0.0]}
    global_scale  = {"A": 1.0, "B": 1.0}

    return {
        "graph": g,
        "cluster_map": cluster_map,
        "rev_map": rev_map,
        "node_label_list": node_label_list,
        "node_label_to_idx": node_label_to_idx,
        "node_ts_list": node_ts_list,
        "global_layout": global_layout,
        "global_scale": global_scale,
    }

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_expand_returns_null_for_singleton():
    ctx = root_context()

    # add a singleton cluster "S" containing node 6
    g = ctx["graph"]
    g.add_node(6, threshold=0.0)
    ctx["cluster_map"]["6"] = "S"
    ctx["rev_map"] = reverse_cluster_map(ctx["cluster_map"])
    ctx["global_layout"]["S"] = [20.0, 0.0]
    ctx["global_scale"]["S"] = 1.0

    data = expand_node(
        parent_depth=1,
        global_scale=ctx["global_scale"],
        graph=g,
        cluster_map=ctx["cluster_map"],
        reversed_cluster_map=ctx["rev_map"],
        node_label="S",
        node_label_list=ctx["node_label_list"],
        node_index_to_timestamp_list=ctx["node_ts_list"],
        node_label_to_list_index_dict=ctx["node_label_to_idx"],
        max_no_of_clusters=10,
        local_id_to_label_dict={},
        global_layout=ctx["global_layout"],
        edge_activation_dict={},
    )

    assert data["display_data"]["graphID"] == "NULL"


def test_expand_splits_large_cluster():
    ctx = root_context()

    data = expand_node(
        parent_depth=1,
        global_scale=ctx["global_scale"],
        graph=ctx["graph"],
        cluster_map=ctx["cluster_map"],
        reversed_cluster_map=ctx["rev_map"],
        node_label="A",
        node_label_list=ctx["node_label_list"],
        node_index_to_timestamp_list=ctx["node_ts_list"],
        node_label_to_list_index_dict=ctx["node_label_to_idx"],
        max_no_of_clusters=2,    # forces split of 4 UC nodes → 2 sub‑clusters
        local_id_to_label_dict={},
        global_layout=ctx["global_layout"],
        edge_activation_dict={},
    )

    disp = data["display_data"]
    subgraph = data["graph_data"]["subgraph"]

    # returned graphID should be a UUID (not NULL)
    assert disp["graphID"] != "NULL"

    # there should be exactly 2 *new* cluster nodes (A_0, A_1)
    new_cluster_labels = [n for n in subgraph.nodes() if n.startswith("A_")]
    assert len(new_cluster_labels) == 2

    # each new cluster size recorded correctly in clusterSizeList
    assert disp["clusterSizeList"] == [2, 2]

    # blacklist length equals node array length
    assert len(disp["blacklistList"]) == len(disp["circlePosList"])

    # every edge in edgeIDList is within bounds of circlePosList
    n_nodes = len(disp["circlePosList"])
    for u, v in disp["edgeIDList"]:
        assert 0 <= u < n_nodes and 0 <= v < n_nodes

    # bitmapIndex aligns with bitmapList
    bmp   = disp["bitmapList"]
    index = disp["bitmapIndex"]
    for pos in index:
        assert 0 <= pos < len(bmp)
        assert pos + 1 + bmp[pos] <= len(bmp)