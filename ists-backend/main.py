"""

This module defines a FastAPI application providing endpoints to:
  * Generate or load graphs (Watts‑Strogatz or Barabási‑Albert)
  * Run influence‑propagation simulations (Linear Threshold / Independent Cascade)
  * Cluster the resulting graphs with Louvain community detection (via RAPIDS cuGraph)
  * Prepare packed display data for a WebGL front‑end (circle‑packing layout, bit‑mapped edge activations, etc.)
  * Persist and reload simulation runs and heuristic performance data

"""

# ---------------------------------------------------------------------------
# Standard‑library imports
# ---------------------------------------------------------------------------
import json
import logging
import math
import os
import random
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import isfile, join
from typing import Any, Dict, List, Set, Tuple
import re
import uuid

# ---------------------------------------------------------------------------
# Third‑party imports
# ---------------------------------------------------------------------------

# GPU‑accelerated dataframe / graph libraries (RAPIDS)
import cudf
import cugraph

# General purpose libraries
import msgpack
import networkx as nx

# Web server & schema validation
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import Response

# NetworkX helper for serialisation
from networkx.readwrite import json_graph

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Logging setup & request/response logging middleware
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")

    # Log request body if it's a POST request
    if request.method in ("POST", "PUT"):
        body = await request.body()
        logger.info(f"Request Body: {body.decode('utf-8')}")

    response = await call_next(request)
    logger.info(f"Response: Status code {response.status_code}")

    return response


# ---------------------------------------------------------------------------
# In‑memory cache of recent simulation runs (keyed by integer ID)
# ---------------------------------------------------------------------------
simulation_cache = dict()


# ---------------------------------------------------------------------------
# Pydantic schemas for request payloads
# ---------------------------------------------------------------------------
class SimulationRequest(BaseModel):
    graph_name: str
    layout_seed: int
    simulation_seed: int
    heuristic_k: int
    model_name: str
    heuristic_name: str
    simulationID: int


class HeuristicRequest(BaseModel):
    heuristic_name: str
    heuristic: str


class GraphRequest(BaseModel):
    graph_type: str
    graph_seed: int
    n: int
    k: int
    m: int
    beta: float
    graph_name: str


class UserGraphRequest(BaseModel):
    graph_json: str
    graph_name: str


class HeuristicGetRequest(BaseModel):
    heuristic_name: str


class WSRequest(BaseModel):
    n: int
    k: int
    beta: float
    graph_name: str


class BARequest(BaseModel):
    n: int
    m: int
    graph_name: str


class GraphFileRequest(BaseModel):
    graph_json_string: str
    graph_name: str


def numbers_to_bitmaps(numbers: List[int]) -> List[int]:
    """Convert a list of non‑negative ints to a variable‑length 32‑bit bitmap list.

    Returned format: ``[block_count, block0, block1, ...]`` where each *block*
    is a 32‑bit bitmap representing presence within that slice.
    """
    from collections import defaultdict

    # If there are no numbers, keep the original behavior of returning [0].
    if not numbers:
        return [0]

    bitmaps_dict = defaultdict(int)

    # Fill blocks based on the 32-bit block index (num // 32)
    for num in numbers:
        idx = num // 32
        bit = num % 32
        bitmaps_dict[idx] |= (1 << bit)

    # Instead of only storing blocks that appear, fill up every block
    # from 0 up to the maximum block index encountered.
    max_idx = max(bitmaps_dict.keys())

    bitmaps = []
    for i in range(max_idx + 1):
        bitmaps.append(bitmaps_dict[i])  # defaults to 0 if not present

    # Prepend the number of blocks
    return [len(bitmaps)] + bitmaps


# ---------------------------------------------------------------------------
# Graph generation routines (with random weights & zero thresholds)
# ---------------------------------------------------------------------------
def generate_watts_strogatz(n: int, k: int, beta: float, seed: int) -> nx.Graph:
    # Seed random weight generation
    random.seed(seed)
    if k % 2 != 0:
        raise ValueError("k must be even.")
    g = nx.watts_strogatz_graph(n=n, k=k, p=beta, seed=seed)
    for u in g.nodes():
        g.nodes[u]["threshold"] = 0.0  # Assign 0 threshold for demo purposes
    for u, v in g.edges():
        rand_val = random.uniform(0.01, 1)
        g[u][v]["weight"] = rand_val
        g[v][u]["weight"] = rand_val
        # Assign random weights
    return g


def generate_barabasi_albert(n: int, m: int, seed: int) -> nx.Graph:
    g = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    for u in g.nodes():
        g.nodes[u]["threshold"] = 0.0  # Assign 0 threshold for demo purposes
    for u, v in g.edges():
        rand_val = random.uniform(0.01, 1)
        g[u][v]["weight"] = rand_val
        g[v][u]["weight"] = rand_val
    return g


# ---------------------------------------------------------------------------
# Clustering & layout helpers
# ---------------------------------------------------------------------------
def cluster_graph(g: nx.Graph) -> dict[str, str]:
    # Convert NetworkX graph to CuGraph format
    edges = [(u, v, data.get("weight", 1.0)) for u, v, data in g.edges(data=True)]
    df = cudf.DataFrame(edges, columns=["src", "dst", "weight"])
    g_cugraph = cugraph.Graph(directed=False)
    g_cugraph.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="weight")

    # Run Louvain clustering
    louvain_df, modularity_score = cugraph.louvain(g_cugraph)

    # Convert CuGraph results to a Python dictionary with string keys and values
    cluster_map = louvain_df.set_index("vertex")["partition"].to_pandas().to_dict()
    cluster_map_str = {str(k): str(v + g.number_of_nodes()) for k, v in cluster_map.items()}

    return cluster_map_str


def calculate_layout(g: nx.Graph, seed: int) -> Dict[str, List[float]]:
    pos = nx.spring_layout(G=g, scale=10, center=[0.0, 0.0], seed=seed)
    return {str(node): list(coord) for node, coord in pos.items()}


def create_clustered_simulation_graph(old_graph: nx.Graph, cluster_map: Dict[str, str]) -> nx.Graph:
    # Create a new graph for clusters
    clustered_graph = nx.Graph()

    # Store aggregated edge weights between clusters
    cluster_edge_weights = defaultdict(float)

    # Add nodes for each cluster
    for old_node in old_graph.nodes():
        cluster = cluster_map[str(old_node)]
        if cluster not in clustered_graph:
            clustered_graph.add_node(cluster, activation_threshold=0.0)

    # Use G.edges(data=True) to iterate over edges once
    for old_source, old_target, edge_data in old_graph.edges(data=True):
        source_cluster = cluster_map[str(old_source)]
        target_cluster = cluster_map[str(old_target)]
        weight = edge_data.get("weight", 1.0)  # Default weight if not specified

        # Ensure edges between different clusters are aggregated
        if source_cluster != target_cluster:
            clustered_edge = tuple(sorted([source_cluster, target_cluster]))  # Avoid duplicate edges
            cluster_edge_weights[clustered_edge] += weight  # Aggregate weight

    # Add aggregated edges to the clustered graph
    for (source_cluster, target_cluster), total_weight in cluster_edge_weights.items():
        clustered_graph.add_edge(source_cluster, target_cluster, weight=total_weight)

    return clustered_graph


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def initialize_storage(storage_dir: str = "storage"):
    os.makedirs(storage_dir, exist_ok=True)
    return storage_dir


def save_simulation_data(storage_dir: str, simulation_id: int, data: Dict[str, Any]):
    lock = threading.Lock()
    with lock:
        file_path = os.path.join(storage_dir, f"Simulation_{simulation_id}.bin")
        with open(file_path, "wb") as file:
            file.write(msgpack.packb(data, use_bin_type=True))


def save_heuristic_data(graph_name, no_nodes, no_edges, execution_time, heuristic_name, heuristic_k, simulationID):
    # Load heuristic data file (or create one if it doesn't exist)
    heuristic_data_dict = load_heuristic_data()
    # Update the dictionary
    if heuristic_name not in heuristic_data_dict:
        heuristic_data_dict[heuristic_name] = []
    heuristic_data_dict[heuristic_name] += [
        "Graph Name" + graph_name + " , Nodes:" + str(no_nodes) + " , Edges:" + str(no_edges) + " , K" + str(
            heuristic_k) + " , Runtime:" + str(round(execution_time, 8)) + str(simulationID)]
    # Save the file
    lock = threading.Lock()
    with lock:
        file_path = os.path.join("storage", "Performance Data.bin")
        with open(file_path, "wb") as file:
            file.write(msgpack.packb(heuristic_data_dict, use_bin_type=True))


def load_heuristic_data() -> Dict[str, Any]:
    file_path = os.path.join("storage", "Performance Data.bin")
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "rb") as file:
        return msgpack.unpackb(file.read(), raw=False)


def load_simulation_data(storage_dir: str, simulation_id: int) -> Dict[str, Any]:
    file_path = os.path.join(storage_dir, f"Simulation_{simulation_id}.bin")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as file:
        unserialized_data = msgpack.unpackb(file.read(), raw=False)
        # Parse the NetworkX Graph
        return {  # todo make this neater
            "display_data": unserialized_data["display_data"],
            "graph_data": {
                "node_label_to_list_index_dict": unserialized_data["graph_data"]["node_label_to_list_index_dict"],
                "node_label_list": unserialized_data["graph_data"]["node_label_list"],
                "reversed_cluster_map": unserialized_data["graph_data"]["reversed_cluster_map"],
                "uc_graph": nx.node_link_graph(unserialized_data["graph_data"]["uc_graph_dict"], edges="edges"),
                "c_graph": nx.node_link_graph(unserialized_data["graph_data"]["c_graph_dict"], edges="edges"),
                "local_id_to_label_dict": unserialized_data["graph_data"]["local_id_to_label_dict"],
                "cluster_map": unserialized_data["graph_data"]["cluster_map"],
                "global_layout": unserialized_data["graph_data"]["global_layout"],
                "global_scale": unserialized_data["graph_data"]["global_scale"],
                "graph_dictionary": {key: nx.node_link_graph(value, edges="edges") for key, value in
                                     unserialized_data["graph_data"]["graph_dictionary"].items()},
                "edge_activation_dict": unserialized_data["graph_data"]["edge_activation_dict"]
            }
        }


# ---------------------------------------------------------------------------
# Heuristic execution & simulation models
# ---------------------------------------------------------------------------
def generate_seed_set(g: nx.Graph, heuristic: str, k: int) -> List[str]:
    if heuristic == "DEBUG":
        heuristic = """
def run_heuristic(g, k):
    import random
    return [str(node) for node in random.sample(list(g.nodes()), min(k, len(g.nodes())))]
"""
    local_vars = {}
    start_time = time.perf_counter()
    exec(heuristic, {}, local_vars)
    end_time = time.perf_counter()
    difference = end_time - start_time
    return [local_vars["run_heuristic"](g, k), difference]


def _ic_process_node(
        g: nx.Graph,
        current: str,
        active: Set[str],
        rnd: random.Random
) -> Tuple[Set[str], List[Tuple[str, str]]]:
    newly_activated = set()
    fired_edges = []
    curr_int = int(current)

    for neighbor in g.neighbors(curr_int):
        neighbor_str = str(neighbor)
        # If neighbor not active yet, attempt to activate
        if neighbor_str not in active and rnd.random() < g[curr_int][neighbor]["weight"]:
            newly_activated.add(neighbor_str)
            # Record which edge actually propagated influence
            fired_edges.append((current, neighbor_str))

    return newly_activated, fired_edges


influence_lock = threading.Lock()


def independent_cascade_model(g: nx.Graph, seed_set: List[str]):
    active = set(seed_set)
    newly_active = deque(seed_set)

    # Track activation times (timestep) for nodes
    activation_time = {node: 0 for node in seed_set}

    # Track which timesteps each edge successfully propagated influence
    edge_activations = defaultdict(list)

    rnd = random.Random()
    timestep = 1

    while newly_active:
        futures = []
        new_activations = set()

        with ThreadPoolExecutor() as executor:
            while newly_active:
                current = newly_active.popleft()
                # Submit tasks in parallel
                futures.append(executor.submit(_ic_process_node, g, current, active, rnd))

            # Once all threads are completed:
            for future in as_completed(futures):
                newly_activated_nodes, fired_edges = future.result()
                # Update the set of newly activated nodes
                new_activations.update(newly_activated_nodes)
                # Record which edges propagated influence at this timestep
                for edge in fired_edges:
                    edge_activations[edge].append(timestep)

        # Record the activation time for any nodes activated in this step
        for node in new_activations:
            if node not in activation_time:
                activation_time[node] = timestep

        # If any new nodes got activated, move them into the active set/queue
        if new_activations:
            active.update(new_activations)
            newly_active.extend(new_activations)

        timestep += 1

    # Return activation times, total timesteps used, and edge activation dictionary
    return [activation_time, timestep, dict(edge_activations)]


def _ltm_process_node(g: nx.Graph, current: str, active: Set[str], influence: Dict[str, float]):
    newly_activated = set()
    fired_edges = []
    curr_int = int(current)

    for neighbor in g.neighbors(curr_int):
        neighbor_str = str(neighbor)
        if neighbor_str not in active:
            with influence_lock:
                influence[neighbor] += g[curr_int][neighbor]["weight"]
                current_influence = influence[neighbor]
            if current_influence >= g.nodes[neighbor]["threshold"]:
                newly_activated.add(neighbor_str)
                fired_edges.append((current, neighbor_str))
    return newly_activated, fired_edges


def linear_threshold_model(g: nx.Graph, seed_set: List[str]):
    active = set(seed_set)
    influence = defaultdict(float)
    queue = deque(seed_set)

    # Track activation times (timestep) for nodes
    activation_time = {node: 0 for node in seed_set}

    # Track edges that cause activation at each timestep
    edge_activations = defaultdict(list)

    timestep = 1

    while queue:
        futures = []
        new_activations = set()

        with ThreadPoolExecutor() as executor:
            while queue:
                current = queue.popleft()
                futures.append(
                    executor.submit(_ltm_process_node, g, current, active, influence)
                )

            # Gather results
            for future in as_completed(futures):
                newly_activated_nodes, fired_edges = future.result()
                new_activations.update(newly_activated_nodes)
                # Record times when edges succeed in activating neighbors
                for edge in fired_edges:
                    edge_activations[edge].append(timestep)

        # Record the activation time for any nodes activated in this step
        for node in new_activations:
            if node not in activation_time:
                activation_time[node] = timestep

        # If new activations occurred, update sets and queue
        if new_activations:
            active.update(new_activations)
            queue.extend(new_activations)

        timestep += 1

    return [activation_time, timestep, dict(edge_activations)]


def reverse_cluster_map(cluster_map: Dict[str, str]) -> Dict[str, List[str]]:
    reversed_cluster_map = {}
    for key, value in cluster_map.items():
        reversed_cluster_map.setdefault(value, []).append(key)
    return reversed_cluster_map


def circle_packing(n, r, x_c, y_c):
    R = r * (1 + math.sqrt(n))
    positions = [[x_c, y_c]]

    if n == 1:
        return positions

    angle_increment = 2 * math.pi / n
    current_angle = 0
    for i in range(1, n):
        x = (R - r) * math.cos(current_angle) + x_c
        y = (R - r) * math.sin(current_angle) + y_c
        positions.append([x, y])
        current_angle += angle_increment

    return positions


def pack_circles(reversed_cluster_map: Dict[str, List[str]],
                 clustered_layout: Dict[str, List[float]],
                 minimum_unclustered_node_radius: float) -> Dict[str, List[float]]:
    unclustered_layout = {}

    for p_cluster, children in reversed_cluster_map.items():
        if p_cluster not in clustered_layout:
            continue

        parent_cluster_center = clustered_layout[p_cluster]
        cluster_size = len(children)

        # Utility method for circle packing
        packed_circles = circle_packing(
            n=cluster_size,
            r=minimum_unclustered_node_radius,
            x_c=parent_cluster_center[0],
            y_c=parent_cluster_center[1]
        )

        for i, position in enumerate(packed_circles):
            unclustered_layout[children[i]] = position

    return unclustered_layout


def generate_display_data(graph_uc, graph_c, layout_uc, layout_c, reversed_cluster_map, cluster_map,
                          node_label_to_timestep_dict, max_frame, heuristic, execution_time, edge_activation_dict) -> \
Dict[
    str, any]:
    start_pos_list = []
    end_pos_list = []
    circle_pos_list = []
    node_scale_list = []
    edge_color_list = []
    uc_node_index_to_c_node_index = []
    cluster_size_list = []
    node_index_to_timestamp_list = []
    node_label_list = []
    node_label_to_list_index_dict = dict()
    edge_id_list = []
    global_scale_dict = dict()
    local_id_to_label_dict = dict()
    graph_id = str(uuid.uuid4())

    c_edge_to_index_dict = dict()
    edge_timestep_list = [[] for _ in range(graph_c.number_of_edges())]  # Memory ineffieiency here...

    edge_tracker = set()

    node_index = 0
    # Node list is the key of adjacency list
    # Edge adding logic means each edge is added twice for UD graphs...
    node_list = list(graph_c.nodes())  # Preserve order
    # Handle C nodes
    for s_node in node_list:
        start_node = str(s_node)  # unnecessary
        node_label_list.append(start_node)
        local_id_to_label_dict[str((node_index, graph_id))] = s_node
        # Populate node name -> list index for later use
        node_label_to_list_index_dict[start_node] = node_index
        uc_node_index_to_c_node_index.append(node_index)
        node_index += 1
        node_index_to_timestamp_list.append(-1)  # Calculated in shader...
        cluster_size_list.append(len(reversed_cluster_map[start_node]))

        start_pos = layout_c[start_node]
        start = (
            float(start_pos[0]),
            float(start_pos[1])
        )

        circle_pos_list.append(start)
        # Node size proportional to cluster size / total size
        scale = 10 * (len(reversed_cluster_map[start_node]) / len(graph_c.nodes()))
        node_scale_list.append(scale)
        global_scale_dict[start_node] = scale
    # Handle C edges
    for s_node in node_list:
        start_node = str(s_node)  # unnecessary
        start_pos = layout_c[start_node]
        start = (
            float(start_pos[0]),
            float(start_pos[1])
        )

        # Get neighbours
        neighbours = graph_c.neighbors(s_node)
        for e_node in neighbours:
            end_node = str(e_node)
            if (end_node, start_node) in edge_tracker:
                continue
            c_edge_to_index_dict[tuple(sorted([start_node, end_node]))] = len(edge_id_list)
            edge_id_list.append((node_label_to_list_index_dict[start_node], node_label_to_list_index_dict[end_node]))
            edge_tracker.add((start_node, end_node))
            end_pos = layout_c[end_node]
            end = (
                float(end_pos[0]),
                float(end_pos[1])
            )

            start_pos_list.append(start)
            end_pos_list.append(end)

            # Tint corresponding to weight
            tint = graph_c[s_node][e_node]["weight"]
            edge_color_list.append((tint, tint, tint, 1.0))  # RGBA color

    node_list = list(graph_uc.nodes())
    # Handle UC Nodes
    for s_node in node_list:
        local_id_to_label_dict[str((node_index, graph_id))] = s_node
        start_node = str(s_node)
        node_label_to_list_index_dict[start_node] = node_index
        node_index += 1
        node_label_list.append(start_node)
        if start_node in node_label_to_timestep_dict:
            node_index_to_timestamp_list.append(
                node_label_to_timestep_dict[start_node])  # Redundant conversion here?
        else:
            node_index_to_timestamp_list.append(-2)
        uc_node_index_to_c_node_index.append(
            # Corresponding index of c in the lists we have built so far
            node_label_to_list_index_dict[
                # String of which cluster the current node belongs to (c)
                cluster_map[start_node]
            ]
        )
        start_pos = layout_uc[start_node]
        start = (
            float(start_pos[0]),
            float(start_pos[1])
        )

        circle_pos_list.append(start)
    # Handle UC Edges
    for s_node in node_list:
        start_node = str(s_node)

        start_pos = layout_uc[start_node]
        start = (
            float(start_pos[0]),
            float(start_pos[1])
        )
        # Get neighbours
        neighbours = graph_uc.neighbors(s_node)
        for e_node in neighbours:
            end_node = str(e_node)
            if (end_node, start_node) in edge_tracker:
                continue
            edge_id_list.append((node_label_to_list_index_dict[start_node], node_label_to_list_index_dict[end_node]))
            edge_tracker.add((start_node, end_node))

            # Get the corresponding edge in the clustered graph - tuple sorted thing to guarantee same order ...
            corresponding_clustered_edge_tuple = tuple(sorted([cluster_map[start_node], cluster_map[end_node]]))
            # If the edge connects 2 clusters...
            if corresponding_clustered_edge_tuple in c_edge_to_index_dict:
                c_edge_list_index = c_edge_to_index_dict[corresponding_clustered_edge_tuple]
                # Add the edge times
                tuple_t = str((start_node, end_node))
                if tuple_t in edge_activation_dict:
                    edge_timestep_list[c_edge_list_index] += edge_activation_dict[tuple_t]
                tuple_t1 = str((end_node, start_node))
                if tuple_t1 in edge_activation_dict:
                    edge_timestep_list[c_edge_list_index] += edge_activation_dict[tuple_t1]  # is this required?

            end_pos = layout_uc[end_node]
            end = (
                float(end_pos[0]),
                float(end_pos[1])
            )
            start_pos_list.append(start)
            end_pos_list.append(end)
            # Tint corresponding to weight
            tint = graph_uc[s_node][e_node]["weight"]
            edge_color_list.append((tint, tint, tint, 1.0))
    # Now we convert into the required bitmap format
    edge_id_to_bitmap_list_index = []
    bitmap_list = []
    curr_index = 0
    for i in range(len(edge_timestep_list)):
        # Flatten the list using the index
        chunk = numbers_to_bitmaps(edge_timestep_list[i])
        bitmap_list += chunk
        edge_id_to_bitmap_list_index.append(curr_index)
        curr_index += len(chunk)

    c_graph_dict = nx.node_link_data(graph_c, edges="edges")
    data = {
        "display_data": {
            "circlePosList": circle_pos_list,
            "nodeScaleList": node_scale_list,
            "startPosList": start_pos_list,
            "endPosList": end_pos_list,
            "edgeColorList": edge_color_list,
            "nodeToClusterList": uc_node_index_to_c_node_index,
            "clusterSizeList": cluster_size_list,
            "nodeToActiveTimestampList": node_index_to_timestamp_list,
            "maxFrame": max_frame,
            "graphID": graph_id,
            "edgeIDList": edge_id_list,
            "maxUnclusteredNodePosition": graph_uc.number_of_nodes(),
            "maxClusteredNodePosition": graph_c.number_of_nodes(),
            "maxClusteredEdgePosition": graph_c.number_of_edges(),
            "blacklistList": [0 for _ in range(graph_c.number_of_nodes())],
            "parentScale": 1.0,
            # "edgeTimestepList": edge_timestep_list,
            "bitmapList": bitmap_list,
            "bitmapIndex": edge_id_to_bitmap_list_index,
        }, "graph_data": {
            "node_label_list": node_label_list,
            "node_label_to_list_index_dict": node_label_to_list_index_dict,
            "uc_graph_dict": nx.node_link_data(graph_uc, edges="edges"),
            "c_graph_dict": c_graph_dict,
            "reversed_cluster_map": reversed_cluster_map,
            "cluster_map": cluster_map,
            "local_id_to_label_dict": local_id_to_label_dict,
            "global_layout": layout_c,
            "global_scale": global_scale_dict,
            "graph_dictionary": {graph_id: c_graph_dict},
            "edge_activation_dict": edge_activation_dict
        },
        "heuristic_data": {
            "heuristic": heuristic,
            "execution_time": execution_time
        }
    }
    return data


def expand_node(parent_depth, global_scale, graph, cluster_map, reversed_cluster_map, node_label, node_label_list,
                node_index_to_timestamp_list, node_label_to_list_index_dict,
                max_no_of_clusters, local_id_to_label_dict, global_layout, edge_activation_dict):
    # ID system here - there are edges between ints and strs
    node_to_expand = node_label  # investigate todo
    # If node can't be expanded, return nothing
    unclustered_node_count = len(reversed_cluster_map[node_to_expand])
    max_size = math.ceil(unclustered_node_count / max_no_of_clusters)
    if unclustered_node_count <= max_size:
        return {
            "display_data": {
                "graphID": "NULL"
            }
        }
    subgraph = nx.Graph()
    subgraph_id = str(uuid.uuid4())
    # Get all nodes belonging to the node to expand
    root_child_nodes = reversed_cluster_map[node_to_expand]
    # Partition using modulo operator
    current_expanded_cluster = ""
    i = 0
    i2 = -1  # So zero indexed...
    edge_to_count_dict = dict()

    edge_timestep_dict = dict()
    edge_timestep_list = []

    new_node_to_cluster_list = []
    new_cluster_size_list = []
    new_node_timestamp_list = []

    max_uc_node = 0

    new_node_set = set()
    for node in root_child_nodes:
        max_uc_node = max(max_uc_node, int(node))
        # We look at all nodes from the original graph that the cluster contains
        # Update the new timestamp list with only the nodes from the original graph which are
        # contained within the cluster
        new_node_timestamp_list.append(
            # Node global list index -> timestamp
            node_index_to_timestamp_list[
                # Node 'label' -> global list index
                node_label_to_list_index_dict[node]
            ]
        )
        # If a cluster becomes full, we shift i2 to reflect the creation of a new cluster
        if i % max_size == 0:
            # Create a new partition
            i2 += 1
            current_expanded_cluster = node_to_expand + "_" + str(i2)
            # ID of the new clustered node maps to the 'label'
            # local_id_to_label_dict[str((i2, subgraph_id))] = current_expanded_cluster
            # Add new entry for the cluster contents
            reversed_cluster_map[current_expanded_cluster] = []
            # We assume the cluster will be at full capacity...
            new_cluster_size_list.append(max_size)
            # Update the set of newly created nodes
            new_node_set.add(current_expanded_cluster)
            # Add to label dict (we map the new nodes to indexes from 0 - n, corresponding to the points)
            node_label_to_list_index_dict[current_expanded_cluster] = i2
            # Add node to subgraph
            subgraph.add_node(current_expanded_cluster)

        # Add the node to the new cluster. (root node label -> [list of corresponding clusters]) todo introduce graph id system
        reversed_cluster_map[current_expanded_cluster].append(node)
        if type(cluster_map[node]) is not list:
            cluster_map[node] = [cluster_map[node]]
        cluster_map[node].append(current_expanded_cluster)
        i += 1
        # Update node id -> cluster list
        new_node_to_cluster_list.append(i2)
    # If the last cluster wasn't fully filled up...
    if i % max_size != 0:
        new_cluster_size_list[-1] = (i % max_size)

    edge_tracker = set()
    aggregated_edge_activation_dict = dict()

    # Aggregate edges from original graph now that the cluster maps have been updated
    for node in root_child_nodes:
        node_str = str(node)
        current_expanded_cluster = cluster_map[node][-1]  # We look at the latest cluster assignment
        for neighbour in graph.neighbors(int(node)):
            neighbour_str = str(neighbour)
            # Find which cluster the neighbour belongs to
            corresponding_cluster = cluster_map[neighbour_str]
            # If more than one cluster mapping exists, take the most recent (ie last) one
            if type(corresponding_cluster) is list:
                corresponding_cluster = corresponding_cluster[-1]  # most likely the fault comes from here...
            # Add the edge to the subgraph if the original edge is between unique clusters
            if corresponding_cluster != current_expanded_cluster:
                # Make the edge tuple
                edge_tuple = str(tuple(sorted([corresponding_cluster, current_expanded_cluster])))
                # Update edge timestep dict
                # This is fine because we take for each edge in the unclustered graph...
                tuple_t = str((node_str, neighbour_str))  # modded here
                if edge_tuple not in aggregated_edge_activation_dict:
                    aggregated_edge_activation_dict[edge_tuple] = []
                if tuple_t in edge_activation_dict:
                    aggregated_edge_activation_dict[edge_tuple] += edge_activation_dict[tuple_t]
                tuple_t1 = str((neighbour_str, node_str))
                if tuple_t1 in edge_activation_dict:
                    aggregated_edge_activation_dict[edge_tuple] += edge_activation_dict[tuple_t1]

                # Get the weight of the corresponding edge to be aggregated
                weight_of_uc_edge = graph[int(node)][neighbour]["weight"]  # todo would weight issues be here...
                # Update edge tuple-> [sum of weights from other aggregated edges, no. of aggregated edges] dict
                # todo maybe the undirected thing is why the expanded weights look diff: we take graph[u][v] not [v][u]...
                if edge_tuple not in edge_to_count_dict:
                    edge_to_count_dict[edge_tuple] = [0, 0]
                aggregated_edge_info = edge_to_count_dict[edge_tuple]
                aggregated_edge_info[0] += weight_of_uc_edge
                aggregated_edge_info[1] += 1
                # Divide sum of all weights by number of weights to get average
                aggregated_weight = aggregated_edge_info[0] / aggregated_edge_info[1]
                subgraph.add_edge(current_expanded_cluster, corresponding_cluster)
                subgraph[current_expanded_cluster][corresponding_cluster]["weight"] = aggregated_weight

    # Layout subgraph.
    pos = nx.spring_layout(G=subgraph, scale=1 / (math.pow(10, parent_depth - 1)), center=global_layout[node_to_expand],
                           seed=1)  # todo set seed to graph seed...
    layout = {str(node): list(coord) for node, coord in pos.items()}
    # Modify overlapping (ie edges where u or v aren't in new_node_list)
    for node in layout:
        if node not in new_node_set:
            # Replace with prior position.
            layout[node] = global_layout[node]
        else:
            # Node is new and doesn't exist in global layout, so update
            global_layout[node] = layout[node]

    new_circle_pos_list = [0] * subgraph.number_of_nodes()
    new_start_pos_list = []
    new_end_pos_list = []
    new_edge_color_list = []
    new_node_scale_list = []
    edge_id_list = []
    edge_tracker = set()
    parent_scale = global_scale[node_to_expand]
    sorted_edge_tuple_list = []

    # max_id_0 = 0

    # does order matter here?

    # First, add 'invisible' nodes in the space after the visible nodes and maintain label -> index list
    new_label_to_local_index_dict = dict()
    i3 = len(new_node_set)
    for node in subgraph.nodes():
        if node not in new_node_set:
            new_circle_pos_list[i3] = layout[node]
            new_label_to_local_index_dict[node] = i3
            local_id_to_label_dict[str((i3, subgraph_id))] = str(node)
            i3 += 1
    i3 = 0
    for node in subgraph.nodes():
        if node in new_node_set:
            new_circle_pos_list[i3] = layout[node]
            new_label_to_local_index_dict[node] = i3
            local_id_to_label_dict[str((i3, subgraph_id))] = node
            i3 += 1
    # Positions only for the new nodes and their ingoing and outgoing edges are considered here...
    for start_node in sorted(new_node_set):  # List reused to make sure generated lists correspond
        scale = max(0.3, parent_scale * (len(reversed_cluster_map[start_node]) / unclustered_node_count))
        # # print(scale)
        global_scale[start_node] = scale
        new_node_scale_list.append(scale)
        node_label_list.append(start_node)
        neighbours = subgraph.neighbors(start_node)
        start_pos = layout[start_node]
        start = (
            float(start_pos[0]),
            float(start_pos[1])
        )
        # new_circle_pos_list.append(start)
        for end_node in neighbours:
            # Avoid duplicate edges in other directions -can this check be extended for duplicates in same direction?
            if (end_node, start_node) in edge_tracker:
                continue
            edge_tuple = (start_node, end_node)
            sorted_tuple = str(tuple(sorted([start_node, end_node])))
            sorted_edge_tuple_list.append(sorted_tuple)
            edge_tracker.add(edge_tuple)
            end_pos = layout[end_node]
            end = (
                float(end_pos[0]),
                float(end_pos[1])
            )
            # # print(str(start) + " , " + str(end))
            new_start_pos_list.append(start)
            new_end_pos_list.append(end)
            # edge_id_list.append((node_label_to_list_index_dict[start_node], node_label_to_list_index_dict[end_node]))
            edge_id_list.append((new_label_to_local_index_dict[start_node], new_label_to_local_index_dict[end_node]))
            # max_id_0 = max(edge_id_list[-1][0], edge_id_list[-1][1], max_id_0)

            tint = subgraph[start_node][end_node]["weight"]
            new_edge_color_list.append((tint, tint, tint, 1.0))

    # Now we convert into the required bitmap format
    edge_id_to_bitmap_list_index = []
    bitmap_list = []
    curr_index = 0
    for i in range(len(sorted_edge_tuple_list)):
        # Flatten the list using the index
        chunk = numbers_to_bitmaps(aggregated_edge_activation_dict[sorted_edge_tuple_list[i]])
        bitmap_list += chunk
        edge_id_to_bitmap_list_index.append(curr_index)
        curr_index += len(chunk)
    data = {
        "display_data": {
            "circlePosList": new_circle_pos_list,
            "nodeScaleList": new_node_scale_list,
            "startPosList": new_start_pos_list,
            "endPosList": new_end_pos_list,
            "edgeColorList": new_edge_color_list,
            "nodeToClusterList": new_node_to_cluster_list,
            "clusterSizeList": new_cluster_size_list,
            "nodeToActiveTimestampList": new_node_timestamp_list,
            "maxUnclusteredNodePosition": len(new_node_timestamp_list),  # For compute shader...
            "maxClusteredNodePosition": len(new_node_set),
            "maxClusteredEdgePosition": len(new_start_pos_list),
            "edgeIDList": edge_id_list,
            "graphID": subgraph_id,
            "blacklistList": [0] * len(new_circle_pos_list),
            "parentScaleFactor": parent_scale,
            # "edgeTimestepList": edge_timestep_list,
            "bitmapList": bitmap_list,
            "bitmapIndex": edge_id_to_bitmap_list_index,
        },
        "graph_data": {
            # "subgraph": nx.node_link_data(subgraph, edges="edges")
            "subgraph": subgraph
        }
    }
    # print(data)
    return data


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.get("/api/simulation/{simulationID}/{nodeID}/{subgraphID}/{depth}/expand")
def handle_expansion(simulationID: int, nodeID: int, subgraphID: str, depth: int):
    data = simulation_cache[simulationID]
    graph_data = data["graph_data"]
    display_data = data["display_data"]
    graph = graph_data["uc_graph"]
    cluster_map = graph_data["cluster_map"]
    reversed_cluster_map = graph_data["reversed_cluster_map"]
    local_id_to_label_dict = graph_data["local_id_to_label_dict"]
    node_label = local_id_to_label_dict[str((nodeID, subgraphID))]
    node_label_list = graph_data["node_label_list"]
    node_index_to_timestamp_list = display_data["nodeToActiveTimestampList"]
    node_label_to_list_index_dict = graph_data["node_label_to_list_index_dict"]
    global_layout = graph_data["global_layout"]
    global_scale = graph_data["global_scale"]
    edge_activation_dict = graph_data["edge_activation_dict"]
    max_size = 10
    # todo is modification of simulation_cache in sync with file?
    # We have parent subgraph id coming in, so we can map circlePosList etc to subgraphID
    # circle_pos_list and node_label_to_list_index_dict
    data = expand_node(depth, global_scale, graph, cluster_map, reversed_cluster_map, node_label, node_label_list,
                       node_index_to_timestamp_list,
                       node_label_to_list_index_dict, max_size, local_id_to_label_dict, global_layout,
                       edge_activation_dict)
    # Update graph dictionary
    graph_data["graph_dictionary"][data["display_data"]["graphID"]] = data["graph_data"]["subgraph"]
    # Serialize data to binary using msgpack
    binary_data = msgpack.packb(data["display_data"], use_bin_type=True)

    # Return the binary response with appropriate content type
    return Response(content=binary_data, media_type="application/octet-stream")


@app.post("/api/simulation/run")
def run(request: SimulationRequest):
    # Load graph

    g = None
    file_path = os.path.join("graphs", f"{request.graph_name}")
    if not os.path.exists(file_path):
        g = generate_watts_strogatz(10000, 4, 0.3, 1)
    else:
        with open(file_path, "r") as file:
            g_json = json.load(file)
            g = nx.node_link_graph(g_json, edges="edges")

    # Load Heuristic
    file_path = os.path.join("heuristics", f"{request.heuristic_name}.heuristic")
    if not os.path.exists(file_path):
        heuristic = "DEBUG"
    else:
        with open(file_path, "rb") as file:
            heuristic = file.read().decode('utf-8')
    # Run simulation
    heuristic_results = generate_seed_set(g, heuristic, request.heuristic_k)
    seed_set = heuristic_results[0]
    execution_time = heuristic_results[1]
    random.seed(request.simulation_seed)
    simulation_results = linear_threshold_model(g,
                                                seed_set) if request.model_name == "Linear Threshold" else independent_cascade_model(
        g, seed_set) if request.model_name == "Independent Cascade" else {}

    node_label_to_timestep_dict = simulation_results[0]
    max_frame = simulation_results[1]
    edge_activation_dict = dict()
    for key, value in simulation_results[2].items():
        edge_activation_dict[str(key)] = value

    # Generate cluster map
    cluster_map = cluster_graph(g)
    # Create clustered simulation graph
    clustered_graph = create_clustered_simulation_graph(g, cluster_map)
    # Calculate layout for clustered graph
    clustered_layout = calculate_layout(clustered_graph, request.layout_seed)
    reversed_cluster_map = reverse_cluster_map(cluster_map)
    # thinking about unclustered layout. we have points for clustered layout
    unclustered_layout = pack_circles(reversed_cluster_map, clustered_layout, 0.05)
    # Save data
    display_data = generate_display_data(g, clustered_graph,
                                         unclustered_layout, clustered_layout, reversed_cluster_map,
                                         cluster_map, node_label_to_timestep_dict, max_frame,
                                         request.heuristic_name, execution_time, edge_activation_dict)
    if request.simulationID in simulation_cache:
        simulation_cache.pop(request.simulationID)
    save_simulation_data(s_dir, request.simulationID,
                         display_data)
    save_heuristic_data(request.graph_name, g.number_of_nodes(), g.number_of_edges(), execution_time,
                        request.heuristic_name,
                        request.heuristic_k, request.simulationID)

    return {"message": f"Simulation {request.simulationID} completed and saved"}


@app.get("/api/simulation/get_heuristic_list")
def get_heuristic_list():
    # Move this to a seperate method
    in_storage_list = [f[:-10] for f in os.listdir("heuristics")
                       if isfile(join("heuristics", f)) and ".heuristic" in f]
    in_storage_list.sort()  # Is this necessary?
    data = {
        "heuristicList": in_storage_list
    }
    # Serialize data to binary using msgpack
    binary_data = msgpack.packb(data, use_bin_type=True)
    # Return the binary response with appropriate content type
    return Response(content=binary_data, media_type="application/octet-stream")


@app.post("/api/simulation/get_heuristic")
def open_heuristic(request: HeuristicGetRequest):
    file_path = os.path.join("heuristics", f"{request.heuristic_name}.heuristic")
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r") as file:
        return Response(content=msgpack.packb({"heuristic": file.read()}, use_bin_type=True),
                        media_type="application/octet-stream")


@app.get("/api/simulation/get_graph_list")
def get_graph_list():
    # Move this to a separate method
    in_storage_list = [f for f in os.listdir("graphs")
                       if isfile(join("graphs", f)) and ".json" in f]
    in_storage_list.sort()  # Is this necessary?
    data = {
        "graphList": in_storage_list
    }
    # Serialize data to binary using msgpack
    binary_data = msgpack.packb(data, use_bin_type=True)
    # Return the binary response with appropriate content type
    return Response(content=binary_data, media_type="application/octet-stream")


@app.post("/api/simulation/save_heuristic")
def save_heuristic(heuristic_request: HeuristicRequest):
    # Move this to a seperate method
    if heuristic_request.heuristic_name != "" and heuristic_request.heuristic != "":
        # Save the heuristic as a file
        lock = threading.Lock()
        with lock:
            file_path = os.path.join("heuristics", f"{heuristic_request.heuristic_name}.heuristic")
            with open(file_path, "w") as file:
                file.write(heuristic_request.heuristic)


@app.post("/api/simulation/generate_ws_ba")
def handle_ws_ba(request: GraphRequest):
    if request.graph_type == "WS":
        lock = threading.Lock()
        with lock:
            file_path = os.path.join("graphs", f"{request.graph_name}.json")
            with open(file_path, "w") as file:
                json.dump(
                    json_graph.node_link_data(
                        generate_watts_strogatz(request.n, request.k, request.beta, request.graph_seed), edges="edges"),
                    file, indent=4)
    else:
        print(request)
        lock = threading.Lock()
        with lock:
            file_path = os.path.join("graphs", f"{request.graph_name}.json")
            with open(file_path, "w") as file:
                json.dump(
                    json_graph.node_link_data(
                        generate_barabasi_albert(request.n, request.m, request.graph_seed), edges="edges"),
                    file, indent=4)


@app.post("/api/simulation/user_graph")
def handle_user_graph(request: UserGraphRequest):
    lock = threading.Lock()
    with lock:
        file_path = os.path.join("graphs", f"{request.graph_name}.json")
        with open(file_path, "w") as file:
            file.write(request.graph_json)


@app.get("/api/simulation/get_heuristic_performance_data/")
def get_perf_data():
    binary_data = msgpack.packb(load_heuristic_data(), use_bin_type=True)
    # Return the binary response with appropriate content type
    return Response(content=binary_data, media_type="application/octet-stream")


@app.get("/api/simulation/list")
def get_simulation_list():
    in_memory_set = set([str(f) for f in simulation_cache.keys()])
    in_storage_list = [re.search(r'\d+', f).group() for f in os.listdir("storage")
                       if isfile(join("storage", f)) and "Simulation" in f]
    final_list = []
    id_list = []
    for f in in_storage_list:
        id_list.append(int(f))
        s = "Simulation " + f
        if f in in_memory_set:
            s = s + " (in memory)"
        final_list.append(s)
    data = {
        "list": final_list,
        "idList": id_list
    }

    # Serialize data to binary using msgpack
    binary_data = msgpack.packb(data, use_bin_type=True)

    # Return the binary response with appropriate content type
    return Response(content=binary_data, media_type="application/octet-stream")


# this may not work for 'expanded' node children...
@app.get("/api/simulation/{simulationID}/{nodeID}/{subgraphID}/{UC}/{timestep}/get_info")
def get_info(simulationID: int, nodeID: int, subgraphID: str, UC: int, timestep: int):
    # Get data object first
    if simulationID not in simulation_cache:
        data = load_simulation_data(s_dir, simulationID)
        if not data:
            raise HTTPException(status_code=404, detail="Simulation not found")
        simulation_cache[simulationID] = data
    else:
        data = simulation_cache[simulationID]

    graph_data = data["graph_data"]
    g = graph_data["uc_graph"]
    cluster_size = 0
    node_label = graph_data["local_id_to_label_dict"][str((nodeID, subgraphID))]
    if UC == 0:  # We read from the clustered graph
        g = graph_data["graph_dictionary"][subgraphID]
        # TODO (convert to nodeID then same...)
        # cluster_size = data["display_data"]["clusterSizeList"][nodeID]
        cluster_size = len(graph_data["reversed_cluster_map"][node_label])
        # Create the dictionary for the node. From ID, we need to send back a JSON with...
    # ID of all neighbours
    neighbour_id_list = [int(graph_data["node_label_to_list_index_dict"][str(label)]) for label in
                         g.neighbors(node_label)]
    # Label of all neighbours
    neighbour_label_list = [label for label in g.neighbors(node_label)]
    # Count number of active nodes
    no_active = -1
    if UC == 0:
        no_active = 0
        # reverse the cluster map
        reversed_cluster_map = graph_data["reversed_cluster_map"]
        cluster_contents = reversed_cluster_map[str(node_label)]

        for uc_node in cluster_contents:
            timestep_when_active = data["display_data"]["nodeToActiveTimestampList"][
                graph_data["node_label_to_list_index_dict"][uc_node]  # Get index corresponding to uc node
            ]
            if timestep >= timestep_when_active and timestep_when_active != -2:
                no_active += 1

    node_data = {
        "label": str(node_label),
        "neighbourLabelList": neighbour_label_list,
        "neighbourIndexList": neighbour_id_list,
        "clusterSize": cluster_size,
        "noActive": no_active
    }

    # Serialize data to binary using msgpack
    binary_data = msgpack.packb(node_data, use_bin_type=True)

    # Return the binary response with appropriate content type
    return Response(content=binary_data, media_type="application/octet-stream")


@app.get("/api/simulation/{simulationID}/display")
def get_display(simulationID: int):
    if simulationID not in simulation_cache:
        data = load_simulation_data(s_dir, simulationID)
        if not data:
            raise HTTPException(status_code=404, detail="Simulation not found")
        simulation_cache[simulationID] = data
    else:
        data = simulation_cache[simulationID]

    # Serialize data to binary using msgpack
    binary_data = msgpack.packb(data["display_data"], use_bin_type=True)

    # Return the binary response with appropriate content type
    return Response(content=binary_data, media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# Storage initialisation & server entry‑point
# ---------------------------------------------------------------------------
s_dir = initialize_storage()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
