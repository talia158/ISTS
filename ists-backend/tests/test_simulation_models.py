"""
Regression-tests for the influence spread simulation used by the backend.
"""
import random
import networkx as nx
import pytest

from main import independent_cascade_model, linear_threshold_model


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_line_graph(n: int, *, weight: float = 1.0, threshold: float = 0.0):
    """Pₙ with identical weights / thresholds so we can reason analytically."""
    g = nx.path_graph(n)
    for u, v in g.edges():
        g[u][v]["weight"] = weight
        g[v][u]["weight"] = weight
    for u in g.nodes():
        g.nodes[u]["threshold"] = threshold
    return g


def assert_monotone(timestamps: dict[str, int]):
    """Every node appears once and at a non-negative time."""
    assert len(timestamps) == len(set(timestamps))          # unique keys
    assert all(t >= 0 for t in timestamps.values())         # no negatives


# ---------------------------------------------------------------------------
#  Independent Cascade
# ---------------------------------------------------------------------------
def test_ic_full_activation_weight_one():
    """
    On a 5-node line with weight=1 the influence should reach everyone.
    The implementation *always* reports one extra trailing timestep.
    """
    g = make_line_graph(5, weight=1.0)
    random.seed(1)
    act, steps, _ = independent_cascade_model(g, ["0"])

    assert len(act) == g.number_of_nodes()                  # all nodes fire
    assert steps == g.number_of_nodes() + 1                 # ripple +1
    assert_monotone(act)


def test_ic_no_spread_weight_zero():
    """
    With weight=0 nothing propagates.  We still expect the extra +1 step
    (seed processing + trailing empty iteration).
    """
    g = make_line_graph(7, weight=0.0)
    random.seed(2024)
    act, steps, _ = independent_cascade_model(g, ["3"])

    assert act == {"3": 0}
    assert steps == 2                                        # 1 + trailing
    assert_monotone(act)


# ---------------------------------------------------------------------------
#  Linear Threshold
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "threshold, expect_all",
    [
        (0.0, True),     # trivially satisfied
        (1.1, False),    # unreachable (weights ≤ 1)
    ],
)
def test_ltm_extreme_thresholds(threshold, expect_all):
    g = make_line_graph(4, weight=1.0, threshold=threshold)
    act, steps, _ = linear_threshold_model(g, ["1", "2"])

    if expect_all:
        assert len(act) == g.number_of_nodes()
    else:
        assert set(act) == {"1", "2"}
    assert steps >= 2                                        # at least seed+1
    assert_monotone(act)


def test_ltm_progressive_activation():
    """
    threshold=0.5 on a 6-node line: each neighbour triggers the next,
    finishing in len(nodes)+1 timesteps (same extra +1 as IC).
    """
    g = make_line_graph(6, weight=1.0, threshold=0.5)
    act, steps, _ = linear_threshold_model(g, ["0"])

    assert len(act) == g.number_of_nodes()
    assert steps == g.number_of_nodes() + 1
    assert_monotone(act)
