"""Objective function: mean scenario cost of a threshold vector.

F = cascade_size + w_false_trips * false_trips + w_latency * latency_penalty
averaged over a list of scenario seeds. Lower is better.

The weights default to the study values (2.0 and 0.1) but are explicit
parameters so hidden shifts can change the objective without code edits.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .simulator import simulate_many

W_FALSE_TRIPS = 2.0
W_LATENCY = 0.1


def scenario_cost(
    metrics: dict,
    w_false_trips: float = W_FALSE_TRIPS,
    w_latency: float = W_LATENCY,
) -> float:
    """Cost of a single simulated scenario."""
    return (
        metrics["cascade_size"]
        + w_false_trips * metrics["false_trips"]
        + w_latency * metrics["latency_penalty"]
    )


def evaluate(
    graph: nx.DiGraph,
    theta: np.ndarray,
    seeds: list[int],
    w_false_trips: float = W_FALSE_TRIPS,
    w_latency: float = W_LATENCY,
    **sim_kwargs,
) -> dict:
    """Evaluate ``theta`` over ``seeds``; return mean F and mean components.

    ``w_false_trips`` / ``w_latency`` reweight the objective; everything in
    ``sim_kwargs`` is forwarded to the simulator (they must not collide).
    """
    results = simulate_many(graph, theta, seeds, **sim_kwargs)
    costs = np.array(
        [scenario_cost(r, w_false_trips, w_latency) for r in results]
    )
    return {
        "F": float(costs.mean()),
        "F_std": float(costs.std(ddof=1)),  # sample sd over scenarios
        "cascade_size": float(np.mean([r["cascade_size"] for r in results])),
        "false_trips": float(np.mean([r["false_trips"] for r in results])),
        "latency_penalty": float(np.mean([r["latency_penalty"] for r in results])),
        "n_seeds": len(seeds),
    }
