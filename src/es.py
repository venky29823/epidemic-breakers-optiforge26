"""(1+1) evolution strategy baseline.

A deliberately simple, strong baseline: one parent, one Gaussian-mutated
child per step, keep the child when it is no worse on train F. Same
evaluation budget (1,230), mutation scale (sigma=0.08) and init range as
the GA, so any gap vs the GA reflects the algorithm, not the budget.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from src import fitness as fitmod


def run_es(graph: nx.DiGraph, train_seeds: list[int], budget: int, rng: np.random.Generator,
           sigma: float = 0.08, init_lo: float = 0.2, init_hi: float = 0.7,
           **sim_kwargs) -> dict:
    """Run a (1+1)-ES for ``budget`` fitness evaluations.

    Returns dict(best, best_F, evals). Deterministic given ``rng``.
    """
    n_edges = graph.number_of_edges()
    parent = rng.uniform(init_lo, init_hi, size=n_edges)
    parent_F = fitmod.evaluate(graph, parent, train_seeds, **sim_kwargs)["F"]
    evals = 1
    best = parent.copy()
    best_F = float(parent_F)
    while evals < budget:
        child = np.clip(parent + rng.normal(0.0, sigma, size=n_edges), 0.0, 1.0)
        child_F = fitmod.evaluate(graph, child, train_seeds, **sim_kwargs)["F"]
        evals += 1
        if child_F <= parent_F:  # (1+1) selection: keep if not worse
            parent, parent_F = child, child_F
            if child_F < best_F:
                best = child.copy()
                best_F = float(child_F)
    return {"best": best, "best_F": best_F, "evals": evals}
