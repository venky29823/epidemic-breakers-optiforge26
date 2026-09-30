"""Regression pins for default objective / simulator behaviour.

The pinned values were captured 2026-09-30 from the pre-fix baseline
(tag ``pre-fix-baseline``, commit 1756a16) in a clean worktree, by
evaluating with all-default arguments on 4 test seeds (the first 4 of
``make_seeds(24, base=2000)`` -- the Round-1 test-seed scheme):

    PINNED_FIXED_F    = 42.91103603603604    # theta_e = 0.5 for all edges

``PINNED_GA_THETA_F`` was re-pinned when ``results/best_theta.npy`` was
regenerated with provenance by ``scripts/make_best_theta.py`` (see
``results/best_theta.json`` for seeds, config and git SHA); it is the
new theta's F on the same 4 test seeds:

    PINNED_GA_THETA_F = 33.0106981981982    # regenerated 2026-09-30

If either test fails, a default changed silently (objective weights,
simulator constants, seed scheme, or theta/edge ordering). Update the
pins only after confirming the change is intended.
"""

from pathlib import Path

import numpy as np
import pytest

from src import fitness as fit
from src import graph_gen
from src.utils import make_seeds

REPO = Path(__file__).resolve().parent.parent

PINNED_FIXED_F = 42.91103603603604
PINNED_GA_THETA_F = 33.0106981981982


def _study_setup():
    G = graph_gen.generate_service_graph(n_nodes=40, seed=7)
    seeds = make_seeds(4, base=2000)
    return G, seeds


def test_default_objective_fixed_threshold():
    G, seeds = _study_setup()
    theta = np.full(G.number_of_edges(), 0.5)
    r = fit.evaluate(G, theta, seeds)  # all defaults
    assert r["F"] == pytest.approx(PINNED_FIXED_F, abs=1e-9)


def test_default_objective_saved_ga_theta():
    G, seeds = _study_setup()
    theta = np.load(REPO / "results" / "best_theta.npy")
    assert theta.shape == (G.number_of_edges(),)
    r = fit.evaluate(G, theta, seeds)  # all defaults
    assert r["F"] == pytest.approx(PINNED_GA_THETA_F, abs=1e-9)
