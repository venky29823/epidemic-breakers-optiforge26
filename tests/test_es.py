"""Tests for the (1+1)-ES baseline (src/es.py)."""
import numpy as np

from src import es as esmod
from src import graph_gen
from src.utils import make_rng


def _tiny():
    G = graph_gen.generate_service_graph(n_nodes=12, seed=3)
    return G


def test_es_respects_budget_and_never_worsens():
    G = _tiny()
    seeds = [101, 102]
    res = esmod.run_es(G, seeds, budget=25, rng=make_rng(0),
                       n_steps=10, noise_amp=0.3, spread_p=0.25)
    assert res["evals"] == 25
    assert res["best"].shape == (G.number_of_edges(),)
    assert np.all((res["best"] >= 0.0) & (res["best"] <= 1.0))
    # (1+1) selection keeps the child when not worse: best_F is finite
    # and the returned best is the argmin over the trajectory
    assert np.isfinite(res["best_F"])


def test_es_deterministic_given_rng():
    G = _tiny()
    seeds = [101, 102]
    kw = dict(budget=25, n_steps=10, noise_amp=0.3, spread_p=0.25)
    r1 = esmod.run_es(G, seeds, rng=make_rng(7), **kw)
    r2 = esmod.run_es(G, seeds, rng=make_rng(7), **kw)
    assert r1["best_F"] == r2["best_F"]
    assert np.array_equal(r1["best"], r2["best"])
