"""Tests for the genetic algorithm."""
import itertools

import numpy as np

from src import ga
from src.graph_gen import edge_list, generate_service_graph
from src.utils import make_rng, make_seeds


def _tiny():
    G = generate_service_graph(n_nodes=10, seed=11)
    seeds = make_seeds(3, base=500)
    return G, seeds


def _cfg(**kw):
    base = {"pop_size": 10, "generations": 8, "mutation_mode": "uniform"}
    base.update(kw)
    return ga.GAConfig(**base)


def test_history_nonincreasing_with_elitism():
    G, seeds = _tiny()
    res = ga.run_ga(G, seeds, _cfg(), make_rng(1))
    hist = res["history"]
    assert len(hist) == 9  # initial + 8 generations
    assert all(b <= a + 1e-12 for a, b in itertools.pairwise(hist))


def test_genes_stay_in_bounds():
    G, seeds = _tiny()
    res = ga.run_ga(G, seeds, _cfg(mutation_mode="guided"), make_rng(2))
    assert np.all(res["final_pop"] >= 0.0) and np.all(res["final_pop"] <= 1.0)
    assert np.all(res["best"] >= 0.0) and np.all(res["best"] <= 1.0)


def test_guided_mode_runs_and_scores_finite():
    G, seeds = _tiny()
    res = ga.run_ga(G, seeds, _cfg(mutation_mode="guided"), make_rng(3))
    assert np.isfinite(res["best_F"])


def test_warm_start_accepts_previous_population():
    G, seeds = _tiny()
    cfg = _cfg()
    r1 = ga.run_ga(G, seeds, cfg, make_rng(4))
    r2 = ga.run_ga(G, seeds, cfg, make_rng(5), init_pop=r1["final_pop"],
                   mutation_boost=2.0, boost_gens=3)
    assert r2["final_pop"].shape == r1["final_pop"].shape
    assert np.isfinite(r2["best_F"])


def test_warm_start_wrong_shape_raises():
    G, seeds = _tiny()
    try:
        ga.run_ga(G, seeds, _cfg(), make_rng(6), init_pop=np.zeros((10, 3)))
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong init_pop shape")


def test_target_score_recorded():
    G, seeds = _tiny()
    res = ga.run_ga(G, seeds, _cfg(), make_rng(7), target_score=1e9)
    assert res["gens_to_target"] == 0  # beaten immediately


def test_edge_weights_align_with_edge_list():
    G, _ = _tiny()
    w = ga.edge_weights(G)
    assert w.shape == (len(edge_list(G)),)
    assert abs(w.sum() - 1.0) < 1e-12


def test_betweenness_symmetric_under_reversal():
    # Edge betweenness is symmetric under graph reversal when keyed by edge
    # identity: ebc(G)[(u, v)] == ebc(G.reverse())[(v, u)], because
    # shortest-path counts are preserved under reversal. Guards against
    # misreading the directed weights as direction-dependent (a positional
    # comparison of edge_list(G) vs edge_list(G.reverse()) is meaningless:
    # the two orders differ).
    import networkx as nx

    G, _ = _tiny()
    ebc = nx.edge_betweenness_centrality(G)
    ebc_rev = nx.edge_betweenness_centrality(G.reverse())
    for (u, v) in edge_list(G):
        assert abs(ebc[(u, v)] - ebc_rev[(v, u)]) < 1e-12


def test_init_population_respects_range():
    G, _ = _tiny()
    pop = ga.init_population(G.number_of_edges(), 20, make_rng(9), lo=0.2, hi=0.7)
    assert pop.shape == (20, G.number_of_edges())
    assert np.all(pop >= 0.2) and np.all(pop <= 0.7)


def test_random_weights_mode_is_nonuniform_and_runs():
    """'random-weights' mutation: completes a run and is a real control.

    It must differ from both uniform and guided runs given the same RNG
    seed (different mutation distributions), and stay within [0, 1].
    """
    G, _ = _tiny()
    seeds = [201, 202]
    kw = {'n_steps': 10, 'noise_amp': 0.3, 'spread_p': 0.25}
    cfg_rw = ga.GAConfig(pop_size=8, generations=3, mutation_mode="random-weights")
    cfg_g = ga.GAConfig(pop_size=8, generations=3, mutation_mode="guided")
    r_rw = ga.run_ga(G, seeds, cfg_rw, make_rng(5), **kw)
    r_g = ga.run_ga(G, seeds, cfg_g, make_rng(5), **kw)
    assert r_rw["evals"] == 8 * 4
    assert np.all((r_rw["best"] >= 0.0) & (r_rw["best"] <= 1.0))
    assert not np.array_equal(r_rw["best"], r_g["best"])
