"""Tests for warm-starting across hidden shifts (graph / objective changes)."""
import numpy as np

from src import ga
from src.graph_gen import edge_list, generate_service_graph
from src.utils import make_rng, make_seeds


def _tiny():
    G = generate_service_graph(n_nodes=10, seed=11)
    seeds = make_seeds(3, base=500)
    return G, seeds


def _cfg(**kw):
    base = {"pop_size": 8, "generations": 4, "mutation_mode": "guided"}
    base.update(kw)
    return ga.GAConfig(**base)


def _trained(G, seeds):
    return ga.run_ga(G, seeds, _cfg(), make_rng(7))


def _valid_theta(best: np.ndarray, e_new: int):
    assert best.shape == (e_new,), f"shape {best.shape} != ({e_new},)"
    assert np.all(best >= 0.0) and np.all(best <= 1.0), "theta out of [0,1]"


def test_warm_start_edge_removed():
    G, seeds = _tiny()
    r1 = _trained(G, seeds)
    old_edges = edge_list(G)
    G2 = G.copy()
    G2.remove_edge(*old_edges[0])
    e_new = G2.number_of_edges()
    assert e_new == len(old_edges) - 1

    res = ga.run_ga(G2, seeds, _cfg(), make_rng(9),
                    init_pop=r1["final_pop"], init_edges=old_edges)
    _valid_theta(res["best"], e_new)

    # identity mapping: surviving edges keep their learned column
    remapped = ga.remap_population(r1["final_pop"], old_edges, G2, make_rng(11))
    assert remapped.shape == (8, e_new)
    new_edges = edge_list(G2)
    j = new_edges.index(old_edges[1])
    assert np.allclose(remapped[:, j], r1["final_pop"][:, 1])


def test_warm_start_edge_added():
    G, seeds = _tiny()
    r1 = _trained(G, seeds)
    old_edges = edge_list(G)
    G2 = G.copy()
    pair = next((u, v) for u in range(10) for v in range(10)
                if u != v and not G2.has_edge(u, v))
    G2.add_edge(*pair)
    e_new = G2.number_of_edges()
    assert e_new == len(old_edges) + 1

    res = ga.run_ga(G2, seeds, _cfg(), make_rng(9),
                    init_pop=r1["final_pop"], init_edges=old_edges)
    _valid_theta(res["best"], e_new)

    # the new edge is initialised from the shared init prior U[0.2, 0.7]
    remapped = ga.remap_population(r1["final_pop"], old_edges, G2, make_rng(11))
    j = edge_list(G2).index(pair)
    assert np.all(remapped[:, j] >= 0.2) and np.all(remapped[:, j] <= 0.7)


def test_warm_start_changed_weights():
    G, seeds = _tiny()
    r1 = _trained(G, seeds)
    res = ga.run_ga(G, seeds, _cfg(), make_rng(9),
                    init_pop=r1["final_pop"],
                    w_false_trips=5.0, w_latency=0.5)
    _valid_theta(res["best"], G.number_of_edges())
    # the weights must actually reweight the objective (exact arithmetic)
    from src import fitness as fitmod
    m = {"cascade_size": 10, "false_trips": 3, "latency_penalty": 50.0}
    assert fitmod.scenario_cost(m) == 10 + 2 * 3 + 0.1 * 50
    assert fitmod.scenario_cost(m, w_false_trips=5.0, w_latency=0.5) == \
        10 + 5 * 3 + 0.5 * 50
