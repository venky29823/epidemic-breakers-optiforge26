"""Genetic algorithm over circuit-breaker thresholds.

Individual: real vector theta in [0, 1]^E (one threshold per graph edge).
Minimises train fitness F.

Two mutation modes:
  * "uniform" - mutated genes chosen uniformly at random (vanilla GA);
  * "guided"  - mutated genes chosen with probability proportional to edge
                betweenness centrality (immunisation-inspired: protect the
                super-spreader edges first).

Supports warm-starting from a previous population and a temporary
mutation-rate boost (used for Round 2 re-adaptation).
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from . import fitness as fitmod
from .graph_gen import edge_list


@dataclass
class GAConfig:
    """Genetic algorithm hyperparameters.

    Attributes:
        pop_size: Number of individuals per generation.
        generations: Number of generations to evolve.
        tournament_k: Tournament size for parent selection.
        crossover_rate: Probability of crossover per pair.
        mutation_rate: Per-gene mutation probability.
        mutation_sigma: Std dev of Gaussian mutation noise.
        elitism: Number of best individuals carried over unchanged.
        mutation_mode: "uniform" (each gene equally likely) or "guided"
            (genes sampled proportional to edge-betweenness weights).
    """
    pop_size: int = 30
    generations: int = 40
    tournament_k: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.05
    mutation_sigma: float = 0.08
    elitism: int = 2
    mutation_mode: str = "uniform"  # "uniform" | "guided"
    init_lo: float = 0.2
    init_hi: float = 0.7


def edge_weights(graph: nx.DiGraph, uniform_blend: float = 0.5) -> np.ndarray:
    """Per-edge mutation weights: betweenness blended with uniform.

    Pure betweenness weights starve low-betweenness edges of mutations, so
    their thresholds never get tuned. Blending with uniform
    (``uniform_blend=0.5``) biases search toward structurally important
    edges without abandoning the rest. Aligned with ``edge_list(graph)``
    order; falls back to uniform if every betweenness value is zero.
    """
    edges = edge_list(graph)
    btw = nx.edge_betweenness_centrality(graph)
    w = np.array([btw.get(e, 0.0) for e in edges], dtype=float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    u = np.ones_like(w) / len(w)
    return (1.0 - uniform_blend) * w + uniform_blend * u


def init_population(
    n_edges: int,
    pop_size: int,
    rng: np.random.Generator,
    lo: float = 0.2,
    hi: float = 0.7,
) -> np.ndarray:
    """Initial population, uniform in [lo, hi].

    Informed prior, not a trick: thresholds below the noise floor (~0.3 max
    per-edge amplitude) false-trip constantly, and thresholds near 1.0 react
    too slowly to matter. Useful thresholds live in the mid-range, so that
    is where search starts. Both GA variants and random search share it.
    """
    return rng.uniform(lo, hi, size=(pop_size, n_edges))


def remap_population(
    old_pop: np.ndarray,
    old_edges: list[tuple[int, int]],
    new_graph: nx.DiGraph,
    rng: np.random.Generator,
    init_lo: float = 0.2,
    init_hi: float = 0.7,
) -> np.ndarray:
    """Remap a saved population onto a new graph's edge set.

    Thresholds follow edge *identity*: an edge ``(u, v)`` that still exists
    keeps its learned threshold column. Edges added by the shift are
    initialised from the shared init prior ``U[init_lo, init_hi]``;
    removed edges are dropped. Returns an array in
    ``[0, 1]^(pop_size x E_new)`` aligned with ``edge_list(new_graph)``.
    """
    old_pop = np.asarray(old_pop, dtype=float)
    if old_pop.ndim != 2 or old_pop.shape[1] != len(old_edges):
        raise ValueError(
            f"old_pop has shape {old_pop.shape} but old_edges has "
            f"{len(old_edges)} edges"
        )
    new_edges = edge_list(new_graph)
    old_index = {e: i for i, e in enumerate(old_edges)}
    pop_size = old_pop.shape[0]
    new_pop = np.empty((pop_size, len(new_edges)), dtype=float)
    for j, e in enumerate(new_edges):
        if e in old_index:
            new_pop[:, j] = old_pop[:, old_index[e]]
        else:
            new_pop[:, j] = rng.uniform(init_lo, init_hi, size=pop_size)
    return np.clip(new_pop, 0.0, 1.0)


def tournament_select(
    pop: np.ndarray, fitnesses: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Pick the best (lowest F) of k random individuals."""
    idx = rng.choice(len(pop), size=k, replace=False)
    return pop[idx[np.argmin(fitnesses[idx])]].copy()


def uniform_crossover(
    p1: np.ndarray, p2: np.ndarray, rate: float, rng: np.random.Generator
) -> np.ndarray:
    """Per-gene uniform crossover with probability ``rate``."""
    if rng.random() < rate:
        mask = rng.random(len(p1)) < 0.5
        return np.where(mask, p1, p2)
    return p1.copy()


def mutate(
    ind: np.ndarray,
    rng: np.random.Generator,
    rate: float,
    sigma: float,
    mode: str,
    weights: np.ndarray | None,
) -> np.ndarray:
    """Gaussian mutation, clipped to [0, 1].

    Uniform mode mutates each gene independently with probability ``rate``.
    Guided mode mutates ``Binomial(n, rate)`` genes chosen with probability
    proportional to ``weights`` (edge betweenness).
    """
    child = ind.copy()
    n = len(ind)
    if mode in ("guided", "random-weights") and weights is not None:
        k = int(rng.binomial(n, rate))
        idx = rng.choice(n, size=min(max(k, 1), n), replace=False, p=weights)
    else:
        idx = np.flatnonzero(rng.random(n) < rate)
    if len(idx):
        child[idx] += rng.normal(0.0, sigma, size=len(idx))
    return np.clip(child, 0.0, 1.0)


def run_ga(
    graph: nx.DiGraph,
    train_seeds: list[int],
    config: GAConfig,
    rng: np.random.Generator,
    init_pop: np.ndarray | None = None,
    init_edges: list[tuple[int, int]] | None = None,
    mutation_boost: float = 1.0,
    boost_gens: int = 0,
    target_score: float | None = None,
    w_false_trips: float = fitmod.W_FALSE_TRIPS,
    w_latency: float = fitmod.W_LATENCY,
    **sim_kwargs,
) -> dict:
    """Run the GA. Returns best individual, history, and warm-start state.

    ``init_pop`` warm-starts from a previous population. If ``init_edges``
    (the edge list the population was trained on) is given and differs
    from the current graph's edges -- e.g. a hidden shift added or removed
    edges -- the population is remapped by edge identity via
    :func:`remap_population` instead of raising. Without ``init_edges`` a
    shape mismatch still raises, since silent truncation would hide bugs.

    ``mutation_boost`` multiplies the mutation rate for the first
    ``boost_gens`` generations. ``target_score`` records the first generation
    whose best train F reaches it (generations-to-target).
    ``w_false_trips`` / ``w_latency`` reweight the objective.
    """
    n_edges = graph.number_of_edges()
    if config.mutation_mode == "guided":
        weights = edge_weights(graph)
    elif config.mutation_mode == "random-weights":
        # Control: non-uniform mutation weights with NO structural signal.
        # If guided beats this, the win comes from the betweenness prior,
        # not from non-uniformity alone.
        w = rng.random(n_edges)
        weights = w / w.sum()
    elif config.mutation_mode == "uniform":
        weights = None
    else:
        raise ValueError(f"unknown mutation_mode: {config.mutation_mode!r}")

    if init_pop is not None:
        if init_edges is not None and list(init_edges) != edge_list(graph):
            pop = remap_population(init_pop, list(init_edges), graph, rng,
                                   config.init_lo, config.init_hi)
        elif np.shape(init_pop) != (config.pop_size, n_edges):
            raise ValueError(
                "init_pop has wrong shape for warm start; pass init_edges "
                "to remap a population across a graph shift"
            )
        else:
            pop = np.clip(np.asarray(init_pop, dtype=float).copy(), 0.0, 1.0)
    else:
        pop = init_population(n_edges, config.pop_size, rng,
                              config.init_lo, config.init_hi)

    def batch_fitness(p: np.ndarray) -> np.ndarray:
        """Evaluate a population's fitness on train seeds.

        Args:
            p: Population array, shape (pop_size, n_edges), thresholds in [0, 1].

        Returns:
            Array of mean-F values, one per individual (lower is better).
        """
        return np.array(
            [fitmod.evaluate(graph, ind, train_seeds,
                             w_false_trips=w_false_trips, w_latency=w_latency,
                             **sim_kwargs)["F"] for ind in p]
        )

    fitnesses = batch_fitness(pop)
    evals = len(pop)
    best_idx = int(np.argmin(fitnesses))
    best, best_f = pop[best_idx].copy(), float(fitnesses[best_idx])
    history = [best_f]
    best_per_gen = [best.copy()]  # running best individual after each generation
    gens_to_target = 0 if target_score is not None and best_f <= target_score else None

    for g in range(config.generations):
        rate = config.mutation_rate * (mutation_boost if g < boost_gens else 1.0)
        order = np.argsort(fitnesses)
        new_pop = [pop[i].copy() for i in order[: config.elitism]]
        while len(new_pop) < config.pop_size:
            p1 = tournament_select(pop, fitnesses, config.tournament_k, rng)
            p2 = tournament_select(pop, fitnesses, config.tournament_k, rng)
            child = uniform_crossover(p1, p2, config.crossover_rate, rng)
            child = mutate(child, rng, rate, config.mutation_sigma,
                           config.mutation_mode, weights)
            new_pop.append(child)
        pop = np.array(new_pop)
        fitnesses = batch_fitness(pop)
        evals += len(pop)
        gen_best = float(fitnesses.min())
        if gen_best < best_f:
            best_f = gen_best
            best = pop[int(np.argmin(fitnesses))].copy()
        history.append(best_f)
        best_per_gen.append(best.copy())
        if (
            target_score is not None
            and gens_to_target is None
            and best_f <= target_score
        ):
            gens_to_target = g + 1

    return {
        "best": best,
        "best_F": best_f,
        "history": history,
        "best_per_gen": best_per_gen,
        "gens_to_target": gens_to_target,
        "final_pop": pop.copy(),
        "evals": evals,
    }
