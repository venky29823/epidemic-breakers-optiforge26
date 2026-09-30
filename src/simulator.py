"""Discrete-step cascade simulator with circuit breakers.

Model
-----
* Each step, a slow service infects each of its *callers* (predecessors in the
  call graph) with a per-edge probability ``p_e``, unless the breaker on that
  edge is open.
* Slowness is absorbing within a scenario (no recovery): cascades only grow.
* Each edge carries an exponential moving average of "callee is slow" stress.
  The breaker observes ``stress + noise`` where the noise amplitude is
  *per-edge*; if the observation exceeds the edge threshold ``theta``, the
  breaker trips and stays open for ``cooldown`` steps (including the trip
  step).
* A *false trip* is a trip on an edge whose callee was healthy at trip time:
  the breaker reacted to noise, not to a real failure.
* ``latency_penalty`` is the percentage (0-100) of edge-steps spent with the
  breaker open: open breakers force fail-fast / rerouted traffic.

Heterogeneity
-------------
Edges differ deterministically per graph (derived from the edge list, not
from scenario seeds): each edge gets its own noise amplitude in
``[0.2, 1.0] * noise_amp`` and its own spread probability in
``[0.3, 1.7] * spread_p`` (clipped to [0.01, 0.95]). Because noisy edges need
high thresholds and quiet edges reward low ones, no single fixed threshold
can be optimal -- per-edge tuning genuinely matters. Pass
``heterogeneous=False`` for uniform edges (useful for debugging).

Pass ``theta >= 2.0`` on every edge to disable breakers entirely (the
observed signal never exceeds 1 + noise amplitude).
"""
from __future__ import annotations

import hashlib

import networkx as nx
import numpy as np


def _matrices(graph: nx.DiGraph):
    """Adjacency matrix and canonical edge list. Nodes must be 0..n-1."""
    n = graph.number_of_nodes()
    nodes = sorted(graph.nodes())
    if nodes != list(range(n)):
        raise ValueError("graph nodes must be labelled 0..n-1")
    A = nx.to_numpy_array(graph, dtype=bool, nodelist=nodes)
    return A, list(graph.edges())


def _per_edge_params(edges: list[tuple[int, int]], n: int,
                     noise_amp: float, spread_p: float):
    """Deterministic per-edge noise amplitudes and spread probabilities.

    Seeded by a stable hash of the edge list, so the environment is fixed for
    a given graph and identical across processes/runs.
    """
    h = hashlib.sha256(repr(sorted(edges)).encode()).hexdigest()
    rng = np.random.default_rng(int(h[:8], 16))
    noise_mat = np.zeros((n, n))
    p_mat = np.zeros((n, n))
    for (u, v) in edges:
        noise_mat[u, v] = noise_amp * (0.2 + 0.8 * rng.random())
        p_mat[u, v] = float(np.clip(spread_p * (0.3 + 1.4 * rng.random()), 0.01, 0.95))
    return noise_mat, p_mat


def _run(
    A: np.ndarray,
    edges: list[tuple[int, int]],
    theta: np.ndarray,
    noise_mat: np.ndarray,
    p_mat: np.ndarray,
    seed: int,
    cooldown: int,
    n_steps: int,
    ema_alpha: float,
    record_history: bool = False,
    record_obs: bool = False,
) -> dict:
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    theta_mat = np.zeros((n, n))
    for (u, v), t in zip(edges, theta):
        theta_mat[u, v] = t

    slow = np.zeros(n, dtype=bool)
    slow[int(rng.integers(n))] = True  # inject the initial failure

    ema = np.zeros((n, n))  # per-edge stress estimate
    open_timer = np.zeros((n, n), dtype=int)
    false_trips = 0
    open_edge_steps = 0
    n_edges = int(A.sum())
    slow_history: list[int] = []
    obs_history: list[np.ndarray] = []
    slow_vec_history: list[np.ndarray] = []

    for _ in range(n_steps):
        # 1. spread along closed edges: callers of slow callees, per-edge p
        A_eff = A & (open_timer == 0)
        infect = A_eff & slow[np.newaxis, :] & (rng.random((n, n)) < p_mat)
        slow |= infect.any(axis=1) & ~slow

        # 2. update per-edge stress (1 where the callee is slow)
        stress = A & slow[np.newaxis, :]
        ema = (1.0 - ema_alpha) * ema + ema_alpha * stress

        # 3. noisy observation -> trips
        observed = ema + rng.uniform(-1.0, 1.0, size=(n, n)) * noise_mat
        trips = A & (open_timer == 0) & (observed > theta_mat)
        false_trips += int(np.sum(trips & ~slow[np.newaxis, :]))
        open_timer[trips] = cooldown

        # 4. account open time, then tick timers down
        open_edge_steps += int(np.sum(open_timer > 0))
        open_timer[open_timer > 0] -= 1
        if record_history:
            slow_history.append(int(slow.sum()))
        if record_obs:
            obs_history.append(observed.copy())
            slow_vec_history.append(slow.copy())

    cascade_size = int(slow.sum())
    latency_penalty = 100.0 * open_edge_steps / max(1, n_edges * n_steps)
    out = {
        "cascade_size": cascade_size,
        "false_trips": false_trips,
        "latency_penalty": float(latency_penalty),
        "n_steps": n_steps,
        "n_edges": n_edges,
    }
    if record_history:
        out["slow_history"] = slow_history
    if record_obs:
        out["obs_history"] = obs_history
        out["slow_vec_history"] = slow_vec_history
    return out


def _prepare(graph: nx.DiGraph, theta, noise_amp: float, spread_p: float,
             heterogeneous: bool):
    A, edges = _matrices(graph)
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (len(edges),):
        raise ValueError(
            f"theta has {theta.shape[0]} values but the graph has {len(edges)} edges"
        )
    n = A.shape[0]
    if heterogeneous:
        noise_mat, p_mat = _per_edge_params(edges, n, noise_amp, spread_p)
    else:
        noise_mat = np.zeros((n, n))
        p_mat = np.zeros((n, n))
        for (u, v) in edges:
            noise_mat[u, v] = noise_amp
            p_mat[u, v] = spread_p
    return A, edges, theta, noise_mat, p_mat


def simulate(
    graph: nx.DiGraph,
    theta: np.ndarray,
    seed: int,
    spread_p: float = 0.25,
    cooldown: int = 5,
    n_steps: int = 40,
    noise_amp: float = 0.3,
    ema_alpha: float = 0.3,
    heterogeneous: bool = True,
    record_history: bool = False,
    record_obs: bool = False,
) -> dict:
    """Run one failure scenario. ``theta`` has one value per graph edge.

    With ``record_history=True`` the returned dict also carries
    ``slow_history``: the number of slow services after each step.
    With ``record_obs=True`` it also carries ``obs_history`` (the noisy
    per-edge observation matrix per step) and ``slow_vec_history`` (the
    per-service slow vector per step) -- used to calibrate heuristics
    from observables without touching hidden per-edge parameters.
    """
    A, edges, theta, noise_mat, p_mat = _prepare(
        graph, theta, noise_amp, spread_p, heterogeneous
    )
    return _run(A, edges, theta, noise_mat, p_mat, seed,
                cooldown, n_steps, ema_alpha, record_history, record_obs)


def simulate_many(
    graph: nx.DiGraph,
    theta: np.ndarray,
    seeds: list[int],
    **kwargs,
) -> list[dict]:
    """Run many scenarios, reusing matrices. Deterministic given the seeds."""
    heterogeneous = kwargs.get("heterogeneous", True)
    A, edges, theta, noise_mat, p_mat = _prepare(
        graph, theta, kwargs.get("noise_amp", 0.3),
        kwargs.get("spread_p", 0.25), heterogeneous,
    )
    cooldown = kwargs.get("cooldown", 5)
    n_steps = kwargs.get("n_steps", 40)
    ema_alpha = kwargs.get("ema_alpha", 0.3)
    return [
        _run(A, edges, theta, noise_mat, p_mat, int(s),
             cooldown, n_steps, ema_alpha)
        for s in seeds
    ]
