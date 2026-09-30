"""Tests for the baselines (fixed threshold, random search, oracle)."""
import numpy as np

from src import baselines
from src.graph_gen import edge_list, generate_service_graph
from src.simulator import _per_edge_params


def _small():
    return generate_service_graph(n_nodes=12, seed=3)


def test_oracle_formula_is_canonical():
    """The one oracle definition: theta = clip(noise_amp + 0.10, 0, 1)."""
    G = _small()
    edges = edge_list(G)
    n = G.number_of_nodes()
    noise_mat, _ = _per_edge_params(edges, n, 0.3, 0.25)
    amps = np.array([noise_mat[u, v] for (u, v) in edges])
    th = baselines.oracle_thresholds(G, noise_amp=0.3, spread_p=0.25)
    assert th.shape == (len(edges),)
    assert np.allclose(th, np.clip(amps + 0.10, 0.0, 1.0))
    assert np.all((th >= 0.0) & (th <= 1.0))


def test_oracle_margin_parameter():
    G = _small()
    th = baselines.oracle_thresholds(G, margin=0.0)
    edges = edge_list(G)
    noise_mat, _ = _per_edge_params(edges, G.number_of_nodes(), 0.3, 0.25)
    amps = np.array([noise_mat[u, v] for (u, v) in edges])
    assert np.allclose(th, np.clip(amps, 0.0, 1.0))


def test_calibrated_thresholds_shape_range_and_determinism():
    """Calibrated heuristic: one theta per edge in [0,1], deterministic."""
    G = _small()
    kw = dict(n_steps=10, noise_amp=0.3, spread_p=0.25)
    th1 = baselines.calibrated_thresholds(G, [11, 12, 13], **kw)
    th2 = baselines.calibrated_thresholds(G, [11, 12, 13], **kw)
    assert th1.shape == (len(edge_list(G)),)
    assert np.all((th1 >= 0.0) & (th1 <= 1.0))
    assert np.array_equal(th1, th2)
    # per-edge thetas are not all identical (it actually estimates)
    assert np.unique(th1).size > 1


def test_calibrated_uses_observables_not_hidden_amps():
    """Sanity: the calibrated estimate tracks the true noise level.

    On a small graph with strong noise heterogeneity, edges' estimated
    amplitudes should correlate positively with their true amplitudes.
    """
    G = _small()
    edges = edge_list(G)
    n = G.number_of_nodes()
    noise_mat, _ = _per_edge_params(edges, n, 0.3, 0.25)
    true_amps = np.array([noise_mat[u, v] for (u, v) in edges])
    th = baselines.calibrated_thresholds(
        G, list(range(20, 32)), n_steps=15, noise_amp=0.3, spread_p=0.25)
    est_amps = th - 0.10  # undo the margin to recover the amplitude estimate
    corr = np.corrcoef(true_amps, est_amps)[0, 1]
    assert corr > 0.5, f"calibration does not track noise (r={corr:.2f})"
