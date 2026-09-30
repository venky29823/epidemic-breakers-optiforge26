"""Regenerate the precomputed GA-optimized threshold vector.

Runs the guided GA with the study configuration on the documented train
seeds, saves ``best_theta.npy``, and writes a ``best_theta.json`` sidecar
logging every seed, the full GA config, the simulator settings, the git
revision, and the resulting train/test scores -- so the file's provenance
is recoverable from the repo alone.

Usage: python3 scripts/make_best_theta.py [--outdir results] [--seed 42]
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from main import ROUND1_KW  # noqa: E402
from src import fitness as fitmod  # noqa: E402
from src import ga as gamod  # noqa: E402
from src import graph_gen  # noqa: E402
from src.graph_gen import edge_list  # noqa: E402
from src.utils import ensure_dir, make_rng, make_seeds  # noqa: E402

TRAIN_N, TRAIN_BASE = 24, 1000
TEST_N, TEST_BASE = 12, 2000
GRAPH_SEED, N_NODES = 7, 40


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seed", type=int, default=42,
                    help="master RNG seed for the GA run")
    args = ap.parse_args()

    G = graph_gen.generate_service_graph(n_nodes=N_NODES, seed=GRAPH_SEED)
    edges = edge_list(G)
    train_seeds = make_seeds(TRAIN_N, base=TRAIN_BASE)
    test_seeds = make_seeds(TEST_N, base=TEST_BASE)
    assert not set(train_seeds) & set(test_seeds)

    cfg = gamod.GAConfig(pop_size=30, generations=40, mutation_mode="guided")
    rng = make_rng(args.seed)
    res = gamod.run_ga(G, train_seeds, cfg, rng, **ROUND1_KW)
    test = fitmod.evaluate(G, res["best"], test_seeds, **ROUND1_KW)

    outdir = ensure_dir(REPO / args.outdir)
    np.save(outdir / "best_theta.npy", res["best"])
    sidecar = {
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "script": "scripts/make_best_theta.py",
        "graph": {"n_nodes": N_NODES, "seed": GRAPH_SEED,
                  "n_edges": len(edges)},
        "master_seed": args.seed,
        "train_seeds": {"n": TRAIN_N, "base": TRAIN_BASE,
                        "seeds": train_seeds},
        "test_seeds": {"n": TEST_N, "base": TEST_BASE,
                       "seeds": test_seeds},
        "ga_config": {
            "pop_size": cfg.pop_size, "generations": cfg.generations,
            "tournament_k": cfg.tournament_k,
            "crossover_rate": cfg.crossover_rate,
            "mutation_rate": cfg.mutation_rate,
            "mutation_sigma": cfg.mutation_sigma,
            "elitism": cfg.elitism,
            "mutation_mode": cfg.mutation_mode,
            "init_lo": cfg.init_lo, "init_hi": cfg.init_hi,
        },
        "sim": dict(ROUND1_KW),
        "result": {
            "train_F": res["best_F"],
            "test_F": test["F"],
            "test_F_std": test["F_std"],
            "evals": res["evals"],
        },
        "note": ("best_theta.npy is the argmin-train-F individual of this "
                 "single guided-GA run; test_F is post-hoc on held-out seeds."),
    }
    (outdir / "best_theta.json").write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {outdir / 'best_theta.npy'} and {outdir / 'best_theta.json'}")
    print(f"train F={res['best_F']:.3f} test F={test['F']:.3f} "
          f"+/- {test['F_std']:.3f}")


if __name__ == "__main__":
    main()
