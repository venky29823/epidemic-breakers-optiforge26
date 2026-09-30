"""Re-evaluate fixed theta=0.5 on the current test seeds.

Writes results/fixed05_test_seeds.csv with per-seed F, cascade_size,
false_trips, and latency_penalty. Uses the same graph (40 nodes, seed 7)
and test seeds (12 seeds, base 2000) as the Round-1 comparison.

Usage:
    python3 scripts/reeval_fixed05.py [--outdir results]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow running as `python3 scripts/reeval_fixed05.py` from the repo root.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from src import graph_gen
from src.fitness import W_FALSE_TRIPS, W_LATENCY, scenario_cost, simulate_many
from src.utils import make_seeds

# Must match main.py ROUND1_KW and graph construction.
SIM_KW = {'spread_p': 0.25, 'cooldown': 5, 'n_steps': 40}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    graph = graph_gen.generate_service_graph(n_nodes=40, seed=7)
    theta = np.full(graph.number_of_edges(), 0.5)
    test_seeds = make_seeds(12, base=2000)

    sim_results = simulate_many(graph, theta, test_seeds, **SIM_KW)
    rows = []
    for seed, r in zip(test_seeds, sim_results):
        rows.append({
            "seed": seed,
            "F": scenario_cost(r, W_FALSE_TRIPS, W_LATENCY),
            "cascade_size": r["cascade_size"],
            "false_trips": r["false_trips"],
            "latency_penalty": r["latency_penalty"],
        })

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "fixed05_test_seeds.csv"
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    f_vals = np.array([r["F"] for r in rows])
    print(f"wrote {outpath}: mean F {f_vals.mean():.2f} ± {f_vals.std(ddof=1):.2f} "
          f"(n={len(rows)} seeds)")


if __name__ == "__main__":
    main()
