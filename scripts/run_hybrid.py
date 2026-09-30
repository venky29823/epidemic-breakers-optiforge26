"""GA warm-started from calibrated thetas (hybrid), same budget and seeds.

For each of the 10 paired runs (train seeds base 1000+run, test seeds
base 2000+run — identical to --ablation --extended):
  1. Compute the calibrated theta on the run's train seeds (deterministic).
  2. Run the guided GA (pop 30, 40 gens = 1,230 evals) with init_pop =
     [calibrated theta] + 29 random individuals, so the optimizer starts
     from the heuristic solution instead of a random population.
  3. Evaluate the final theta on the run's test seeds.

Writes results/ablation_hybrid.csv (separate file; ablation_extended.csv
is never modified). Compare against calibrated-only and guided GA with
paired bootstrap CIs; make no claims beyond what the CIs support.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import baselines as blmod
from src import fitness as fitmod
from src import ga as gamod
from src import graph_gen
from src.utils import make_rng, make_seeds

N_RUNS = 10
N_TRAIN = 24
N_TEST = 12
SEED = 42
POP = 30
GENS = 40
ROUND1_KW = {'n_steps': 40, 'noise_amp': 0.3, 'spread_p': 0.25, 'cooldown': 5}


def paired_bootstrap_ci(diffs, n_boot=10000, seed=0):
    rng = make_rng(seed)
    n = len(diffs)
    boots = rng.choice(diffs, size=(n_boot, n), replace=True).mean(axis=1)
    return (float(np.mean(diffs)),
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


def main():
    G = graph_gen.generate_service_graph(n_nodes=40, seed=7)
    n_edges = G.number_of_edges()
    # budget = POP * (GENS + 1)  # 1,230 evals, same as the GA baselines
    cfg = gamod.GAConfig(pop_size=POP, generations=GENS,
                         mutation_mode="guided")

    rows = []
    for run in range(N_RUNS):
        train_seeds = make_seeds(N_TRAIN, base=1000 + run)
        test_seeds = make_seeds(N_TEST, base=2000 + run)
        t0 = time.time()

        theta_cal = blmod.calibrated_thresholds(G, train_seeds, **ROUND1_KW)
        rng = make_rng(SEED + 6000 + run * 10 + 4)  # fresh offset; no collision
        init_pop = np.empty((POP, n_edges))
        init_pop[0] = np.clip(theta_cal, 0.0, 1.0)
        init_pop[1:] = rng.uniform(cfg.init_lo, cfg.init_hi,
                                   size=(POP - 1, n_edges))
        res = gamod.run_ga(G, train_seeds, cfg, rng, init_pop=init_pop,
                           **ROUND1_KW)
        test = fitmod.evaluate(G, res["best"], test_seeds, **ROUND1_KW)
        rows.append({
            "method": "hybrid (calibrated warm-start)",
            "run": run,
            "train_F": round(float(res["best_F"]), 3),
            "test_F": round(float(test["F"]), 3),
            "test_F_std": round(float(test["F_std"]), 3),
            "evals": int(res["evals"]),
            "secs": round(time.time() - t0, 1),
        })
        print(f"hybrid run {run + 1}/{N_RUNS}: train F={res['best_F']:.3f} "
              f"test F={test['F']:.3f}", flush=True)

    out = REPO / "results" / "ablation_hybrid.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")

    # Paired comparison vs calibrated-only and guided GA (from extended CSV).
    with open(REPO / "results" / "ablation_extended.csv") as f:
        ext = list(csv.DictReader(f))
    test_f = {m: np.array([float(r["test_F"]) for r in ext
                           if r["method"] == m])
              for m in ("calibrated", "guided GA")}
    hyb = np.array([r["test_F"] for r in rows])
    print("\nHYBRID vs baselines (paired, n=10, >0 favours the second method)")
    print("-" * 70)
    print(f"{'contrast':<32}{'mean diff':<12}{'95% CI'}")
    for label, base in (("calibrated", "hybrid"), ("guided GA", "hybrid")):
        diffs = test_f[label] - hyb
        d_mean, d_lo, d_hi = paired_bootstrap_ci(diffs, seed=SEED + 999)
        wins = int((diffs > 0).sum())
        print(f"{label:<32}{d_mean:<12.3f}[{d_lo:.3f}, {d_hi:.3f}] "
              f"(hybrid wins {10 - wins}/10)")


if __name__ == "__main__":
    main()
