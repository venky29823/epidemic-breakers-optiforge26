"""Entry point: baselines vs GA with a strict train/test seed split.

Runs (default): fixed threshold, random search, vanilla GA, guided GA --
each over several seeded runs. Prints a results table, saves CSV + an SVG
convergence plot under results/.

Round 2 mode (--round2): harder environment (higher spread probability,
longer breaker cooldown). Compares warm-starting the GA from the Round 1
population (with a temporary mutation boost) against restarting from
scratch, measuring recovery on Round 2 test seeds.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from src import baselines as blmod
from src import es as esmod
from src import fitness as fitmod
from src import ga as gamod
from src import graph_gen
from src.utils import (
    ensure_dir,
    get_logger,
    make_rng,
    make_seeds,
    paired_bootstrap_ci,
    save_convergence_svg,
    write_csv,
)

log = get_logger()

ROUND1_KW = dict(spread_p=0.25, cooldown=5, n_steps=40)
ROUND2_KW = dict(spread_p=0.45, cooldown=12, n_steps=40)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Epidemic Breakers - OptiForge 2026")
    p.add_argument("--runs", type=int, default=None,
                   help="independent runs per method (default 3; 10 with --ablation)")
    p.add_argument("--train-seeds", type=int, default=24)
    p.add_argument("--test-seeds", type=int, default=12)
    p.add_argument("--pop", type=int, default=30, help="GA population size")
    p.add_argument("--gens", type=int, default=40, help="GA generations")
    p.add_argument("--seed", type=int, default=42, help="master seed")
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--quick", action="store_true", help="tiny budgets for smoke tests")
    p.add_argument("--round2", action="store_true", help="run the Round 2 shift experiment")
    p.add_argument("--round2-graph-shift", action="store_true",
                   help="also remove ~10% of edges in the Round 2 shift")
    p.add_argument("--ablation", action="store_true",
                   help="paired guided vs vanilla GA vs random search over --runs runs")
    p.add_argument("--extended", action="store_true",
                   help="with --ablation: also run random-weights GA, "
                        "(1+1)-ES and calibrated-noise baselines; writes "
                        "ablation_extended.csv instead of ablation.csv")
    return p


def summarise(method: str, run: int, res: dict, train_f: float, secs: float) -> dict:
    t = res["test"]
    return {
        "method": method,
        "run": run,
        "test_F_mean": round(t["F"], 3),
        "test_F_std": round(t["F_std"], 3),
        "cascade_size": round(t["cascade_size"], 2),
        "false_trips": round(t["false_trips"], 2),
        "latency_penalty": round(t["latency_penalty"], 2),
        "train_F": round(train_f, 3),
        "gens_to_target": res.get("gens_to_target") if res.get("gens_to_target") is not None else "-",
        "evals": res.get("evals", ""),
        "secs": round(secs, 1),
    }


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print("-" * 108)
    hdr = (f"{'method':<16}{'run':<5}{'test F':<16}{'cascade':<10}"
           f"{'false':<8}{'latency':<10}{'gens->tgt':<10}{'secs':<6}")
    print(hdr)
    print("-" * 108)
    for r in rows:
        fstr = f"{r['test_F_mean']:.2f} +/- {r['test_F_std']:.2f}"
        print(f"{r['method']:<16}{r['run']:<5}{fstr:<16}{r['cascade_size']:<10.1f}"
              f"{r['false_trips']:<8.1f}{r['latency_penalty']:<10.1f}"
              f"{str(r['gens_to_target']):<10}{r['secs']:<6.1f}")
    print("-" * 108)


def run_comparison(args) -> list[dict]:
    G = graph_gen.generate_service_graph(n_nodes=40, seed=7)
    n_edges = G.number_of_edges()
    log.info("graph: %d nodes, %d edges", G.number_of_nodes(), n_edges)
    train_seeds = make_seeds(args.train_seeds, base=1000)
    test_seeds = make_seeds(args.test_seeds, base=2000)
    assert not set(train_seeds) & set(test_seeds), "train/test seeds must be disjoint"

    cfg_v = gamod.GAConfig(pop_size=args.pop, generations=args.gens, mutation_mode="uniform")
    cfg_g = gamod.GAConfig(pop_size=args.pop, generations=args.gens, mutation_mode="guided")
    budget = args.pop * (args.gens + 1)  # == GA evals: initial pop + one per gen

    # fixed baseline; the target is beating it by 15% on train (demanding,
    # since the informed init already starts near the baseline)
    t0 = time.time()
    fixed = blmod.fixed_threshold(G, 0.5, train_seeds, test_seeds, **ROUND1_KW)
    target = 0.85 * fixed["train_F"]
    rows = [summarise(fixed["method"], 0, fixed, fixed["train_F"], time.time() - t0)]
    log.info("fixed-0.5 train F = %.3f -> target %.3f", fixed["train_F"], target)

    histories = {"vanilla GA": [], "guided GA": []}
    for run in range(args.runs):
        run_rng = make_rng(args.seed + 100 + run)

        t0 = time.time()
        rs = blmod.random_search(G, train_seeds, test_seeds, budget, run_rng, **ROUND1_KW)
        rows.append(summarise("random-search", run, rs, rs["train_F"], time.time() - t0))

        for name, cfg in (("vanilla GA", cfg_v), ("guided GA", cfg_g)):
            t0 = time.time()
            ga_rng = make_rng(args.seed + 1000 + run * 10 + (0 if name == "vanilla GA" else 1))
            res = gamod.run_ga(G, train_seeds, cfg, ga_rng,
                               target_score=target, **ROUND1_KW)
            test = fitmod.evaluate(G, res["best"], test_seeds, **ROUND1_KW)
            res["test"] = test
            rows.append(summarise(name, run, res, res["best_F"], time.time() - t0))
            histories[name].append(res["history"])
            log.info("%s run %d: train F=%.3f test F=%.3f", name, run, res["best_F"], test["F"])

    print_table(rows, "ROUND 1 RESULTS (test seeds, lower F is better)")

    outdir = ensure_dir(args.outdir)
    write_csv(outdir / "round1_results.csv", rows, list(rows[0].keys()))
    mean_hist = {
        name: list(np.mean(h, axis=0)) for name, h in histories.items() if h
    }
    if mean_hist:
        save_convergence_svg(outdir / "convergence.svg", mean_hist)
        log.info("wrote %s", outdir / "convergence.svg")
    log.info("wrote %s", outdir / "round1_results.csv")
    return rows


def run_round2(args) -> list[dict]:
    """Shift the environment, then compare warm-start vs scratch recovery.

    The honest metric for adaptation is *speed*: generations needed to
    recover past the stale (non-adapted) Round 1 thresholds, measured on
    Round 2 test seeds. Warm-start = Round 1 final population + diversity
    noise + temporary mutation boost; scratch = fresh population.

    With ``--round2-graph-shift`` the shift additionally removes ~10% of
    edges; thresholds then follow edge identity (surviving edges keep their
    values, removed edges are dropped) via :func:`ga.remap_population`.
    """
    G = graph_gen.generate_service_graph(n_nodes=40, seed=7)
    train_seeds = make_seeds(args.train_seeds, base=1000)
    test_seeds = make_seeds(args.test_seeds, base=2000)
    assert not set(train_seeds) & set(test_seeds)

    old_edges = graph_gen.edge_list(G)
    if args.round2_graph_shift:
        grng = make_rng(args.seed + 999)
        G2 = G.copy()
        n_remove = max(1, int(round(0.10 * len(old_edges))))
        drop = grng.choice(len(old_edges), size=n_remove, replace=False)
        for i in sorted(drop, reverse=True):
            G2.remove_edge(*old_edges[i])
        log.info("graph shift: removed %d/%d edges (%.0f%%)",
                 n_remove, len(old_edges), 100 * n_remove / len(old_edges))
        G_adapt, shift_tag = G2, f"edges-{n_remove}-removed"
    else:
        G_adapt, shift_tag = G, "none"

    adapt_gens = max(10, args.gens // 2)
    n_reps = 2
    rows: list[dict] = []
    curves: dict[str, list[list[float]]] = {"warm-start": [], "from-scratch": []}
    stale_fs: list[float] = []

    for rep in range(n_reps):
        log.info("Round 2 rep %d: training guided GA on Round 1 env ...", rep)
        r1 = gamod.run_ga(
            G, train_seeds,
            gamod.GAConfig(pop_size=args.pop, generations=args.gens,
                           mutation_mode="guided"),
            make_rng(args.seed + rep), **ROUND1_KW,
        )
        # stale: Round 1 thresholds deployed as-is (remapped by edge
        # identity when the graph shift removed edges)
        wrng = make_rng(args.seed + 100 + rep)
        if args.round2_graph_shift:
            stale_theta = gamod.remap_population(
                r1["best"].reshape(1, -1), old_edges, G_adapt, wrng)[0]
        else:
            stale_theta = r1["best"]
        stale = fitmod.evaluate(G_adapt, stale_theta, test_seeds, **ROUND2_KW)
        stale_fs.append(stale["F"])
        log.info("rep %d: Round 1 train F=%.3f; stale in Round 2: test F=%.3f",
                 rep, r1["best_F"], stale["F"])
        log.info("rep %d: shift spread_p %.2f -> %.2f, cooldown %d -> %d", rep,
                 ROUND1_KW["spread_p"], ROUND2_KW["spread_p"],
                 ROUND1_KW["cooldown"], ROUND2_KW["cooldown"])

        adapt_cfg = gamod.GAConfig(pop_size=args.pop, generations=adapt_gens,
                                   mutation_mode="guided")
        # warm start: previous population (remapped across a graph shift) +
        # diversity injection + boost
        base = (gamod.remap_population(r1["final_pop"], old_edges, G_adapt, wrng)
                if args.round2_graph_shift else r1["final_pop"])
        init = np.clip(
            base + wrng.normal(0.0, 0.08, base.shape),
            0.0, 1.0,
        )
        warm = gamod.run_ga(G_adapt, train_seeds, adapt_cfg, wrng, init_pop=init,
                            mutation_boost=3.0, boost_gens=10, **ROUND2_KW)
        scratch = gamod.run_ga(G_adapt, train_seeds, adapt_cfg,
                               make_rng(args.seed + 200 + rep), **ROUND2_KW)

        for name, res in (("warm-start", warm), ("from-scratch", scratch)):
            test_curve = [fitmod.evaluate(G_adapt, ind, test_seeds, **ROUND2_KW)["F"]
                          for ind in res["best_per_gen"]]
            curves[name].append(test_curve)
            rec = next((g for g, f in enumerate(test_curve) if f <= stale["F"]),
                       None)
            rows.append({
                "method": name, "rep": rep,
                "graph_shift": shift_tag,
                "final_test_F": round(test_curve[-1], 3),
                "stale_test_F": round(stale["F"], 3),
                "gens_to_recover": rec if rec is not None else "-",
                "evals": res["evals"],
            })
            log.info("rep %d %s: final test F=%.3f, gens_to_recover=%s",
                     rep, name, test_curve[-1], rec)

    mean_curves = {k: list(np.mean(v, axis=0)) for k, v in curves.items()}
    mean_stale = float(np.mean(stale_fs))

    print("\nROUND 2 RECOVERY (shifted env; lower test F is better)")
    if args.round2_graph_shift:
        print(f"graph shift: {shift_tag}")
    print("-" * 70)
    print(f"{'method':<14}{'rep':<5}{'final test F':<14}{'stale F':<10}{'gens_to_recover'}")
    print("-" * 70)
    for r in rows:
        print(f"{r['method']:<14}{r['rep']:<5}{r['final_test_F']:<14.2f}"
              f"{r['stale_test_F']:<10.2f}{r['gens_to_recover']}")
    print("-" * 70)
    print(f"stale (no adaptation) mean test F: {mean_stale:.2f}")

    outdir = ensure_dir(args.outdir)
    write_csv(outdir / "round2_results.csv", rows, list(rows[0].keys()))
    curve_rows = [
        {"gen": g, "warm_start": round(w, 3), "from_scratch": round(s, 3)}
        for g, (w, s) in enumerate(zip(mean_curves["warm-start"],
                                       mean_curves["from-scratch"]))
    ]
    write_csv(outdir / "round2_curves.csv", curve_rows,
              ["gen", "warm_start", "from_scratch"])
    save_convergence_svg(
        outdir / "recovery.svg",
        {"warm-start": mean_curves["warm-start"],
         "from-scratch": mean_curves["from-scratch"]},
        title="Round 2 recovery (test F per adaptation generation)",
        xlabel="adaptation generation",
        ylabel="test F (lower is better)",
    )
    log.info("wrote %s, %s, %s", outdir / "round2_results.csv",
             outdir / "round2_curves.csv", outdir / "recovery.svg")
    return rows


def run_ablation(args) -> list[dict]:
    """Paired ablation: guided GA vs vanilla GA vs random search.

    ``args.runs`` runs (default 10; 2 under ``--quick``). Run ``i`` draws
    its own train/test seed sets and all three methods train and test on
    those identical sets, so per-run differences are paired. Every method
    gets the same evaluation budget: ``pop * (gens + 1)``, matching the
    GA's initial population plus one fitness evaluation per individual
    per generation.

    Prints per-method mean/sd, the paired (vanilla - guided) differences,
    and a paired bootstrap 95% CI for the mean difference. Saves per-run
    rows -- train F, test F and test-F std for each method's final theta,
    i.e. the train/test diagnostic -- to ``results/ablation.csv`` and the
    summary to ``results/ablation_summary.csv``.

    With ``--extended``, three more baselines join the same paired design
    (fresh RNG offsets, so the original three methods' streams -- and
    results -- are untouched): a random-weights GA control (non-uniform
    mutation with no structural signal), a (1+1)-ES, and a calibrated
    noise heuristic fitted from probe observables. Extended output goes
    to ``results/ablation_extended.csv`` /
    ``results/ablation_extended_summary.csv``.
    """
    G = graph_gen.generate_service_graph(n_nodes=40, seed=7)
    log.info("graph: %d nodes, %d edges", G.number_of_nodes(),
             G.number_of_edges())
    n_runs = args.runs
    extended = args.extended
    budget = args.pop * (args.gens + 1)  # == GA evals: initial pop + one per gen
    cfg_v = gamod.GAConfig(pop_size=args.pop, generations=args.gens,
                           mutation_mode="uniform")
    cfg_g = gamod.GAConfig(pop_size=args.pop, generations=args.gens,
                           mutation_mode="guided")
    cfg_rw = gamod.GAConfig(pop_size=args.pop, generations=args.gens,
                            mutation_mode="random-weights")
    methods = ["random-search", "vanilla GA", "guided GA"]
    if extended:
        methods += ["random-weights GA", "(1+1)-ES", "calibrated"]
    prefix = "ablation_extended" if extended else "ablation"

    rows: list[dict] = []
    for run in range(n_runs):
        train_seeds = make_seeds(args.train_seeds, base=1000 + run)
        test_seeds = make_seeds(args.test_seeds, base=2000 + run)
        assert not set(train_seeds) & set(test_seeds)

        t0 = time.time()
        rs = blmod.random_search(G, train_seeds, test_seeds, budget,
                                 make_rng(args.seed + 5000 + run), **ROUND1_KW)
        rows.append(_ablation_row("random-search", run, rs["train_F"],
                                  rs["test"], rs["evals"], time.time() - t0))

        for name, cfg, off in (("vanilla GA", cfg_v, 0),
                               ("guided GA", cfg_g, 1)):
            t0 = time.time()
            ga_rng = make_rng(args.seed + 6000 + run * 10 + off)
            res = gamod.run_ga(G, train_seeds, cfg, ga_rng, **ROUND1_KW)
            test = fitmod.evaluate(G, res["best"], test_seeds, **ROUND1_KW)
            rows.append(_ablation_row(name, run, res["best_F"], test,
                                      res["evals"], time.time() - t0))

        if extended:
            t0 = time.time()
            rw_rng = make_rng(args.seed + 6000 + run * 10 + 2)
            rw_res = gamod.run_ga(G, train_seeds, cfg_rw, rw_rng, **ROUND1_KW)
            rw_test = fitmod.evaluate(G, rw_res["best"], test_seeds,
                                      **ROUND1_KW)
            rows.append(_ablation_row("random-weights GA", run,
                                      rw_res["best_F"], rw_test,
                                      rw_res["evals"], time.time() - t0))

            t0 = time.time()
            es_res = esmod.run_es(G, train_seeds, budget,
                                  make_rng(args.seed + 6000 + run * 10 + 3),
                                  **ROUND1_KW)
            es_test = fitmod.evaluate(G, es_res["best"], test_seeds,
                                      **ROUND1_KW)
            rows.append(_ablation_row("(1+1)-ES", run, es_res["best_F"],
                                      es_test, es_res["evals"],
                                      time.time() - t0))

            t0 = time.time()
            theta_cal = blmod.calibrated_thresholds(G, train_seeds,
                                                    **ROUND1_KW)
            train_cal = fitmod.evaluate(G, theta_cal, train_seeds,
                                        **ROUND1_KW)
            test_cal = fitmod.evaluate(G, theta_cal, test_seeds, **ROUND1_KW)
            # evals for calibrated = probe simulations, not optimizer evals
            rows.append(_ablation_row("calibrated", run, train_cal["F"],
                                      test_cal, len(train_seeds),
                                      time.time() - t0))
        log.info("ablation run %d/%d done", run + 1, n_runs)

    test_f = {m: np.array([r["test_F"] for r in rows if r["method"] == m])
              for m in methods}
    train_f = {m: np.array([r["train_F"] for r in rows if r["method"] == m])
               for m in methods}
    pairs = ([("vanilla GA", "guided GA")] if not extended
             else [(m, "guided GA") for m in methods if m != "guided GA"])
    pair_stats = []
    for a, b in pairs:
        diffs_ab = test_f[a] - test_f[b]  # > 0 favours b (guided)
        d_mean, d_lo, d_hi = paired_bootstrap_ci(
            diffs_ab, rng=make_rng(args.seed + 777 + len(pair_stats)))
        d_sd = diffs_ab.std(ddof=1) if n_runs > 1 else 0.0
        n_pos = int((diffs_ab > 0).sum())
        pair_stats.append((a, b, d_mean, d_lo, d_hi, d_sd, n_pos))

    print(f"\nABLATION: paired runs, equal budget ({budget} evals each)")
    print("-" * 78)
    print(f"{'method':<17}{'n':<4}{'train F':<10}{'test F (mean ± sd)':<22}")
    print("-" * 78)
    for m in methods:
        sd = test_f[m].std(ddof=1) if n_runs > 1 else 0.0
        print(f"{m:<17}{n_runs:<4}{train_f[m].mean():<10.2f}"
              f"{test_f[m].mean():.2f} ± {sd:.2f}")
    print("-" * 78)
    for a, b, d_mean, d_lo, d_hi, d_sd, n_pos in pair_stats:
        print(f"paired ({a} - {b}) test F: mean {d_mean:.2f}, sd {d_sd:.2f}; "
              f"bootstrap 95% CI [{d_lo:.2f}, {d_hi:.2f}] "
              f"(positive favours {b}); {b} won {n_pos}/{n_runs}")

    outdir = ensure_dir(args.outdir)
    write_csv(outdir / f"{prefix}.csv", rows, list(rows[0].keys()))
    summary = [
        {"method": m, "n_runs": n_runs,
         "mean_train_F": round(float(train_f[m].mean()), 3),
         "mean_test_F": round(float(test_f[m].mean()), 3),
         "sd_test_F": round(float(test_f[m].std(ddof=1)) if n_runs > 1 else 0.0, 3),
         "paired_diff_mean": "", "paired_diff_ci_lo": "", "paired_diff_ci_hi": ""}
        for m in methods
    ]
    for a, b, d_mean, d_lo, d_hi, d_sd, _n in pair_stats:
        summary.append({"method": f"paired: {a} - {b}", "n_runs": n_runs,
                        "mean_train_F": "", "mean_test_F": "",
                        "sd_test_F": round(float(d_sd), 3),
                        "paired_diff_mean": round(d_mean, 3),
                        "paired_diff_ci_lo": round(d_lo, 3),
                        "paired_diff_ci_hi": round(d_hi, 3)})
    write_csv(outdir / f"{prefix}_summary.csv", summary,
              ["method", "n_runs", "mean_train_F", "mean_test_F", "sd_test_F",
               "paired_diff_mean", "paired_diff_ci_lo", "paired_diff_ci_hi"])
    log.info("wrote %s, %s", outdir / f"{prefix}.csv",
             outdir / f"{prefix}_summary.csv")
    return rows


def _ablation_row(method: str, run: int, train_f: float, test: dict,
                  evals: int, secs: float) -> dict:
    """One per-run ablation record: train/test diagnostic for a final theta."""
    return {
        "method": method,
        "run": run,
        "train_F": round(train_f, 3),
        "test_F": round(test["F"], 3),
        "test_F_std": round(test["F_std"], 3),
        "evals": evals,
        "secs": round(secs, 1),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.quick:  # tiny budgets for smoke tests
        args.pop, args.gens = 8, 6
        args.train_seeds, args.test_seeds = 4, 4
        if args.runs is None:
            args.runs = 2
    if args.runs is None:
        args.runs = 10 if args.ablation else 3
    if args.ablation:
        run_ablation(args)
    elif args.round2:
        run_round2(args)
    else:
        run_comparison(args)


if __name__ == "__main__":
    main()
