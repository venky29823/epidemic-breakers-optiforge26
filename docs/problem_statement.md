# Problem statement: Epidemic Breakers

## The problem

A 40-node / 111-edge microservice call graph (seed 7). Each directed edge
carries a circuit breaker with a per-edge threshold θ. When the noisy
stress signal on an edge (EMA of callee slowness + uniform noise) exceeds
θ, the breaker opens for a cooldown. The goal: choose all 111 thresholds
to minimize F = cascade_size + 2·false_trips + 0.1·latency_penalty over
held-out failure scenarios (24 train seeds, base 1000; 12 test seeds,
base 2000).

## What the data say (claim → supporting file)

All quantitative claims below come only from the listed files. No claim
is made without a cited source.

1. **Guided GA beats fixed-0.5 by ~18% on held-out seeds.**
   Guided GA test F 33.28 ± 1.51 vs fixed-0.5 40.56 (n=3 runs).
   → `results/round1_results.csv`
2. **Vanilla GA beats fixed-0.5 by ~12%.** Test F 35.68 ± 3.17 vs 40.56.
   → `results/round1_results.csv`
3. **Random search roughly ties vanilla GA.** 35.35 ± 1.69 vs 35.68 ± 3.17.
   → `results/round1_results.csv`
4. **Paired ablation (n=10): guidance effect not established.**
   Guided 31.61 ± 3.00 vs vanilla 32.88 ± 3.19; paired diff (vanilla −
   guided) mean 1.27, bootstrap 95% CI [−0.35, 3.03] — includes zero.
   → `results/ablation_summary.csv`
5. **Headroom exists.** Oracle reference (hidden noise + 0.10) scores
   24.74 ± 10.48 on 24 test seeds — a heuristic, not a ceiling, and not
   available to any real tuner. → reproducible snippet in README
   (uses `src/baselines.oracle_thresholds`).
6. **Demo theta is one run.** Precomputed thresholds: train F 21.77 /
   test F 34.91 (single guided-GA run, 1,230 evals), not a mean.
   → `results/best_theta.json`
7. **Round 2 recovery (n=2, indicative).** Warm-start recovers past stale
   in 5 and 7 generations on the shifted environment; from-scratch is
   steadier but the sample is two reps — not a reliability claim.
   → README "Round 2" table (reps documented there).

## Extended baselines (design; results pending)

`--ablation --extended` (n=3 pilot, same seeds/budget per pair) adds:

- **random-weights GA** — non-uniform mutation, no structural signal
  (control for the guided operator).
- **(1+1)-ES** — elitist single-parent ES, σ=0.1, same 1,230-eval budget.
- **calibrated** — per-edge noise amps from 24 probe sims
  (`sqrt(3·var)` on healthy-callee steps), θ = clip(amp+0.10); margin
  copies the oracle (not tuned); 24 probe sims, zero optimizer evals.

Pre-specified contrasts: calibrated vs guided; guided vs vanilla;
guided vs random-weights. All else exploratory. n=3 → very wide CIs;
pilot only. → `docs/analysis_plan.md`, `results/ablation_extended.csv`
(when the run finishes).

## Known limits

- Calibrated heuristic is flattered by stationary zero-centered noise;
  real telemetry (drift, diurnal, non-zero baselines) would break it.
- The +0.10 margin (oracle and calibrated) is arbitrary; margin 0 scores
  slightly better on the same seeds.
- Train/test gap is real (train ~21, test ~33): overfitting plus
  irreducible scenario variance.
