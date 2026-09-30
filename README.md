# Epidemic Breakers — OptiForge 2026

Tune circuit-breaker thresholds on a microservice call graph so cascading
slowness is contained without tripping breakers on healthy traffic.

## The problem

A microservice call graph: if service `v` slows down, every service that
*calls* `v` slows down with some probability each step — a cascade, like an
epidemic. Each dependency edge carries a circuit breaker with a threshold
`theta` in `[0, 1]`. The breaker watches a noisy stress signal; when the
signal exceeds `theta` it trips and stays open for a cooldown, cutting that
cascade path.

The dilemma: `theta` too low → noise alone trips breakers (**false trips**,
expensive). `theta` too high → the cascade spreads before breakers react.
Edges differ: each has its own monitoring-noise amplitude and spread
probability (fixed per graph, deterministic), so **no single fixed threshold
can be optimal** — thresholds must be tuned per edge.

Objective (lower is better), averaged over many simulated failure scenarios:

```
F = cascade_size + 2 * false_trips + 0.1 * latency_penalty
```

`latency_penalty` is the % of edge-steps spent with breakers open.

## Method

A genetic algorithm over the threshold vector `theta ∈ [0,1]^E`:

- tournament selection, uniform crossover, Gaussian mutation, elitism
- **guided mutation** (the contribution): mutated genes are chosen with
  probability proportional to edge betweenness centrality, blended 50/50
  with uniform choice — search effort concentrates on the structural
  super-spreader edges without starving the rest
- warm-start from a previous population + temporary mutation boost, for
  Round 2 re-adaptation

Baselines at the same evaluation budget: fixed `theta = 0.5` everywhere,
uniform random search (sharing the GA's informed init prior, so the
comparison isolates the optimizer), and a vanilla GA (uniform mutation) for
the ablation.

**Init prior (shared):** thresholds start uniform in `[0.2, 0.7]` — below
the noise floor everything false-trips, near 1.0 nothing trips in time.
This is domain knowledge, not cheating; every method gets it.

**Train/test discipline:** the search only ever sees TRAIN scenario seeds;
all reported numbers are on held-out TEST seeds.

## Layout

```
main.py            # entry point: comparison + Round 2 mode
src/graph_gen.py   # seeded random service graph (directed: u->v means u calls v)
src/simulator.py   # discrete-step cascade sim with breakers, cooldowns, noise
src/fitness.py     # objective F over scenario seeds
src/ga.py          # genetic algorithm (uniform / guided mutation, warm start)
src/baselines.py   # fixed threshold + random search
src/utils.py       # seeding, CSV, dependency-free SVG plotting
tests/             # pytest: simulator, fitness, GA
results/           # CSV tables + convergence plot (generated)
```

### Domain model (code ↔ problem statement)

| Problem-statement concept | Where it lives in the code |
|---------------------------|----------------------------|
| Microservice call graph (epidemic network) | `src/graph_gen.py` — seeded 40-node directed graph; `u→v` means `u` calls `v`, so failures propagate caller-ward like infections |
| Per-edge circuit-breaker threshold vector θ (111 dims) | `src/ga.py` — the GA individual; one threshold per edge |
| Cascade dynamics (slow-service infection, breaker trips, cooldown) | `src/simulator.py` — discrete-step epidemic-style spread with breaker state and cooldown timers |
| Objective F = cascade_size + 2·false_trips + 0.1·latency_penalty | `src/fitness.py` — averaged over held-out scenario seeds |
| Guided mutation (chokepoint-aware) | `src/ga.py` — mutation targets sampled ∝ edge-betweenness-blended weights |
| Baselines: fixed-θ, random search, oracle heuristic | `src/baselines.py` — oracle uses hidden per-edge noise, unavailable to any real tuner |
| Environment shift (Round 2) | `main.py --round2` — higher spread probability, longer breaker cooldown, warm-start adaptation |

## How to run

```bash
pip install -r requirements.txt   # numpy, networkx, pytest
pytest tests/ -q                  # 35 tests
python3 main.py                   # full comparison (~20 min, 3 runs/method)
python3 main.py --ablation        # paired ablation, 10 runs/method (~60 min)
python3 main.py --quick           # smoke test (~15 s)
python3 main.py --round2          # Round 2 shift experiment
```

`main.py` options: `--runs`, `--pop`, `--gens`, `--train-seeds`,
`--test-seeds`, `--seed`, `--outdir`.

## Demo

The static site (recommended for sharing):
<https://epidemic-breakers-web-15saicharan-3220s-projects.vercel.app/> —
zero-dependency HTML/CSS/vanilla-JS in the
[epidemic-breakers-web](https://github.com/venky29823/epidemic-breakers-web)
repo. It **re-implements the simulator in JavaScript** (`sim.js`: cascade
dynamics, fitness, edge betweenness, and a small in-browser GA) and bakes
in **precomputed data exported from this repo** — the exact 40-node /
111-edge graph, its per-edge noise/spread parameters, and the
GA-optimized thresholds (`data/*.json`). The precomputed theta is from
**one** guided-GA run (`scripts/make_best_theta.py`: train F 21.77 /
test F 34.91 on its 24 train / 12 test seeds), shown as-is — not an
average over runs. `scripts/check_web_parity.py`
verifies the baked-in constants agree with the Python source of truth
(graph size, edge identity/order, per-edge params, best_theta, sim
defaults); known differences are documented in that script's output
(JS vs numpy RNG streams differ, so single-scenario trajectories differ —
parity is statistical).

Pick a threshold strategy (fixed slider, precomputed GA-optimized, or the
oracle reference), tweak spread probability / cooldown / noise, and watch
the epidemic curve plus the `F` breakdown — or run a head-to-head
comparison on fresh seeds.

## Results (Round 1)

Source: `results/round1_results.csv` (fresh full run, 2026-09-30). 3
independent runs per method (1,230 evals each), 24 train seeds (base
1000) / 12 held-out test seeds (base 2000). Lower F is better. Test F is
mean ± sample sd (ddof=1) across runs.

| method        | test F (mean ± sd, n=3) | vs fixed-0.5 | source CSV |
|---------------|-------------------------|--------------|------------|
| fixed-0.5     | 40.56                   | —            | round1_results.csv |
| random search | 35.35 ± 1.69            | −13%         | round1_results.csv |
| vanilla GA    | 35.68 ± 3.17            | −12%         | round1_results.csv |
| guided GA     | 33.28 ± 1.51            | −18%         | round1_results.csv |

Claim-to-source map (all claims below come only from these files):

- "Both GAs beat fixed-0.5 by ~12–18% on held-out seeds" →
  `results/round1_results.csv` (test-F means 35.68, 33.28 vs 40.56).
- "Oracle scores 24.74 ± 10.48 on 24 test seeds" → reproducible via the
  snippet below (uses `src/baselines.oracle_thresholds`, not a CSV).
- "Paired ablation (n=10): guided 31.61 ± 3.00 vs vanilla 32.88 ± 3.19
  vs random-search 34.63 ± 3.47; paired diff 1.27, CI [−0.35, 3.03]" →
  `results/ablation_summary.csv`. The CI includes zero: the guidance
  effect is not established at n=10.
- "Random search roughly ties vanilla GA (35.35 vs 35.68)" →
  `results/round1_results.csv`.
- "Precomputed demo theta: train 21.77 / test 34.91 (one run)" →
  `results/best_theta.json` (`result.train_F`, `result.test_F`).

What the numbers actually say:

- Both GAs beat the fixed baseline clearly on held-out seeds (~12–18%).
  The oracle (`theta = clip(noise_amp + 0.10)` per edge — a reference
  heuristic using hidden noise values, not available to any real tuner)
  scores 24.74 ± 10.48 (mean ± sd over the 24 test scenarios) on the same
  test seeds, so real headroom exists and the
  GAs capture part of it. Reproduce it exactly with:

  ```bash
  python3 - <<'EOF'
  from src import graph_gen, baselines as bl, fitness as fit
  from src.utils import make_seeds
  from main import ROUND1_KW
  G = graph_gen.generate_service_graph(n_nodes=40, seed=7)
  theta = bl.oracle_thresholds(G)  # canonical: clip(noise_amp + 0.10)
  r = fit.evaluate(G, theta, make_seeds(24, base=2000), **ROUND1_KW)
  print(f"oracle test F = {r['F']:.2f} ± {r['F_std']:.2f}")
  EOF
  ```
- Paired ablation (n=10, same seeds and budget per pair): guided
  31.61 ± 3.00 vs vanilla 32.88 ± 3.19 vs random-search 34.63 ± 3.47
  (test F, mean ± sd). Paired diff (vanilla − guided): mean 1.27,
  sd 2.91, bootstrap 95% CI [−0.35, 3.03]; guided won 5/10 paired runs.
  The point estimate favors guidance, but the CI includes zero — the
  guidance effect is not statistically established at n=10. (This
  supersedes the earlier n=3 pilot, which was likewise inconclusive.)
- Random search is *stronger than folklore suggests*: with the same
  informed init prior it roughly ties the vanilla GA (35.35 vs 35.68).
  We report it because a strong baseline makes the guided variant's win
  meaningful instead of inflated. The GA's edge at this budget is final
  quality and consistency, not dramatic sample efficiency.

See `results/round1_results.csv` and `results/convergence.svg`.

### Ablation (paired, n=10)

`python3 main.py --ablation --outdir results` (~60 min). Same 24 train /
12 test seeds and evaluation budget per pair (fresh seed bases per run:
train base 1000+run, test base 2000+run); raw per-run values in
`results/ablation.csv`, summary in `results/ablation_summary.csv`.

| method        | train F | test F (mean ± sd) |
|---------------|---------|--------------------|
| random-search | 30.64   | 34.63 ± 3.47       |
| vanilla GA    | 24.01   | 32.88 ± 3.19       |
| guided GA     | 23.54   | 31.61 ± 3.00       |

Paired (vanilla − guided) test F: mean 1.27, sd 2.91, bootstrap 95% CI
[−0.35, 3.03] (positive favors guided); guided won 5/10 paired runs.
Regenerate with `python3 main.py --ablation --outdir results`.

### Extended baselines (paired, n=3 pilot)

`python3 main.py --ablation --extended --runs 3 --outdir results` (~40 min)
adds three baselines to the same paired design (fresh RNG offsets; the
original three methods' streams are untouched). Output:
`results/ablation_extended.csv`. Pre-specified contrasts are in
`docs/analysis_plan.md`.

- **random-weights GA**: guided-mutation machinery with uniform-random
  weights — isolates whether the win comes from non-uniformity alone.
- **(1+1)-ES**: single-parent elitist evolution strategy, Gaussian
  mutation (σ=0.1), same 1,230-eval budget.
- **calibrated**: estimates per-edge noise amplitudes from 24 probe
  simulations (breakers disabled, healthy-callee steps only) via
  `amp_hat = sqrt(3*var(obs))`, then `theta = clip(amp_hat + 0.10)`.
  The +0.10 margin was **not** tuned on train seeds — it copies the
  oracle's margin for comparability, and is arbitrary (see Known
  limitations). The calibrated method uses 24 probe simulator calls and
  **zero** optimizer evaluations; the 1,230-eval budget applies to the
  optimizer methods only (each eval = 24 train-seed simulations).

| method            | train F | test F (mean ± sd, n=3) | status |
|-------------------|---------|-------------------------|--------|
| random-search     | 26.70   | 36.50 ± 4.47            | n=3 pilot |
| vanilla GA        | 21.64   | 36.20 ± 2.45            | n=3 pilot |
| guided GA         | 20.43   | 33.53 ± 2.54            | n=3 pilot |
| random-weights GA | 20.77   | 34.65 ± 3.26            | n=3 pilot |
| (1+1)-ES          | 31.67   | 40.00 ± 0.70            | n=3 pilot |
| calibrated        | 20.03   | 24.19 ± 2.23            | n=3 pilot |

Source: `results/ablation_extended.csv`. Pre-specified paired contrasts
(bootstrap 95% CI, 10,000 resamples; positive diff favors the second
method):

- **calibrated vs guided GA**: mean diff −9.34, CI [−11.72, −7.91] —
  calibrated lower (better) in all 3 runs; CI excludes zero.
- **guided vs vanilla GA**: mean diff +2.68, CI [−0.49, 4.66] — point
  estimate favors guided, CI includes zero: not established.
- **guided vs random-weights GA**: mean diff +1.13, CI [−2.42, 3.56] —
  CI includes zero: not established.

n=3 is a pilot: CIs are wide and these are not definitive. No claim is
made beyond what the intervals support. All other pairwise comparisons
are exploratory (see `docs/analysis_plan.md`).

## Round 2: hidden shift

`python3 main.py --round2` hardens the environment: spread probability
0.25 → 0.45 and breaker cooldown 5 → 12 steps. It then compares, on
Round 2 test seeds (2 reps, 20 adaptation generations each):

- **warm-start**: Round 1 final population + diversity noise + 3x mutation
  boost for 10 generations
- **from-scratch**: fresh population, same generation budget
- **stale**: Round 1 thresholds deployed as-is (no adaptation)

| method       | rep | final test F | stale F | gens to recover past stale |
|--------------|-----|--------------|---------|----------------------------|
| warm-start   | 0   | 29.30        | 35.67   | 5                          |
| from-scratch | 0   | 37.79        | 35.67   | — (never in 20 gens)       |
| warm-start   | 1   | 35.78        | 39.93   | 7                          |
| from-scratch | 1   | 32.31        | 39.93   | 0 (lucky init)             |

Reading: warm-start recovered past the stale baseline on **both** reps
(5 and 7 generations); restart-from-scratch recovered on only one rep and
failed outright on the other. On these two reps the warm-start advantage is
steadier recovery rather than final quality — it pays a small upfront cost from the
diversity noise (see the early part of `results/recovery.svg`) and then
adapts steadily. Both adapted methods beat doing nothing (stale mean
37.80). Raw curves: `results/round2_curves.csv`.

## Known limitations

- **Small n for Round 2**: 2 reps. The recovery-speed claim is about
  reliability across the reps we ran, not a tight estimate — treat the
  5/7-generation figures as indicative.
- **Train/test gap**: real and reported (train ~21, test ~33 for the
  guided GA). Part overfitting to 24 train seeds, part irreducible
  scenario variance (test-F std ~10 even for the oracle). The
  `--ablation` train/test diagnostic in `results/ablation.csv` separates
  the two per method.
- **Oracle is a heuristic**: `theta = clip(noise_amp + 0.10)` uses hidden
  per-edge noise values no real tuner can see. It marks headroom, not a
  ceiling — a method beating it would not be a contradiction. The +0.10
  margin itself is arbitrary: margin 0 scores 23.15 ± 9.72, slightly
  better than the canonical +0.10 (24.74 ± 10.48) on the same 24 test
  seeds. Don't over-interpret the exact margin.
- **Round-1 table provenance**: the table now comes from a fresh full run
  (2026-09-30, `results/round1_results.csv`, deterministic seeds). The
  original run's per-run CSVs were overwritten by `--quick` smoke runs
  before the initial commit, but the fresh run reproduces the original
  figures' means to 2 decimals (incl. fixed-0.5 = 40.56 on the 12 test
  seeds), so the table stands on the new run. An earlier same-seed
  re-evaluation on 24 test seeds gave fixed-0.5 = 37.70 ± 11.29 vs oracle
  24.74 ± 10.48 — use those for apples-to-apples claims at n=24.
- **Shifts tested**: environment shift (spread 0.25→0.45, cooldown 5→12)
  and edge-removal shift (`--round2-graph-shift`, ~10% of edges). Other
  shift types (new services, correlated noise, weight changes) are
  supported by the code paths but not measured in the report.
- **Calibrated heuristic is simulator-flattered**: it estimates per-edge
  noise amplitudes from probe observables via `sqrt(3*var)` on
  healthy-callee steps, which is exact only because this simulator's
  noise is stationary and zero-centered uniform (E[obs]=0 when the callee
  is healthy). Real telemetry has drift, diurnal patterns, and
  non-zero-centered baselines that would bias or break the estimate; the
  heuristic's edge over the GA here does not imply it transfers to
  production signals.

## Societal impact (SDG 9)

This work targets **SDG 9: Industry, Innovation & Infrastructure**
(Target 9.4 — upgrade infrastructure with resilient, resource-efficient
technologies). Cascading slowdowns are a direct threat to the digital
infrastructure modern industry runs on; per-edge adaptive
circuit-breaking is a concrete mechanism for containing them, and the
Round-2 warm-start result shows tuned defenses can be re-adapted in a
handful of generations when operating conditions shift, rather than
rebuilt from scratch. The method itself is deliberately lightweight —
NumPy and NetworkX only, no accelerators, fully seeded and reproducible —
so the resilience it offers does not come with heavy compute overhead. We
make no claim beyond this: a small, honest contribution to infrastructure
resilience, with all limits documented above.

## Defense notes (for the judges)

- *Why per-edge thresholds?* Edges have different noise/spread profiles;
  the oracle (`theta = clip(noise_amp + 0.10)` per edge — a reference
  heuristic using hidden noise values, unavailable to any real tuner)
  scores 24.74 ± 10.48 vs 37.70 ± 11.29 for fixed-0.5 — roughly 34%
  lower, both re-evaluated on the same 24 test seeds (base 2000). The
  ±10–11 scenario sds mean single-seed comparisons are noisy; the gap is
  real on average, not on every seed.
- *Why not just grid-search one threshold?* Same reason — one number cannot
  fit 111 different edges.
- *Isn't this SIR relabeled?* The mapping is deliberate, but false trips,
  cooldowns, and latency penalties are software-specific costs with no
  epidemic analogue.
- *Overfitting?* Train/test seed split; reported numbers are held-out.
  The train/test gap is real (train ~21, test ~33) — we report both.
- *Why the GA, if random search ties vanilla?* Because the comparison is
  the point: with an informed prior, random search *is* strong, and the
  vanilla GA barely beats it. The contribution is the guided operator,
  which has a positive point estimate against both at the same budget —
  but with n=10 the paired 95% CI ([−0.35, 3.03]) includes zero, so the
  guidance effect is not statistically established. A weak baseline would
  have hidden that.
- *Round 2?* Warm-start recovered past the stale baseline in 5 and 7
  generations on both reps; restart-from-scratch failed on one rep.
  Suggestive of steadier recovery under shift — not a reliability claim
  (n=2).
