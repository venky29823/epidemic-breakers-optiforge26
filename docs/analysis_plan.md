# Analysis plan: extended baselines (paired ablation)

## Design

`python3 main.py --ablation --extended --runs 3 --outdir results`
(~40 min). Paired across 10→3 runs (time-boxed to n=3): each run uses
fresh seed bases (train base 1000+run, test base 2000+run), 24 train /
12 test seeds, and the same evaluation budget per optimizer method
(1,230 evals = pop 30 × 41). All six methods see identical train/test
seeds within each run.

## Methods

- random-search, vanilla GA, guided GA (original three; RNG streams
  untouched by the extension).
- random-weights GA: guided-mutation machinery with uniform-random
  weights (fresh RNG offset) — control for non-uniformity.
- (1+1)-ES: single-parent elitist ES, Gaussian mutation σ=0.1, same
  1,230-eval budget.
- calibrated: per-edge noise amps estimated from 24 probe simulations
  via `sqrt(3*var)` on healthy-callee steps, then
  `theta = clip(amp_hat + 0.10)`. Zero optimizer evals; 24 probe sims.

## Pre-specified primary contrasts (paired test-F differences)

Positive difference favors the second method:

1. calibrated vs guided GA
2. guided GA vs vanilla GA
3. guided GA vs random-weights GA

All other pairwise comparisons are exploratory.

## Statistics

Per-method: test F mean, sample sd (ddof=1), train F (all n=3).
Paired contrasts: mean difference with bootstrap 95% CI (10,000
resamples). Win counts per pair.

## Pilot caveat

n=3, so CIs will be very wide. Treat everything as a pilot: report the
intervals honestly, make no claims beyond what the CIs support, and do
not use directional language ("beats", "better") unless the CI excludes
zero — which at n=3 it is unlikely to.

## Outputs

- `results/ablation_extended.csv` (per-run rows; never edited after writing)
- `results/ablation_extended_summary.csv`
- README "Extended baselines" table (filled only from the CSV; "results
  pending" until the run finishes)
