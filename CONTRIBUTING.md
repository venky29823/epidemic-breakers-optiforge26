# Contributing to Epidemic Breakers

## Running tests

```bash
pip install -r requirements.txt
pytest tests/ -q
```

All 35 tests must pass before a pull request. Tests cover the simulator,
fitness computation, GA operators, and regression defaults (see
`tests/test_regression_defaults.py`).

## Code style

- Follow PEP 8. Run `ruff check .` (or `flake8`) before committing.
- Add type hints to public functions.
- Docstrings: purpose, args, returns for every public function/class.
- Small, conventional commits (`feat:`, `fix:`, `docs:`, `test:`).

## Adding a baseline

1. Implement it in `src/baselines.py` (or a new `src/<name>.py` module)
   following the existing signature: `(graph, train_seeds, test_seeds,
   **sim_kwargs) -> dict` with `"test"` sub-dict from `fitness.evaluate`.
2. Wire it into `main.py` (comparison, ablation, or `--extended`).
3. Add a test in `tests/` covering determinism (same seed → same output).
4. Document it in the README under "Extensions and baselines" — do not
   touch the main-study sections.
5. Never edit files in `results/` by hand; they are generated.

## Reproducibility rules

- All experiments use seeded RNGs (`src/utils.py`). Never use unseeded
  randomness in experiment code.
- Train and test seeds must be disjoint (see `main.py`).
- Every number in the README must cite its source CSV.
