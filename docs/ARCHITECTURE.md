# Architecture: Python core and JS web demo

## The split

The repository has two implementations of the same model:

1. **Python core** (`src/`, `main.py`) — the source of truth. NumPy +
   NetworkX. All experiments, CSVs, and claims come from here.
2. **JavaScript demo** (`../epidemic-breakers-web/`) — a separate repo
   containing a static site. It **re-implements** the simulator, fitness,
   edge-betweenness, and a small GA in vanilla JS (`sim.js`), and ships
   **precomputed data exported from the Python core** (`data/*.json`):
   the exact 40-node / 111-edge graph, per-edge noise/spread parameters,
   and the GA-optimized threshold vector.

The JS demo is a visualization and interaction layer, not a second
source of truth. It never generates claims for the paper/README.

## Parity approach

`scripts/check_web_parity.py` verifies the baked-in JS constants agree
with the Python source of truth:

- 40 nodes / 111 edges, identical edge identity **and** order
- Per-edge noise/spread parameters identical
- `best_theta.json` identical to the exported numpy vector
- Simulator default constants match (cooldown, steps, EMA alpha)

**Known, documented differences (not failures):**

- The JS RNG (mulberry32/cyrb53) differs from NumPy's PCG64, so
  single-scenario trajectories differ for the same seed. Parity is
  **statistical**: mean-F over many seeds agrees within tolerance
  (±2.0), verified by the parity test suite (`node --test` in the web
  repo, 14 tests green).
- The web demo's in-browser GA is a simplified port for interactivity,
  not the full Python optimizer.

## Data flow

```
Python core                    Web demo
───────────                    ────────
graph_gen.py ──export──> data/graph.json
simulator params ──export──> data/edge_params.json
make_best_theta.py ──export──> data/best_theta.json
                                     │
                                     v
                              sim.js (re-implementation)
                                     │
                                     v
                              interactive site
```

Exports are one-way: Python → JSON → JS. The JS side never writes back.
