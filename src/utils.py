"""Shared helpers: seeding, CSV output, and a dependency-free SVG line chart."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Create a seeded numpy random generator (no global state)."""
    return np.random.default_rng(seed)


def paired_bootstrap_ci(
    diffs: Sequence[float],
    n_boot: int = 10000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Paired bootstrap confidence interval for the mean difference.

    Resamples run indices with replacement ``n_boot`` times, recomputes the
    mean difference each time, and returns ``(mean, lower, upper)`` from the
    percentile interval. Pure numpy -- no scipy. Deterministic given ``rng``.
    """
    d = np.asarray(diffs, dtype=float)
    if d.size == 0:
        raise ValueError("need at least one difference")
    rng = rng if rng is not None else np.random.default_rng()
    boots = np.array([d[rng.integers(0, d.size, size=d.size)].mean()
                      for _ in range(n_boot)])
    alpha = 1.0 - ci
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(d.mean()), lo, hi


def make_seeds(n: int, base: int) -> list[int]:
    """Generate n deterministic scenario seeds from a base seed."""
    rng = np.random.default_rng(base)
    return [int(x) for x in rng.integers(0, 2**31 - 1, size=n)]


def get_logger(name: str = "breakers") -> logging.Logger:
    """Return a configured logger (INFO level, timestamped stderr output).

    Args:
        name: Logger name; reuses the existing logger if already configured.

    Returns:
        The configured `logging.Logger`.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (including parents) if it does not exist.

    Args:
        path: Directory path to ensure.

    Returns:
        The path as a `Path` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(path: str | Path, rows: Sequence[Mapping], fieldnames: Sequence[str]) -> Path:
    """Write a list of dicts to CSV."""
    p = Path(path)
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return p


def save_convergence_svg(
    path: str | Path,
    series: dict[str, list[float]],
    title: str = "GA convergence (best train F per generation)",
    xlabel: str = "generation",
    ylabel: str = "best train F (lower is better)",
) -> Path:
    """Write a minimal multi-series line chart as SVG (stdlib only, no matplotlib)."""
    width, height = 680, 420
    margin = {"l": 70, "r": 20, "t": 40, "b": 55}
    iw, ih = width - margin["l"] - margin["r"], height - margin["t"] - margin["b"]

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    all_y = [y for vals in series.values() for y in vals]
    all_x = max((len(v) for v in series.values()), default=1)
    ymin, ymax = min(all_y), max(all_y)
    pad = (ymax - ymin) * 0.08 or 1.0
    ymin, ymax = ymin - pad, ymax + pad

    def sx(i: int) -> float:
        return margin["l"] + (i / max(1, all_x - 1)) * iw

    def sy(v: float) -> float:
        return margin["t"] + ih - ((v - ymin) / (ymax - ymin)) * ih

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width/2}" y="22" text-anchor="middle" font-size="15" '
        f'font-family="sans-serif">{title}</text>',
        f'<text x="{margin["l"] - 50}" y="{margin["t"] + ih/2}" text-anchor="middle" '
        f'font-size="12" font-family="sans-serif" '
        f'transform="rotate(-90 {margin["l"] - 50} {margin["t"] + ih/2})">{ylabel}</text>',
        f'<text x="{margin["l"] + iw/2}" y="{height - 12}" text-anchor="middle" '
        f'font-size="12" font-family="sans-serif">{xlabel}</text>',
    ]
    # y gridlines + labels
    for gi in range(5):
        v = ymin + (ymax - ymin) * gi / 4
        y = sy(v)
        parts.append(
            f'<line x1="{margin["l"]}" y1="{y}" x2="{margin["l"] + iw}" y2="{y}" '
            f'stroke="#e0e0e0"/>'
            f'<text x="{margin["l"] - 8}" y="{y + 4}" text-anchor="end" font-size="10" '
            f'font-family="sans-serif">{v:.2f}</text>'
        )
    for si, (name, vals) in enumerate(series.items()):
        color = colors[si % len(colors)]
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals))
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<rect x="{margin["l"] + 8 + si * 150}" y="{margin["t"] - 28}" width="12" '
            f'height="12" fill="{color}"/>'
            f'<text x="{margin["l"] + 24 + si * 150}" y="{margin["t"] - 18}" '
            f'font-size="12" font-family="sans-serif">{name}</text>'
        )
    parts.append("</svg>")
    p = Path(path)
    p.write_text("\n".join(parts))
    return p
