"""Tests for utils: seeding helpers and the paired bootstrap CI."""
import pytest

from src.utils import make_rng, paired_bootstrap_ci


def test_bootstrap_ci_deterministic_and_sane():
    diffs = [1.0, 2.0, 3.0, 4.0, 5.0]
    a = paired_bootstrap_ci(diffs, n_boot=2000, rng=make_rng(1))
    b = paired_bootstrap_ci(diffs, n_boot=2000, rng=make_rng(1))
    assert a == b  # deterministic given the rng
    mean, lo, hi = a
    assert mean == pytest.approx(3.0)
    assert lo <= mean <= hi
    assert lo < hi


def test_bootstrap_ci_empty_raises():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([])
