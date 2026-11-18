"""B4 tests: DTW with a Sakoe-Chiba band."""
import numpy as np
import pytest

from neuroflow.segmentation.dtw import band_width, dtw_distance


def test_hand_worked_example():
    x, y = [0, 1, 2, 1, 0], [0, 0, 1, 2, 1]   # y = x shifted right by one sample
    assert dtw_distance(x, y, band_frac=0) == 2.0
    assert dtw_distance(x, y, band_frac=0.2) == 1.0
    assert dtw_distance(x, y, band_frac=None) == 1.0


def test_identical_series_is_zero():
    x = np.random.default_rng(1).random(40)
    assert dtw_distance(x, x) == 0.0


def test_band_zero_is_euclidean():
    rng = np.random.default_rng(2)
    a, b = rng.random(30), rng.random(30)
    assert dtw_distance(a, b, 0) == pytest.approx(np.linalg.norm(a - b), abs=1e-12)


def test_wider_band_never_increases_distance():
    rng = np.random.default_rng(3)
    for _ in range(20):
        a, b = rng.normal(size=40), rng.normal(size=40)
        ds = [dtw_distance(a, b, f) for f in (0, 0.05, 0.1, 0.15, 0.2, None)]
        assert all(d1 >= d2 - 1e-12 for d1, d2 in zip(ds, ds[1:]))


def test_band_exposes_shifted_peak():
    # Same range, peak at 30 % instead of 50 %: a 20 % band (or none) warps it away,
    # a 10 % band cannot -- the reason the band is a P2 factor.
    u = np.linspace(0, 1, 60)
    rep = lambda v: (1 - np.cos(2 * np.pi * v)) / 2
    tmpl, early = rep(u), rep(u ** (np.log(0.5) / np.log(0.3)))
    assert dtw_distance(tmpl, early, 0.1) > 5 * dtw_distance(tmpl, early, None)


def test_nan_and_empty_rejected():
    with pytest.raises(ValueError):
        dtw_distance([1.0, np.nan, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        dtw_distance([], [1.0])


def test_matches_dtaidistance():
    ref = pytest.importorskip("dtaidistance.dtw")
    rng = np.random.default_rng(0)
    for _ in range(100):
        n, m = rng.integers(5, 60, size=2)
        p, q = rng.normal(size=n), rng.normal(size=m)
        for f in (None, 0.0, 0.1, 0.2):
            theirs = (ref.distance(p, q, use_c=False) if f is None
                      else ref.distance(p, q, window=band_width(n, m, f) + 1, use_c=False))
            assert dtw_distance(p, q, f) == pytest.approx(theirs, abs=1e-9)
