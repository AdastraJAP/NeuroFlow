"""
B4: dynamic time warping with a Sakoe-Chiba band (owner: Member B).

DTW finds the cheapest way to line up two series when one may run faster or
slower in places. It fills a cost table D where
    D[i, j] = (x[i] - y[j])^2 + min(D[i-1, j-1], D[i-1, j], D[i, j-1])
and the distance is sqrt(D[n-1, m-1]) (same convention as dtaidistance, so the
two can be compared exactly).

The Sakoe-Chiba band only allows cells near the diagonal, so the warping can
shift a point by at most w samples:
    w = round(band_frac * max(n, m))
    allowed offsets  -(w + max(0, n - m))  <=  j - i  <=  w + max(0, m - n)
    i.e. +-w around the diagonal, plus the length difference on the side where
    the longer series needs room (otherwise no path from start to end exists).
    This is the same band rule as dtaidistance (window = w + 1 there).
    band_frac = None -> unconstrained DTW
    band_frac = 0    -> diagonal only = plain Euclidean distance (equal lengths)
A narrower band can only keep the distance the same or raise it.

Worked example (squared cost), x = [0, 1, 2, 1, 0], y = [0, 0, 1, 2, 1]:
    diagonal only (band 0):  diffs 0, 1, 1, -1, -1  -> sqrt(4) = 2
    band 0.2 (w = 1):  path (0,0)(0,1)(1,2)(2,3)(3,4)(4,4) costs 0+0+0+0+0+1 -> 1
    unconstrained:     same path -> 1
y is x shifted right by one sample; one sample of warping undoes the shift.

Pitfall (plan, section 6): on raw degrees the distance is dominated by
amplitude. Callers (B5/B6) should compare theta_HT divided by the personal
peak and keep amplitude as its own feature. This module just measures.

Run (from the repo root):
    python -m neuroflow.segmentation.dtw
"""

import math

import numpy as np


def band_width(n, m, band_frac):
    """Base half-width w of the band for series of lengths n and m (None = no limit)."""
    if band_frac is None:
        return max(n, m)
    return int(round(band_frac * max(n, m)))


def dtw_distance(x, y, band_frac=0.15):
    """
    DTW distance between two 1-D series with a Sakoe-Chiba band.
    Returns sqrt of the summed squared differences along the best path.
    NaN in either series is not allowed (fill or drop gaps first).
    """
    D = _cost_table(x, y, band_frac)
    return math.sqrt(D[-1, -1])


def dtw_path(x, y, band_frac=0.15):
    """
    Like dtw_distance, but also returns the best warping path as a list of
    (i, j) index pairs from (0, 0) to (n-1, m-1). Ties prefer the diagonal step.
    Used by DBA (templates.py) to see which samples line up.
    """
    D = _cost_table(x, y, band_frac)
    i, j = D.shape[0] - 1, D.shape[1] - 1
    path = [(i - 1, j - 1)]
    while (i, j) != (1, 1):
        steps = ((D[i - 1, j - 1], i - 1, j - 1),   # diagonal first, so ties go diagonal
                 (D[i - 1, j], i - 1, j),
                 (D[i, j - 1], i, j - 1))
        _, i, j = min(steps, key=lambda s: s[0])
        path.append((i - 1, j - 1))
    return math.sqrt(D[-1, -1]), path[::-1]


def _cost_table(x, y, band_frac):
    """Accumulated cost table D (with one padding row/column) for x vs y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        raise ValueError("empty series")
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("NaN in input; fill or drop gaps before DTW")

    w = band_width(n, m, band_frac)
    below = w + max(0, n - m)   # how far j may fall behind i
    above = w + max(0, m - n)   # how far j may run ahead of i
    D = np.full((n + 1, m + 1), np.inf)   # one row/column of padding at index 0
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo = max(1, i - below)
        j_hi = min(m, i + above)
        for j in range(j_lo, j_hi + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    return D


def main():
    # Hand-worked example from the docstring
    x = [0, 1, 2, 1, 0]
    y = [0, 0, 1, 2, 1]
    assert dtw_distance(x, y, band_frac=0) == 2.0
    assert dtw_distance(x, y, band_frac=0.2) == 1.0
    assert dtw_distance(x, y, band_frac=None) == 1.0
    assert dtw_distance(x, x) == 0.0
    print("hand example: band 0 -> 2.0, band 0.2 -> 1.0, unconstrained -> 1.0  (ok)")

    # Band 0 on equal lengths is the Euclidean distance
    rng = np.random.default_rng(0)
    a, b = rng.random(30), rng.random(30)
    assert abs(dtw_distance(a, b, 0) - np.linalg.norm(a - b)) < 1e-12
    print("band 0 equals Euclidean distance  (ok)")

    # Wider band never increases the distance
    ds = [dtw_distance(a, b, f) for f in (0, 0.05, 0.1, 0.15, 0.2, None)]
    assert all(d1 >= d2 - 1e-12 for d1, d2 in zip(ds, ds[1:]))
    print("distance by band 0 / 5 / 10 / 15 / 20 % / none: " + ", ".join(f"{d:.3f}" for d in ds))

    # Agreement with dtaidistance (reference library), if installed
    try:
        from dtaidistance import dtw as ref
    except ImportError:
        print("dtaidistance not installed; skipped the reference comparison")
        return
    worst = 0.0
    for k in range(200):
        n, m = rng.integers(5, 60, size=2)
        p, q = rng.normal(size=n), rng.normal(size=m)
        for f in (None, 0.0, 0.1, 0.2):
            mine = dtw_distance(p, q, f)
            if f is None:
                theirs = ref.distance(p, q, use_c=False)
            else:
                theirs = ref.distance(p, q, window=band_width(n, m, f) + 1, use_c=False)
            worst = max(worst, abs(mine - theirs))
    assert worst < 1e-9, worst
    print(f"matches dtaidistance on 200 random pairs x 4 bands, max difference {worst:.1e}  (ok)")


if __name__ == "__main__":
    main()
