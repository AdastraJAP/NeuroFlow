"""
B2: segmenter v0 -- fixed-threshold hysteresis state machine (owner: Member B).

Rep boundary definition (team decision 2026-11-18, used by B1 truth, B2 and B9):
    start = time the angle rises through  rest + 5% of the rep's range
    end   = time the angle falls through  rest + 5% of the rep's range
    range = peak angle - rest angle, per rep
The very first movement sits below the noise and cannot be timed reliably, so it
is not used as the boundary.

Two steps:
  1. Detection. A 4-state machine (REST, RISING, PEAK, RETURNING) runs on the
     smoothed theta_HT. A rep opens when the angle rises above T_HIGH and closes
     when it falls below T_LOW. T_LOW < T_HIGH is the hysteresis gap, so noise
     near one threshold cannot open and close reps repeatedly.
  2. Boundaries. From each threshold crossing, walk outward until the smoothed
     angle drops to rest + BOUNDARY_FRAC * range, then interpolate linearly
     between the two frames on either side of that level.

Methods:
    "pct_range"  the definition above (default)
    "crossing"   raw threshold crossing times, kept only as a reference point

Thresholds 30/20 deg are v0 starting values on mocks, not literature values;
tune them on training subjects only. B3 replaces them with fractions of the
user's calibrated ROM.

Run (from the repo root):
    python -m neuroflow.segmentation.segment
"""

import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

BOUNDARY_FRAC = 0.05    # rep start/end = rest + 5% of the rep's range
T_HIGH = 30.0           # deg, enter a rep above this
T_LOW = 20.0            # deg, leave a rep below this
SMOOTH_WINDOW_S = 0.25  # Savitzky-Golay window (A5 default); swap for A's smooth_savgol later
SMOOTH_ORDER = 2
MAX_SEARCH_S = 1.5      # how far before/after a crossing to look for rest
TOLERANCE_MS = 67.0     # golden criterion: +-2 frames at 30 fps on start and end
METHODS = ("pct_range", "crossing")

REST, RISING, PEAK, RETURNING = "REST", "RISING", "PEAK", "RETURNING"


def smooth(t_ms, x, window_s=SMOOTH_WINDOW_S, order=SMOOTH_ORDER):
    """Zero-phase Savitzky-Golay smoothing. Window is set in seconds from real timestamps.
    NaN gaps are linearly interpolated for smoothing only."""
    x = np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    if ok.sum() < 3:
        return x.copy()
    x = np.interp(t_ms, t_ms[ok], x[ok])
    dt = np.median(np.diff(t_ms))
    win = int(round(window_s * 1000.0 / dt))
    win = max(win | 1, order + 2 | 1)  # odd and long enough for the polynomial
    return savgol_filter(x, win, order)


def detect(theta_s, t_high=T_HIGH, t_low=T_LOW):
    """
    Run the state machine on a smoothed angle.
    Returns a list of (i_up, i_peak, i_down) frame indices: the frame where the
    angle crossed above t_high, the highest frame of the rep, and the frame where
    it crossed back below t_low.
    """
    reps = []
    state = REST
    i_up = i_peak = None
    for i in range(1, len(theta_s)):
        x, prev = theta_s[i], theta_s[i - 1]
        if state == REST:
            if x > t_high:
                state, i_up, i_peak = RISING, i, i
        elif state == RISING:
            if x > theta_s[i_peak]:
                i_peak = i
            if x <= prev:                 # stopped rising
                state = PEAK
        elif state in (PEAK, RETURNING):
            if x > theta_s[i_peak]:      # still climbing after a wobble: same rep
                i_peak, state = i, RISING
            elif state == PEAK and x < prev:
                state = RETURNING
            if x < t_low:
                reps.append((i_up, i_peak, i))
                state = REST
    return reps


def rest_level(theta_s, lo, hi, t_low=T_LOW):
    """Median smoothed angle of the frames in [lo, hi) that are below t_low (the rest)."""
    seg = theta_s[lo:hi]
    seg = seg[seg < t_low]
    return float(np.median(seg)) if len(seg) >= 3 else None


def cross_time(t_ms, x, i_a, i_b, level):
    """Time at which x passes `level` between neighbouring frames i_a and i_b (linear)."""
    xa, xb = x[i_a], x[i_b]
    if xb == xa:
        return t_ms[i_b]
    f = (level - xa) / (xb - xa)
    return t_ms[i_a] + f * (t_ms[i_b] - t_ms[i_a])


def boundaries(theta_s, t_ms, reps, method="pct_range", frac=BOUNDARY_FRAC):
    """Return a list of (t_start_ms, t_peak_ms, t_end_ms) for each detected rep."""
    if method == "crossing":
        return [(t_ms[a], t_ms[p], t_ms[b]) for a, p, b in reps]
    if method != "pct_range":
        raise ValueError(f"unknown method {method}")

    dt = np.median(np.diff(t_ms))
    max_n = int(MAX_SEARCH_S * 1000.0 / dt)
    out = []
    for k, (i_up, i_peak, i_down) in enumerate(reps):
        prev_end = reps[k - 1][2] if k > 0 else 0
        next_up = reps[k + 1][0] if k + 1 < len(reps) else len(theta_s)
        lo = max(prev_end, i_up - max_n)
        hi = min(next_up, i_down + max_n)

        peak = theta_s[i_peak]
        rest_s = rest_level(theta_s, lo, i_up)
        rest_e = rest_level(theta_s, i_down, hi)
        if rest_s is None or rest_e is None:
            out.append((np.nan, t_ms[i_peak], np.nan))  # no rest seen: boundary unknown
            continue

        lvl_s = rest_s + frac * (peak - rest_s)
        i = i_up
        while i > lo and theta_s[i - 1] > lvl_s:
            i -= 1
        t_s = cross_time(t_ms, theta_s, i - 1, i, lvl_s) if i > lo else np.nan

        lvl_e = rest_e + frac * (peak - rest_e)
        j = i_down
        while j < hi - 1 and theta_s[j + 1] > lvl_e:
            j += 1
        t_e = cross_time(t_ms, theta_s, j, j + 1, lvl_e) if j < hi - 1 else np.nan

        out.append((t_s, t_ms[i_peak], t_e))
    return out


def segment_fixed(frames, method="pct_range", t_high=T_HIGH, t_low=T_LOW):
    """
    Segment one block of frames.parquet into reps (v0, fixed thresholds).
    Returns a DataFrame with the reps.parquet timing columns.
    """
    frames = frames.sort_values("t_capture_ms")
    t = frames["t_capture_ms"].to_numpy(dtype=float)
    theta_s = smooth(t, frames["theta_ht_deg"].to_numpy())
    reps = boundaries(theta_s, t, detect(theta_s, t_high, t_low), method)

    first = frames.iloc[0]
    rows = []
    for n, (t_s, t_p, t_e) in enumerate(reps):
        rows.append({
            "rep_id": f"{first['subject_id']}_{first['session_id']}_{first['block_id']}_{n:03d}",
            "subject_id": first["subject_id"],
            "session_id": first["session_id"],
            "condition": first["condition"],
            "level": first["level"],
            "seg_method": f"fsm_fixed_{method}",
            "t_start_ms": t_s,
            "t_peak_ms": t_p,
            "t_end_ms": t_e,
            "duration_s": (t_e - t_s) / 1000.0,
        })
    return pd.DataFrame(rows)


def compare_to_truth(reps, truth):
    """Match each true rep to the detected rep with the nearest peak; return signed errors (ms)."""
    if len(reps) == 0:
        return None
    d_peak = reps["t_peak_ms"].to_numpy()
    rows = []
    for _, r in truth.iterrows():
        j = int(np.argmin(np.abs(d_peak - r["t_peak_ms"])))
        rows.append({
            "err_start_ms": reps["t_start_ms"].iloc[j] - r["t_start_ms"],
            "err_peak_ms": reps["t_peak_ms"].iloc[j] - r["t_peak_ms"],
            "err_end_ms": reps["t_end_ms"].iloc[j] - r["t_end_ms"],
        })
    return pd.DataFrame(rows)


def validate(frames, truth, method="pct_range"):
    """Return (passed, n_found, errors DataFrame) for one mock against its truth file."""
    reps = segment_fixed(frames, method=method)
    err = compare_to_truth(reps, truth)
    if err is None or len(reps) != len(truth):
        return False, len(reps), err
    worst = max(err["err_start_ms"].abs().max(), err["err_end_ms"].abs().max())
    return bool(worst <= TOLERANCE_MS), len(reps), err   # NaN worst -> False


def main():
    fx = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures")
    variants = ["normal", "slow", "capped", "leaning"]

    # Unit check: a clean synthetic rep is found once, with boundaries at the 5% points.
    t = np.arange(0, 6000, 1000 / 30)
    x = 10 + 80 * (1 - np.cos(2 * np.pi * np.clip((t - 2000) / 2000, 0, 1))) / 2
    (ts, _, te), = boundaries(x, t, detect(x))
    u = np.arccos(1 - 2 * BOUNDARY_FRAC) / (2 * np.pi)
    assert abs(ts - (2000 + u * 2000)) < 5 and abs(te - (4000 - u * 2000)) < 5, (ts, te)

    print(f"T_high = {T_HIGH} deg, T_low = {T_LOW} deg, boundary = rest + "
          f"{BOUNDARY_FRAC:.0%} of range, tolerance = +-{TOLERANCE_MS:.0f} ms\n")
    print(f"{'variant':<8} {'method':<10} {'reps':>6} {'start err ms':>23} {'end err ms':>23}  pass")
    print(f"{'':<8} {'':<10} {'':>6} {'mean':>7} {'sd':>7} {'max|e|':>7} "
          f"{'mean':>7} {'sd':>7} {'max|e|':>7}")
    print("-" * 82)
    ok_all = {m: True for m in METHODS}
    for v in variants:
        frames = pd.read_parquet(os.path.join(fx, f"mock_frames_{v}.parquet"))
        truth = pd.read_csv(os.path.join(fx, f"mock_frames_{v}_truth.csv"))
        for m in METHODS:
            ok, n, err = validate(frames, truth, m)
            ok_all[m] &= ok
            if err is None:
                print(f"{v:<8} {m:<10} {n:>3}/{len(truth):<2}  no reps found")
                continue
            s, e = err["err_start_ms"], err["err_end_ms"]
            print(f"{v:<8} {m:<10} {n:>3}/{len(truth):<2} {s.mean():>+7.1f} {s.std():>7.1f} "
                  f"{s.abs().max():>7.1f} {e.mean():>+7.1f} {e.std():>7.1f} {e.abs().max():>7.1f}"
                  f"  {'yes' if ok else 'NO'}")
        print()
    print("passes on all four mocks: " +
          ", ".join(f"{m}={'yes' if p else 'no'}" for m, p in ok_all.items()))


if __name__ == "__main__":
    main()
