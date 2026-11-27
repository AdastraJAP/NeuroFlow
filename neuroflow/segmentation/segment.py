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
     The rest level comes only from where the arm is still (still_rest). If
     the clip ends while the arm is still coming down, the end uses the rest
     before the rep; if that level isn't reached before the clip ends, the end
     is unknown (NaN): the rep still counts, its end is not guessed.

Methods:
    "pct_range"  the definition above (default)
    "crossing"   raw threshold crossing times, kept only as a reference point

Thresholds 30/20 deg are v0 starting values on mocks, not literature values;
tune them on training subjects only.

B3: segmenter v1 -- ROM-normalised thresholds (S3 core), segment_reps().
Same state machine and the same 5% boundary rule, but the thresholds scale with
the user's calibrated range instead of being fixed degrees:
    T_high = rest_ref + START_FRAC * rom_ref   (20%)
    T_low  = rest_ref + END_FRAC   * rom_ref   (15%)
plus two filters on each detected rep:
    minimum prominence: peak - rest_ref >= MIN_PROM_FRAC * rom_ref   (30%)
    minimum duration:   end - start      >= MIN_DUR_S                (0.4 s)
rest_ref and rom_ref come from calibrate() on a separate calibration block (in
the protocol: the Normal block of session 1). All four fractions/limits are
design inferences, not literature values: tune them on training subjects only.
A fixed 30 deg (or Pavlikov's 55 deg) threshold never fires for a user whose
whole range stays below it; fractions of their own range always can.

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
MAX_GAP_S = 0.2         # tracking gaps: measured on the real timestamps, as the time
                        # between the valid samples on either side of the missing ones
                        # (or of a jump in capture time). Up to this long they are
                        # bridged for smoothing; longer ones stay NaN, are never
                        # interpolated across, and a rep that spans one is not
                        # reported (issue #12, PR #15 review). Our choice, not a
                        # literature value.
CALIB_MIN_REPS = 3      # calibrate() needs at least this many fully measured reps (our
                        # choice: G1's practice calibration has exactly 3)
REST_STABLE_S = 0.5     # rest must be still over this long (the part farthest from the rep)
REST_MAX_SLOPE = 3.0    # deg/s; faster than this the arm is still moving, not at rest
                        # (G1 clips end while still falling at about 10 deg/s)
TOLERANCE_MS = 67.0     # golden criterion: +-2 frames at 30 fps on start and end
METHODS = ("pct_range", "crossing")

# B3 (ROM-normalised) settings -- fractions of the calibrated range rom_ref
START_FRAC = 0.20       # open a rep above rest + 20% of rom_ref
END_FRAC = 0.15         # close it below rest + 15% of rom_ref
MIN_PROM_FRAC = 0.30    # a rep must rise at least 30% of rom_ref above rest
MIN_DUR_S = 0.4         # and last at least 0.4 s between its boundaries

REST, RISING, PEAK, RETURNING = "REST", "RISING", "PEAK", "RETURNING"

# Which elevation angle to segment on (agreed with A in issue #4, 2026-11-21):
#   theta_ht_deg  arm vs trunk (needs hip landmarks); the default, used by the mocks
#   theta_v_deg   arm vs image vertical (hip-free; A's V4 seated clips). Trunk lean
#                 adds to it, so it reads higher when the user leans.
ANGLE_COL = "theta_ht_deg"
ANGLE_COLS = ("theta_ht_deg", "theta_v_deg")


def angle_signal(frames, angle_col=ANGLE_COL):
    """(frames sorted by time, t_ms, raw angle) for the chosen angle column.
    Fails loudly if the column is unknown, missing or has no values, instead of
    letting the segmenter quietly find 0 reps."""
    if angle_col not in ANGLE_COLS:
        raise ValueError(f"angle_col must be one of {ANGLE_COLS}, got {angle_col!r}")
    if angle_col not in frames.columns or frames[angle_col].isna().all():
        raise ValueError(f"{angle_col} has no values in these frames; "
                         f"try another angle_col (one of {ANGLE_COLS})")
    frames = frames.sort_values("t_capture_ms")
    return frames, frames["t_capture_ms"].to_numpy(dtype=float), frames[angle_col].to_numpy(dtype=float)


def reps_angle_col(reps, angle_col=None):
    """Angle column for downstream steps: the one the reps were segmented on.
    Passing a different one raises, so two angle definitions are never mixed.
    Read from the angle_col column, or, for a table with 0 reps, from its metadata
    (reps.attrs). An empty table that records neither needs angle_col passed."""
    recorded = reps["angle_col"].dropna().unique() if "angle_col" in reps.columns else []
    if len(recorded) == 0 and reps.attrs.get("angle_col"):
        recorded = [reps.attrs["angle_col"]]
    if len(recorded) > 1:
        raise ValueError(f"reps mix angle columns {list(recorded)}")
    if len(recorded) == 1:
        if angle_col is not None and angle_col != recorded[0]:
            raise ValueError(f"reps were segmented on {recorded[0]}, not {angle_col}")
        return recorded[0]
    if angle_col is None and len(reps) == 0:
        raise ValueError("this reps table has 0 reps and doesn't record its angle; "
                         f"pass angle_col (one of {ANGLE_COLS})")
    return angle_col or ANGLE_COL


def _long_intervals(t_ms, x, max_gap_s=MAX_GAP_S):
    """Indices v of the valid samples, and the positions k where the real time
    from valid sample v[k] to the next one, v[k + 1], is longer than max_gap_s."""
    v = np.flatnonzero(~np.isnan(np.asarray(x, dtype=float)))
    dt = np.diff(np.asarray(t_ms, dtype=float)[v])
    return v, np.flatnonzero(dt > max_gap_s * 1000.0)


def long_gaps(t_ms, x, max_gap_s=MAX_GAP_S):
    """
    True for missing frames that must stay NaN:
      - inside a long gap: the valid samples before and after them are more than
        max_gap_s apart in real time (timestamps, not a median frame period, so
        irregular capture timing can't hide a long gap);
      - before the first or after the last valid sample (nothing on one side to
        interpolate from).
    """
    x = np.asarray(x, dtype=float)
    out = np.isnan(x)
    v, long = _long_intervals(t_ms, x, max_gap_s)
    if len(v) == 0:
        return out
    keep = np.zeros(len(x), dtype=bool)                 # missing frames inside short gaps
    short = np.setdiff1d(np.arange(len(v) - 1), long)
    for k in short:
        keep[v[k] + 1:v[k + 1]] = True
    return out & ~keep


def gap_breaks(t_ms, x, max_gap_s=MAX_GAP_S):
    """Frame indices where a new stretch of measurement starts after a long gap
    (missing frames or a jump in capture time longer than max_gap_s)."""
    v, long = _long_intervals(t_ms, x, max_gap_s)
    return v[long + 1]


def smooth(t_ms, x, window_s=SMOOTH_WINDOW_S, order=SMOOTH_ORDER):
    """Zero-phase Savitzky-Golay smoothing. Window is set in seconds from real timestamps.
    Short NaN gaps (<= MAX_GAP_S between the valid samples around them) are
    linearly interpolated for smoothing only. The signal is cut at every long gap
    (see long_gaps / gap_breaks) and each piece, from its first to its last valid
    sample, is smoothed separately: no value is invented across a long gap or
    before / after the measured samples (issue #12, PR #15 review)."""
    x = np.asarray(x, dtype=float)
    t_ms = np.asarray(t_ms, dtype=float)
    out = np.full(len(x), np.nan)
    if (~np.isnan(x)).sum() < 3:
        return x.copy()
    dt = np.median(np.diff(t_ms))
    win = int(round(window_s * 1000.0 / dt))
    win = max(win | 1, order + 2 | 1)  # odd and long enough for the polynomial
    v, long = _long_intervals(t_ms, x)
    starts = np.r_[v[0], v[long + 1]]
    ends = np.r_[v[long], v[-1]] + 1                                # exclusive
    for a, b in zip(starts, ends):                                 # first..last valid of a piece
        tt, xx = t_ms[a:b], x[a:b]
        ok = ~np.isnan(xx)
        if ok.sum() < 3:
            out[a:b] = xx
            continue
        xx = np.interp(tt, tt[ok], xx[ok])
        w = min(win, len(xx) if len(xx) % 2 else len(xx) - 1)
        out[a:b] = savgol_filter(xx, w, order) if w > order else xx
    return out


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


def still_rest(theta_s, t_ms, lo, hi, t_low=T_LOW, side="after"):
    """
    Rest level next to a rep, from where the arm is actually still: the
    REST_STABLE_S stretch of rest frames (below t_low) in [lo, hi) that is
    nearest to the rep and whose slope is at most REST_MAX_SLOPE. side="after"
    scans away from the rep forwards ([lo, hi) starts at the rep's end), "before"
    backwards. Returns the median of that stretch, or None if there is none
    (e.g. the clip ends while the arm is still coming down).
    """
    idx = np.arange(lo, hi)
    idx = idx[theta_s[lo:hi] < t_low]
    if len(idx) < 3:
        return None
    if side == "before":
        idx = idx[::-1]
    span = REST_STABLE_S * 1000.0
    dt = np.median(np.diff(t_ms))
    tt = t_ms[idx]
    for k in range(len(idx)):
        win = np.abs(tt - tt[k]) <= span
        win &= (tt >= tt[k]) if side == "after" else (tt <= tt[k])
        seg = idx[win]
        if len(seg) < 3 or abs(t_ms[seg[-1]] - t_ms[seg[0]]) < span - 1.5 * dt:
            break                                   # ran out of frames
        if np.any(np.abs(np.diff(seg)) != 1):
            continue                                # stretch crosses a non-rest frame
        if abs(np.polyfit(t_ms[seg] / 1000.0, theta_s[seg], 1)[0]) <= REST_MAX_SLOPE:
            return float(np.median(theta_s[seg]))
    return None


def cross_time(t_ms, x, i_a, i_b, level):
    """Time at which x passes `level` between neighbouring frames i_a and i_b (linear)."""
    xa, xb = x[i_a], x[i_b]
    if xb == xa:
        return t_ms[i_b]
    f = (level - xa) / (xb - xa)
    return t_ms[i_a] + f * (t_ms[i_b] - t_ms[i_a])


def boundaries(theta_s, t_ms, reps, method="pct_range", frac=BOUNDARY_FRAC, t_low=T_LOW):
    """Return a list of (t_start_ms, t_peak_ms, t_end_ms) for each detected rep.
    t_low is the detection's closing threshold; frames below it count as rest."""
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
        # Rest only from where the arm is actually still, looking as far as the
        # neighbouring rep (or the clip edge). If one side never settles (clip
        # ends while the arm is still coming down), use the other side's rest;
        # if neither does, the boundaries are unknown.
        rest_s = still_rest(theta_s, t_ms, prev_end, i_up, t_low, side="before")
        rest_e = still_rest(theta_s, t_ms, i_down, next_up, t_low, side="after")
        if rest_s is None and rest_e is None:
            out.append((np.nan, t_ms[i_peak], np.nan))  # no still rest seen: unknown
            continue
        rest_s = rest_e if rest_s is None else rest_s
        rest_e = rest_s if rest_e is None else rest_e

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


REPS_SCHEMA_VERSION = 1
# Same names and order as 7willA's reps contract (fixtures/make_mock_reps.py), plus
# angle_col (which elevation angle was segmented; proposed in PR #3).
REP_COLUMNS = ["schema_version", "rep_id", "subject_id", "session_id", "block_id",
               "condition", "level", "side", "seg_method", "angle_col",
               "t_start_ms", "t_peak_ms", "t_end_ms", "duration_s"]


def _rep_table(frames, reps, seg_method, angle_col=ANGLE_COL):
    """Build the reps.parquet timing columns from (t_start, t_peak, t_end) tuples."""
    first = frames.iloc[0]
    rows = []
    for n, (t_s, t_p, t_e) in enumerate(reps):
        rows.append({
            "schema_version": REPS_SCHEMA_VERSION,
            "rep_id": f"{first['subject_id']}_{first['session_id']}_{first['block_id']}_{n:03d}",
            "subject_id": first["subject_id"],
            "session_id": first["session_id"],
            "block_id": first["block_id"],
            "condition": first["condition"],
            "level": first["level"],
            "side": first.get("side", np.nan),      # working arm, from frames.parquet
            "seg_method": seg_method,
            "angle_col": angle_col,
            "t_start_ms": t_s,
            "t_peak_ms": t_p,
            "t_end_ms": t_e,
            "duration_s": (t_e - t_s) / 1000.0,
        })
    out = pd.DataFrame(rows, columns=REP_COLUMNS)
    # Also keep the angle in the table's metadata: a block with 0 reps has no
    # angle_col values, and downstream must not fall back to the default angle.
    # (Survives .copy() and to_parquet / read_parquet.)
    out.attrs["angle_col"] = angle_col
    return out


def _run(frames, t_high, t_low, method, angle_col=ANGLE_COL):
    """Smooth, detect and place boundaries. Returns (frames, reps, peak angles,
    number of reps dropped because they span a long tracking gap)."""
    frames, t, theta = angle_signal(frames, angle_col)
    theta_s = smooth(t, theta)
    det = detect(theta_s, t_high, t_low)
    gap, breaks = long_gaps(t, theta), gap_breaks(t, theta)
    kept = [d for d in det                                    # peak/crossings not measured
            if not gap[d[0]:d[2] + 1].any() and not ((breaks > d[0]) & (breaks <= d[2])).any()]
    reps = boundaries(theta_s, t, kept, method, t_low=t_low)
    return frames, reps, [float(theta_s[p]) for _, p, _ in kept], len(det) - len(kept)


def segment_fixed(frames, method="pct_range", t_high=T_HIGH, t_low=T_LOW, angle_col=ANGLE_COL):
    """
    B2: segment one block of frames.parquet into reps with fixed thresholds (deg).
    Returns a DataFrame with the reps.parquet timing columns.
    """
    frames, reps, _, n_gap = _run(frames, t_high, t_low, method, angle_col)
    out = _rep_table(frames, reps, f"fsm_fixed_{method}", angle_col)
    out.attrs["dropped_gap_reps"] = n_gap
    return out


def calibrate(frames, angle_col=ANGLE_COL):
    """
    Estimate (rest_ref, rom_ref) in degrees from a calibration block (~10 normal reps).
    Self-scaling, so it also works for a user with a small range:
      1. rest = 10th percentile of the smoothed angle (most frames are at rest);
         rough range = 98th percentile - rest. Measured frames only: long tracking
         gaps stay NaN (issue #12) and are left out, never filled.
      2. detect reps with thresholds at 50% / 40% of that rough range; a rep that
         spans a long gap (as in segment_reps) is left out, since its peak may not
         have been measured;
      3. rom_ref = median peak of the remaining reps - rest.
    Long invalid stretches (PR #15 review): if fewer than CALIB_MIN_REPS fully
    measured reps remain, calibration raises, saying how many reps were found and
    how many long gaps removed. It never falls back to partial reps or guesses a
    range: record the calibration block again.
    """
    _, t, theta = angle_signal(frames, angle_col)
    theta_s = smooth(t, theta)
    if np.isnan(theta_s).all():
        raise ValueError("calibration block has no measured frames")
    rest = float(np.nanpercentile(theta_s, 10))
    rough = float(np.nanpercentile(theta_s, 98)) - rest
    det = detect(theta_s, rest + 0.5 * rough, rest + 0.4 * rough)
    gap, breaks = long_gaps(t, theta), gap_breaks(t, theta)
    kept = [d for d in det
            if not gap[d[0]:d[2] + 1].any() and not ((breaks > d[0]) & (breaks <= d[2])).any()]
    if len(kept) < CALIB_MIN_REPS:
        raise ValueError(f"calibration block has {len(kept)} fully measured reps, needs "
                         f"{CALIB_MIN_REPS} ({len(det)} detected, {len(det) - len(kept)} left "
                         "out because they span a long tracking gap); record it again")
    rom = float(np.median([theta_s[p] for _, p, _ in kept])) - rest
    return rest, rom


def segment_reps(frames, rom_ref, rest_ref=None, method="pct_range", angle_col=ANGLE_COL):
    """
    B3: segment one block with thresholds scaled to the user's calibrated range.
    rom_ref, rest_ref come from calibrate() on a separate calibration block, on the
    same angle_col; if rest_ref is None it is estimated from this block (10th percentile).
    """
    if rest_ref is None:
        _, _, theta = angle_signal(frames, angle_col)
        rest_ref = float(np.nanpercentile(theta, 10))
    t_high = rest_ref + START_FRAC * rom_ref
    t_low = rest_ref + END_FRAC * rom_ref
    frames, reps, peaks, n_gap = _run(frames, t_high, t_low, method, angle_col)

    keep = []
    for (t_s, t_p, t_e), pk in zip(reps, peaks):
        if pk - rest_ref < MIN_PROM_FRAC * rom_ref:
            continue                                    # too small to be a rep
        if not np.isnan(t_s) and not np.isnan(t_e) and (t_e - t_s) / 1000.0 < MIN_DUR_S:
            continue                                    # too short to be a rep
        keep.append((t_s, t_p, t_e))
    out = _rep_table(frames, keep, f"fsm_rom_{method}", angle_col)
    out.attrs["dropped_gap_reps"] = n_gap      # spanned a long tracking gap: not reported
    return out


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


def validate(frames, truth, method="pct_range", reps=None):
    """Return (passed, n_found, errors DataFrame) for one mock against its truth file.
    By default runs B2 (segment_fixed); pass `reps` to validate another segmenter's output."""
    if reps is None:
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

    # ---- B3: v0 (fixed) vs v1 (ROM-normalised) on all five mocks ----
    calib = calibration_refs()
    print(f"\nB3: ROM-normalised thresholds start {START_FRAC:.0%} / end {END_FRAC:.0%} of rom_ref, "
          f"min prominence {MIN_PROM_FRAC:.0%}, min duration {MIN_DUR_S} s")
    for subj, (rest, rom) in calib.items():
        print(f"  calibration {subj}: rest_ref {rest:.1f} deg, rom_ref {rom:.1f} deg "
              f"-> T_high {rest + START_FRAC * rom:.1f}, T_low {rest + END_FRAC * rom:.1f}")
    print(f"\n{'variant':<8} {'segmenter':<10} {'reps':>6} {'max|start err|':>15} "
          f"{'max|end err|':>13}  pass")
    print("-" * 60)
    for v in variants + ["low_rom"]:
        frames = pd.read_parquet(os.path.join(fx, f"mock_frames_{v}.parquet"))
        truth = pd.read_csv(os.path.join(fx, f"mock_frames_{v}_truth.csv"))
        rest, rom = calib[frames["subject_id"].iloc[0]]
        for name, reps in (("v0 fixed", segment_fixed(frames)),
                           ("v1 rom", segment_reps(frames, rom, rest))):
            ok, n, err = validate(frames, truth, reps=reps)
            if err is None:
                print(f"{v:<8} {name:<10} {n:>3}/{len(truth):<2} {'-':>15} {'-':>13}  NO")
                continue
            print(f"{v:<8} {name:<10} {n:>3}/{len(truth):<2} {err['err_start_ms'].abs().max():>15.1f} "
                  f"{err['err_end_ms'].abs().max():>13.1f}  {'yes' if ok else 'NO'}")


def calibration_refs():
    """(rest_ref, rom_ref) per mock subject, from a separate calibration block
    (a Normal block generated with a different seed, never the block under test)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "fixtures"))
    from make_mock_frames import make_session
    return {
        "mock01": calibrate(make_session("normal", seed=7)[0]),
        "mock02": calibrate(make_session("low_rom", seed=7)[0]),
    }


if __name__ == "__main__":
    main()
