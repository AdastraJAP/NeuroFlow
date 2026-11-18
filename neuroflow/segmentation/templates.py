"""
B5: rep templates (owner: Member B).

Step 1 (this file so far): linear-stretch mean template, the RehabHub-style
reference.
  1. Cut each rep out of the smoothed theta_HT at its start/end times.
  2. Normalise to the user's range: (theta - rest_ref) / rom_ref, so rest = 0 and
     the user's typical peak = 1 (plan pitfall: keeps DTW from being dominated by
     amplitude; amplitude stays a separate feature in B6). This is per-person, not
     per-rep, so a capped rep still has a smaller amplitude than a full one.
  3. Linearly stretch every rep to TEMPLATE_LEN samples (np.interp on real
     timestamps). After this, overall rep duration is gone from the shape; it is
     measured separately as a feature (B6).
  4. Template = sample-by-sample mean of the stretched reps.

TEMPLATE_LEN = 60 (about one 2 s rep at 30 fps) is a design choice, not a
literature value. Templates must be built from training subjects only.

Run (from the repo root):
    python -m neuroflow.segmentation.templates
"""

import os

import numpy as np
import pandas as pd

from neuroflow.segmentation.dtw import dtw_distance, dtw_path
from neuroflow.segmentation.segment import smooth

TEMPLATE_LEN = 60
MIN_VALID_FRAMES = 5


def resample_rep(t_ms, theta, t_start, t_end, n=TEMPLATE_LEN):
    """Linearly stretch one rep (between t_start and t_end, in ms) to n samples.
    Returns None if the rep has too few valid frames or no valid boundaries."""
    if np.isnan(t_start) or np.isnan(t_end) or t_end <= t_start:
        return None
    ok = ~np.isnan(theta)
    inside = ok & (t_ms >= t_start) & (t_ms <= t_end)
    if inside.sum() < MIN_VALID_FRAMES:
        return None
    # Boundaries fall between frames, so keep one valid frame either side as well;
    # otherwise the first/last samples would be clamped instead of interpolated.
    idx = np.flatnonzero(inside)
    valid = np.flatnonzero(ok)
    before = valid[valid < idx[0]]
    after = valid[valid > idx[-1]]
    use = np.concatenate([before[-1:], idx, after[:1]])
    t_new = np.linspace(t_start, t_end, n)
    return np.interp(t_new, t_ms[use], theta[use])


def rep_matrix(frames, reps, rest_ref, rom_ref, n=TEMPLATE_LEN):
    """
    Stack all usable reps of one block as rows of a (k, n) array, normalised to
    the user's range. `reps` needs t_start_ms and t_end_ms (e.g. from segment_reps).
    Also returns the rep_ids that were kept, so rows can be traced back.
    """
    frames = frames.sort_values("t_capture_ms")
    t = frames["t_capture_ms"].to_numpy(dtype=float)
    theta = smooth(t, frames["theta_ht_deg"].to_numpy())
    theta = (theta - rest_ref) / rom_ref
    rows, kept = [], []
    for _, r in reps.iterrows():
        x = resample_rep(t, theta, r["t_start_ms"], r["t_end_ms"], n)
        if x is not None:
            rows.append(x)
            kept.append(r.get("rep_id", len(kept)))
    return np.array(rows), kept


def mean_template(R):
    """Linear-stretch mean template: average of the stretched reps, sample by sample."""
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or len(R) == 0:
        raise ValueError("need a (k, n) array with at least one rep")
    return R.mean(axis=0)


def dba_template(R, band_frac=0.15, init=None, max_iter=20, tol=1e-6):
    """
    DTW Barycenter Averaging (Petitjean et al. 2011).
    Start from `init` (default: the linear-stretch mean) and repeat:
      1. align every rep to the current template with banded DTW;
      2. replace each template sample by the mean of all rep samples aligned to it.
    Each round can only lower (or keep) the total squared DTW distance from the
    reps to the template, so DBA is never worse than its starting point on that
    measure. Returns (template, list of total squared distance per round).
    """
    R = np.asarray(R, dtype=float)
    avg = mean_template(R) if init is None else np.asarray(init, dtype=float).copy()
    history = [float(np.sum(spread(R, avg, band_frac) ** 2))]
    for _ in range(max_iter):
        sums = np.zeros(len(avg))
        counts = np.zeros(len(avg))
        for x in R:
            _, path = dtw_path(avg, x, band_frac)
            for i, j in path:
                sums[i] += x[j]
                counts[i] += 1
        avg = sums / counts            # every template sample is on every path at least once
        history.append(float(np.sum(spread(R, avg, band_frac) ** 2)))
        if history[-2] - history[-1] < tol:
            break
    return avg, history


def timing_varied_reps(k=20, n=TEMPLATE_LEN, peak_lo=0.35, peak_hi=0.65, noise=0.01, seed=0):
    """
    Test data only: k unit-height reps whose peak lands anywhere between peak_lo
    and peak_hi of the rep (the committed mocks all peak at exactly 50%, so they
    cannot show what DBA is for). Warping u -> u**a moves the peak to 0.5**(1/a).
    """
    rng = np.random.default_rng(seed)
    u = np.linspace(0, 1, n)
    rows = []
    for _ in range(k):
        a = np.log(0.5) / np.log(rng.uniform(peak_lo, peak_hi))
        rows.append((1 - np.cos(2 * np.pi * u ** a)) / 2 + rng.normal(0, noise, n))
    return np.array(rows)


def spread(R, template, band_frac=0.15):
    """DTW distance of every rep to the template (smaller = tighter template)."""
    return np.array([dtw_distance(x, template, band_frac) for x in R])


def expected_shape(truth, rest_ref, rom_ref, n=TEMPLATE_LEN, frac=0.05):
    """
    Mock-only: the exact mean shape the template should recover. Each mock rep is a
    half-cosine; between its 5% boundaries u runs from u0 to 1 - u0.
    """
    u0 = np.arccos(1 - 2 * frac) / (2 * np.pi)
    u = np.linspace(u0, 1 - u0, n)
    shape = (1 - np.cos(2 * np.pi * u)) / 2
    rest = truth["rest_deg"].to_numpy()
    amp = (truth["peak_deg"].to_numpy() - rest) / rom_ref
    return amp.mean() * shape + ((rest - rest_ref) / rom_ref).mean()


def main():
    import sys
    from neuroflow.segmentation.segment import calibration_refs, segment_reps
    fx = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures")
    calib = calibration_refs()

    def load(v):
        f = pd.read_parquet(os.path.join(fx, f"mock_frames_{v}.parquet"))
        tr = pd.read_csv(os.path.join(fx, f"mock_frames_{v}_truth.csv"))
        rest, rom = calib[f["subject_id"].iloc[0]]
        return f, tr, rest, rom

    print(f"template length {TEMPLATE_LEN} samples; reps cut by the B3 segmenter; "
          f"angle as (theta - rest_ref) / rom_ref\n")
    print(f"{'variant':<8} {'reps':>4} {'RMSE vs true':>13} {'corr':>7} {'peak idx':>9} "
          f"{'peak':>6} {'ends':>12} {'spread mean':>12}")
    templates = {}
    for v in ["normal", "slow", "capped", "leaning", "low_rom"]:
        f, tr, rest, rom = load(v)
        R, _ = rep_matrix(f, segment_reps(f, rom, rest), rest, rom)
        T = mean_template(R)
        E = expected_shape(tr, rest, rom)
        rmse = float(np.sqrt(np.mean((T - E) ** 2)))
        corr = float(np.corrcoef(T, E)[0, 1])
        d = spread(R, T)
        templates[v] = T
        print(f"{v:<8} {len(R):>4} {rmse:>13.4f} {corr:>7.4f} {int(np.argmax(T)):>9} "
              f"{T.max():>6.3f} {T[0]:>5.3f}/{T[-1]:<5.3f} {d.mean():>12.3f}")

    n = TEMPLATE_LEN
    print(f"\n(peak index of a symmetric rep is {(n - 1) / 2:.1f}; "
          f"ends should sit near 0.05 x amplitude above rest)")
    print(f"DTW(normal template, slow template)    = "
          f"{dtw_distance(templates['normal'], templates['slow'], 0.15):.3f}  "
          f"(stretching removes overall tempo)")
    print(f"DTW(normal template, capped template)  = "
          f"{dtw_distance(templates['normal'], templates['capped'], 0.15):.3f}  "
          f"(per-person normalisation keeps the smaller range visible)")

    # ---- step 2: DBA vs mean ----
    print("\nDBA vs linear-stretch mean (band 15%); spread = mean DTW distance rep -> template")
    print(f"{'data':<22} {'mean spread':>11} {'DBA spread':>11} {'gain':>7} {'rounds':>7} "
          f"{'mean peak':>10} {'DBA peak':>9}")
    sets = {v: rep_matrix(*_load_block(load, v))[0] for v in ["normal", "slow", "capped", "low_rom"]}
    sets["timing-varied (synth)"] = timing_varied_reps()
    for name, R in sets.items():
        T_mean = mean_template(R)
        T_dba, hist = dba_template(R)
        s_mean, s_dba = spread(R, T_mean).mean(), spread(R, T_dba).mean()
        assert all(h1 >= h2 - 1e-9 for h1, h2 in zip(hist, hist[1:])), "DBA objective went up"
        print(f"{name:<22} {s_mean:>11.4f} {s_dba:>11.4f} {1 - s_dba / s_mean:>6.1%} "
              f"{len(hist) - 1:>7} {T_mean.max():>10.3f} {T_dba.max():>9.3f}")
    print("(synthetic reps all have height 1.0; a template peak below 1 means the peak was blurred)")


def _load_block(load, v):
    """Helper for main(): frames, B3 reps, rest_ref, rom_ref for one mock."""
    from neuroflow.segmentation.segment import segment_reps
    f, _, rest, rom = load(v)
    return f, segment_reps(f, rom, rest), rest, rom


if __name__ == "__main__":
    main()
