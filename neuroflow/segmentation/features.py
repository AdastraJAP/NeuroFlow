"""
B6: per-rep features (owner: Member B).

Each feature is its own function and is validated on its own (tests/test_features.py)
before rep_features() combines them into reps.parquet columns.

    duration_s          (t_end - t_start) / 1000, between the 5% boundaries
    time_to_peak_ratio  (t_peak - t_start) / (t_end - t_start); 0.5 = symmetric rep
    rom_peak_deg        peak smoothed theta_HT inside the rep minus rest_ref
                        (degrees moved from rest)
    rom_rel_personal    rom_peak_deg / rom_ref - 1: personal-centred range,
                        0 = the user's typical range, -0.5 = half of it
    sparc               spectral arc length of the angular-speed profile
                        (Balasubramanian et al. 2015); closer to 0 = smoother
  B7 compensation (max/min of A4's per-frame signal between the boundaries):
    max_trunk_lean_deg  max |trunk_lean_deg| (absolute: open question for A,
                        depends on how A4 signs lean)
    min_elbow_deg       min elbow_deg; 180 = straight arm, lower = more bent
    max_hike_ratio      max hike_ratio (exploratory; weak evidence)
    max_shoulder_tilt_deg  max(|wrap(shoulder_tilt_deg - resting tilt)|) over the rep:
                        rest-corrected only, no correction for arm elevation
                        (exploratory, #6/#14; see rest_corrected_tilt for the
                        resting window). A different definition gets a new
                        column name, never this one (PR #15 review).

Choices made here (not literature values): ROM as excursion from rest, the ratio
form of the personal centring, SPARC on the whole rep (raise + lower) using the
absolute angular speed of the smoothed theta_HT. rest_ref / rom_ref come from the
user's calibration block (segment.calibrate). Overall duration is kept as an
absolute value; dividing by the user's calibration median is left to the scorer.

SPARC parameters (zero-padding level 4, cut-off 10 Hz, amplitude threshold 0.05)
follow the authors' published reference implementation as I remember it; check
them against the original code before citing.

Run (from the repo root):
    python -m neuroflow.segmentation.features
"""

import os

import numpy as np
import pandas as pd

from neuroflow.segmentation.segment import angle_signal, reps_angle_col, smooth

SPARC_PADLEVEL = 4
SPARC_FC_HZ = 10.0
SPARC_AMP_TH = 0.05


# ---- 1. duration ----

def rep_duration_s(t_start_ms, t_end_ms):
    """Rep duration in seconds; NaN if a boundary is unknown."""
    return (np.asarray(t_end_ms, dtype=float) - np.asarray(t_start_ms, dtype=float)) / 1000.0


# ---- 2. time-to-peak ratio ----

def time_to_peak_ratio(t_start_ms, t_peak_ms, t_end_ms):
    """Fraction of the rep spent rising: (peak - start) / (end - start)."""
    t_s = np.asarray(t_start_ms, dtype=float)
    return (np.asarray(t_peak_ms, dtype=float) - t_s) / (np.asarray(t_end_ms, dtype=float) - t_s)


# ---- 3. personal-centred peak range ----

def peak_rom(t_ms, theta_s, t_start_ms, t_end_ms, rest_ref, rom_ref):
    """
    (rom_peak_deg, rom_rel_personal) for one rep.
    theta_s is the smoothed theta_HT of the whole block; the peak is its maximum
    between the rep's boundaries.
    """
    if np.isnan(t_start_ms) or np.isnan(t_end_ms):
        return np.nan, np.nan
    inside = (t_ms >= t_start_ms) & (t_ms <= t_end_ms)
    if not inside.any():
        return np.nan, np.nan
    rom = float(np.nanmax(theta_s[inside])) - rest_ref
    return rom, rom / rom_ref - 1.0


# ---- 4. SPARC ----

def sparc(speed, fs, padlevel=SPARC_PADLEVEL, fc=SPARC_FC_HZ, amp_th=SPARC_AMP_TH):
    """
    Spectral arc length of a speed profile sampled uniformly at fs (Hz).
      1. Magnitude spectrum of the zero-padded speed, normalised to its maximum.
      2. Keep frequencies up to fc, then trim to the band where the spectrum is
         at or above amp_th (the adaptive cut-off).
      3. Arc length of that curve with frequency scaled to [0, 1]; SPARC is minus
         that length. A smooth single bump gives a short arc (close to 0); extra
         sub-movements or tremor add spectral bumps and make it more negative.
    Unaffected by the speed's amplitude; designed to be robust to duration.
    """
    speed = np.asarray(speed, dtype=float)
    if len(speed) < 3 or np.isnan(speed).any() or not np.any(speed):
        return np.nan
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    freq = np.arange(nfft) * fs / nfft
    mag = np.abs(np.fft.fft(speed, nfft))
    mag = mag / mag.max()

    sel = freq <= fc
    f_sel, m_sel = freq[sel], mag[sel]
    above = np.flatnonzero(m_sel >= amp_th)
    f_sel = f_sel[above[0]:above[-1] + 1]
    m_sel = m_sel[above[0]:above[-1] + 1]
    if len(f_sel) < 2:
        return 0.0
    df = np.diff(f_sel) / (f_sel[-1] - f_sel[0])
    return -float(np.sum(np.sqrt(df ** 2 + np.diff(m_sel) ** 2)))


def rep_speed(t_ms, theta_s, t_start_ms, t_end_ms):
    """
    Absolute angular speed (deg/s) of one rep on a uniform grid, and its rate fs.
    The smoothed angle is resampled at the block's median frame rate first,
    because the FFT in SPARC needs evenly spaced samples.
    """
    if np.isnan(t_start_ms) or np.isnan(t_end_ms) or t_end_ms <= t_start_ms:
        return None, None
    dt = float(np.median(np.diff(t_ms)))
    t_u = np.arange(t_start_ms, t_end_ms + dt / 2, dt)
    if len(t_u) < 3:
        return None, None
    theta_u = np.interp(t_u, t_ms, theta_s)
    return np.abs(np.gradient(theta_u, dt / 1000.0)), 1000.0 / dt


# ---- 5. compensation (B7) ----

def rep_window_extreme(t_ms, signal_s, t_start_ms, t_end_ms, how, min_valid_frac=0.0):
    """
    Max or min (how = "max" | "min") of a per-frame signal between a rep's boundaries.
    One helper serves all three compensation features, which are the same
    windowed reduction of a column A4 already computes per frame:
        max_trunk_lean_deg  how="max" on trunk_lean_deg
        min_elbow_deg       how="min" on elbow_deg      (180 = straight arm)
        max_hike_ratio      how="max" on hike_ratio     (already relative to rest)
    Pass the signal smoothed (segment.smooth): the max of raw noise is biased upward.
    NaN frames are skipped; NaN if a boundary is unknown, no valid frame lies inside,
    or fewer than min_valid_frac of the rep's frames are valid (a max/min over a
    handful of frames is not a rep-level measurement).
    The literature flags (lean 5.2 deg, elbow 153.4 deg, hike 0.174) belong to the
    scorer (C4), not here.
    """
    if how not in ("max", "min"):
        raise ValueError(f"how must be 'max' or 'min', got {how!r}")
    if np.isnan(t_start_ms) or np.isnan(t_end_ms):
        return np.nan
    t_ms = np.asarray(t_ms, dtype=float)
    inside = np.asarray(signal_s, dtype=float)[(t_ms >= t_start_ms) & (t_ms <= t_end_ms)]
    vals = inside[~np.isnan(inside)]
    if len(vals) == 0 or len(vals) < min_valid_frac * len(inside):
        return np.nan
    return float(vals.max() if how == "max" else vals.min())


# A compensation value needs at least half of the rep's frames valid. On Adastra's G1
# clips elbow was valid on 0-13 frames per rep (wrist out of frame), which gave
# one-frame "minimums". 50 % is our choice, not a literature value; tune on training data.
COMP_MIN_VALID_FRAC = 0.5

# Per-frame column -> (reps column, reduction, transform before smoothing).
COMPENSATION = [
    ("trunk_lean_deg", "max_trunk_lean_deg", "max", np.abs),
    ("elbow_deg", "min_elbow_deg", "min", None),
    ("hike_ratio", "max_hike_ratio", "max", None),
]


# Shoulder-line tilt (exploratory, issue #6; agreed with 7willA as a sub-score, never a
# feedback code). Per-frame shoulder_tilt_deg, positive = right shoulder higher
# (Adastra's V4 convention). Posture and camera already tilt the shoulder line a
# little at rest, so the resting tilt is subtracted.
#   Resting window: the TILT_REST_S = 0.5 s just before the rep's start (arm still
#   at rest), cut at the previous rep's end; resting tilt = median of its valid
#   frames, needs at least TILT_REST_MIN_FRAMES = 8 (as V4's tilt_change baseline).
#   Our choice, not a literature value.
#   Deviation wrapped to [-90, 90) (as V4), then max |deviation| over the rep with
#   the same 50 % valid-frame rule as the other compensation features.
TILT_COL = "shoulder_tilt_deg"
TILT_REST_S = 0.5
TILT_REST_MIN_FRAMES = 8


def wrap_deg(x):
    """Angle difference wrapped to [-90, 90): a line has no direction."""
    return (np.asarray(x, dtype=float) + 90.0) % 180.0 - 90.0


def resting_tilt(t_ms, tilt, t_start_ms, t_prev_end_ms=np.nan):
    """Median raw tilt over the TILT_REST_S before t_start (not before the previous
    rep's end). NaN if the start is unknown or fewer than TILT_REST_MIN_FRAMES are valid."""
    if np.isnan(t_start_ms):
        return np.nan
    lo = t_start_ms - TILT_REST_S * 1000.0
    if not np.isnan(t_prev_end_ms):
        lo = max(lo, t_prev_end_ms)
    t_ms = np.asarray(t_ms, dtype=float)
    win = np.asarray(tilt, dtype=float)[(t_ms >= lo) & (t_ms < t_start_ms)]
    win = win[~np.isnan(win)]
    return float(np.median(win)) if len(win) >= TILT_REST_MIN_FRAMES else np.nan


def rest_corrected_tilt(t_ms, tilt, tilt_s, t_start_ms, t_end_ms, t_prev_end_ms=np.nan):
    """max(|wrap(tilt_s - resting tilt)|) over one rep, rest-corrected only (no
    elevation correction); NaN like the other B7 features."""
    base = resting_tilt(t_ms, tilt, t_start_ms, t_prev_end_ms)
    if np.isnan(base):
        return np.nan
    dev = np.abs(wrap_deg(np.asarray(tilt_s, dtype=float) - base))
    return rep_window_extreme(t_ms, dev, t_start_ms, t_end_ms, "max", COMP_MIN_VALID_FRAC)


# ---- integration ----

def rep_features(frames, reps, rest_ref, rom_ref, angle_col=None):
    """
    Add the B6 feature columns to a reps table (from segment_reps / segment_fixed).
    rest_ref, rom_ref: the user's calibration (segment.calibrate on their own
    calibration block, never the block being scored).
    angle_col: defaults to the angle the reps were segmented on (their angle_col
    column); passing a different one raises.
    """
    frames, t, theta = angle_signal(frames, reps_angle_col(reps, angle_col))
    theta_s = smooth(t, theta)

    out = reps.copy()
    out["duration_s"] = rep_duration_s(out["t_start_ms"], out["t_end_ms"])
    out["time_to_peak_ratio"] = time_to_peak_ratio(out["t_start_ms"], out["t_peak_ms"], out["t_end_ms"])
    roms, rels, sparcs = [], [], []
    for _, r in out.iterrows():
        rom, rel = peak_rom(t, theta_s, r["t_start_ms"], r["t_end_ms"], rest_ref, rom_ref)
        speed, fs = rep_speed(t, theta_s, r["t_start_ms"], r["t_end_ms"])
        roms.append(rom)
        rels.append(rel)
        sparcs.append(sparc(speed, fs) if speed is not None else np.nan)
    out["rom_peak_deg"] = roms
    out["rom_rel_personal"] = rels
    out["sparc"] = sparcs

    # B7: a column missing from frames (e.g. A4 v0 writes theta_HT only) -> NaN, never 0.
    for col, name, how, transform in COMPENSATION:
        if col not in frames.columns:
            out[name] = np.nan
            continue
        sig = frames[col].to_numpy(dtype=float)
        sig_s = smooth(t, transform(sig) if transform else sig)
        sig_s[np.isnan(sig)] = np.nan   # smooth() fills gaps; don't invent values A4 didn't have
        out[name] = [rep_window_extreme(t, sig_s, r["t_start_ms"], r["t_end_ms"], how,
                                        COMP_MIN_VALID_FRAC)
                     for _, r in out.iterrows()]

    # shoulder-line tilt, rest-corrected per rep (frames without the column -> NaN)
    if TILT_COL not in frames.columns:
        out["max_shoulder_tilt_deg"] = np.nan
    else:
        tilt = frames[TILT_COL].to_numpy(dtype=float)
        tilt_s = smooth(t, tilt)
        tilt_s[np.isnan(tilt)] = np.nan
        prev_end = np.r_[np.nan, out["t_end_ms"].to_numpy(dtype=float)[:-1]]
        out["max_shoulder_tilt_deg"] = [
            rest_corrected_tilt(t, tilt, tilt_s, r["t_start_ms"], r["t_end_ms"], pe)
            for (_, r), pe in zip(out.iterrows(), prev_end)]
    return out


def main():
    from neuroflow.segmentation.segment import calibration_refs, segment_reps
    fx = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures")
    calib = calibration_refs()
    print("B6/B7 features on the committed mocks (B3 segmenter, calibration from separate blocks)\n")
    print(f"{'variant':<8} {'dur s':>6} {'(true)':>7} {'ttp':>6} {'rom deg':>8} {'(true)':>7} "
          f"{'rom_rel':>8} {'sparc':>7} {'lean':>6} {'elbow':>6} {'hike':>6}")
    for v in ["normal", "slow", "capped", "leaning", "low_rom", "elbow_30", "hike"]:
        f = pd.read_parquet(os.path.join(fx, f"mock_frames_{v}.parquet"))
        tr = pd.read_csv(os.path.join(fx, f"mock_frames_{v}_truth.csv"))
        rest, rom = calib[f["subject_id"].iloc[0]]
        feats = rep_features(f, segment_reps(f, rom, rest), rest, rom)
        true_dur = ((tr["t_end_ms"] - tr["t_start_ms"]) / 1000).mean()
        true_rom = (tr["peak_deg"] - tr["rest_deg"]).mean()
        print(f"{v:<8} {feats['duration_s'].mean():>6.2f} {true_dur:>7.2f} "
              f"{feats['time_to_peak_ratio'].mean():>6.3f} {feats['rom_peak_deg'].mean():>8.1f} "
              f"{true_rom:>7.1f} {feats['rom_rel_personal'].mean():>+8.2f} {feats['sparc'].mean():>7.3f} "
              f"{feats['max_trunk_lean_deg'].mean():>6.1f} {feats['min_elbow_deg'].mean():>6.1f} "
              f"{feats['max_hike_ratio'].mean():>6.3f}")


if __name__ == "__main__":
    main()
