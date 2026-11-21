"""
B1: mock frames.parquet generator (owner: Member B).

Builds fake per-frame tables that follow the frames.parquet contract
(schema_version 1) so segmentation and features can be developed before
real data exists. Every file comes with a truth CSV holding the exact
start, peak and end time of each repetition.

Variants (one file each):
    normal   10 reps, ~2 s each, peak ~90 deg
    slow     same, but each rep 2x longer       (condition "slow", level "2x")
    capped   peak ~45 deg                        (condition "rom_cap", level "45deg")
    leaning  trunk leans up to ~12 deg per rep   (condition "trunk_lean", level "large")
    low_rom  a second user whose full range is ~25 deg (subject mock02)
    elbow_30 elbow bends from ~175 to ~145 deg per rep (condition "elbow_bend", level "30deg")
    hike     shoulder hike rises to ~0.20 per rep  (condition "hike", level "shrug")

Run:
    python fixtures/make_mock_frames.py
Writes:
    fixtures/mock_frames_<variant>.parquet
    fixtures/mock_frames_<variant>_truth.csv

Rep boundaries in the truth file: t_start_ms / t_end_ms are where the clean
angle crosses rest + 5% of that rep's range (BOUNDARY_FRAC, defined once in
neuroflow/segmentation/segment.py). t_onset_ms / t_offset_ms (first and last
movement) are kept for reference only.

Assumptions (mock only, not literature values): 30 fps with +-3 ms frame
jitter, rest ~10 deg, angle noise 1 deg, symmetric half-cosine reps
(time_to_peak_ratio = 0.5 exactly), theta_v = theta_ht + trunk lean
(first-order relation from the Research Brief, an inference).
B7 extension: the elbow_30 / hike shapes follow the same half-cosine as the
rep; their sizes (30 deg, 0.20) are mock choices picked to sit clearly past
the literature flags (153.4 deg, 0.174), not measured values. They add no
random draws, so the older variants' files are unchanged.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuroflow.segmentation.segment import BOUNDARY_FRAC  # single source of the definition

SCHEMA_VERSION = 1
LANDMARK_IDS = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
LANDMARK_FIELDS = ["x_px", "y_px", "z", "vis", "pres"]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42

FPS = 30.0
FRAME_JITTER_MS = 3.0
T0_MS = 1_000_000.0  # capture clock does not start at 0; code must not assume it does

VARIANTS = {
    #           condition     level    dur_scale  peak  lean_max
    "normal":  ("normal",     "none",  1.0,       90.0, 0.0),
    "slow":    ("slow",       "2x",    2.0,       90.0, 0.0),
    "capped":  ("rom_cap",    "45deg", 1.0,       45.0, 0.0),
    "leaning": ("trunk_lean", "large", 1.0,       90.0, 12.0),
    # B3: a different user whose own full range only reaches ~25 deg. Fixed 30 deg
    # thresholds (v0) never fire; ROM-normalised thresholds (v1) should.
    "low_rom": ("low_rom",    "25deg", 1.0,       25.0, 0.0),
    # B7: one compensation each, to check the compensation features react.
    # Names match Member C's mock blocks (block_id elbow_30 / hike).
    "elbow_30": ("elbow_bend", "30deg", 1.0,      90.0, 0.0),
    "hike":    ("hike",       "shrug", 1.0,       90.0, 0.0),
}
# Per-variant overrides (variants not listed keep the defaults, so their files do not change).
PEAK_SD = {"low_rom": 1.0}          # default 3.0 deg rep-to-rep peak spread
SUBJECT = {"low_rom": "mock02"}     # default "mock01"
ELBOW_DROP = {"elbow_30": 30.0}     # deg the elbow bends at the rep peak (default 0)
HIKE_MAX = {"hike": 0.20}           # hike_ratio at the rep peak (default 0)


def make_rep(t_ms, rest, peak, dur_ms):
    """
    Angle of one repetition as a raised half-cosine.
    t_ms is time since the rep started; outside [0, dur_ms] the arm is at rest.
    u = 0 -> rest, u = 0.5 -> peak, u = 1 -> rest.
    """
    u = np.clip(np.asarray(t_ms, dtype=float) / dur_ms, 0.0, 1.0)
    return rest + (peak - rest) * (1.0 - np.cos(2.0 * np.pi * u)) / 2.0


def boundary_u(frac=BOUNDARY_FRAC):
    """
    Fraction of the rep at which a half-cosine first reaches rest + frac * range.
    Solves (1 - cos 2*pi*u) / 2 = frac.  For frac = 0.05: u = 0.0718.
    The end boundary is the mirror point, 1 - u.
    """
    return np.arccos(1.0 - 2.0 * frac) / (2.0 * np.pi)


def make_timeline(total_ms, rng):
    """Capture timestamps (ms) at ~30 fps with small random jitter."""
    n = int(total_ms / (1000.0 / FPS)) + 1
    steps = 1000.0 / FPS + rng.normal(0.0, FRAME_JITTER_MS, n)
    steps = np.clip(steps, 20.0, None)
    steps[0] = 0.0
    return T0_MS + np.cumsum(steps)


def plan_reps(n_reps, dur_scale, peak, rng, peak_sd=3.0):
    """Choose start time, duration and peak for every rep (times relative to T0)."""
    reps = []
    t = 2000.0  # 2 s of rest before the first rep
    for _ in range(n_reps):
        dur = 2000.0 * dur_scale * rng.uniform(0.9, 1.1)
        reps.append({"start": t, "dur": dur, "peak": peak + rng.normal(0.0, peak_sd)})
        t += dur + rng.uniform(1000.0, 1500.0)  # rest between reps
    return reps, t + 2000.0  # 2 s of rest after the last rep


def make_session(variant, n_reps=10, noise_deg=1.0, seed=SEED):
    """Return (frames DataFrame, truth DataFrame) for one variant."""
    condition, level, dur_scale, peak, lean_max = VARIANTS[variant]
    rng = np.random.default_rng(seed)

    reps, total_ms = plan_reps(n_reps, dur_scale, peak, rng, PEAK_SD.get(variant, 3.0))
    t = make_timeline(total_ms, rng)
    rel = t - T0_MS
    rest = 10.0

    theta_ht = np.full(len(t), rest)
    lean = np.zeros(len(t))
    elbow_drop = np.zeros(len(t))
    hike = np.zeros(len(t))
    for r in reps:
        inside = (rel >= r["start"]) & (rel <= r["start"] + r["dur"])
        t_rep = rel[inside] - r["start"]
        theta_ht[inside] = make_rep(t_rep, rest, r["peak"], r["dur"])
        if lean_max > 0:
            lean[inside] = make_rep(t_rep, 0.0, lean_max, r["dur"])
        # No random draws here, so the older variants stay byte-for-byte the same.
        elbow_drop[inside] = make_rep(t_rep, 0.0, ELBOW_DROP.get(variant, 0.0), r["dur"])
        hike[inside] = make_rep(t_rep, 0.0, HIKE_MAX.get(variant, 0.0), r["dur"])

    n = len(t)
    theta_ht = theta_ht + rng.normal(0.0, noise_deg, n)
    lean = lean + rng.normal(0.0, noise_deg / 2, n)

    frames = pd.DataFrame({
        "schema_version": SCHEMA_VERSION,
        "subject_id": SUBJECT.get(variant, "mock01"),
        "session_id": "s1",
        "block_id": variant,
        "condition": condition,
        "level": level,
        "side": "right",
        "frame_idx": np.arange(n),
        "t_capture_ms": t,
        "img_w": 1280,
        "img_h": 720,
        "pose_model": "mock",
        "pose_model_ver": "b1-v1",
        "detected": True,
    })
    # No real pose in a mock: landmark columns exist but are NaN (never 0).
    for lm in LANDMARK_IDS:
        for f in LANDMARK_FIELDS:
            frames[f"lm{lm}_{f}"] = np.nan

    frames["theta_ht_deg"] = theta_ht
    frames["theta_v_deg"] = theta_ht + lean
    frames["trunk_lean_deg"] = lean
    frames["elbow_deg"] = 175.0 - elbow_drop + rng.normal(0.0, noise_deg, n)
    frames["hike_ratio"] = hike + rng.normal(0.0, 0.01, n)
    frames["in_frame"] = True
    frames["jitter"] = 0.0
    frames["arm_len_ratio"] = 1.0
    frames["frame_conf"] = 1.0

    # Rep boundary definition (team decision 2026-11-18): start/end are where the
    # clean angle crosses rest + BOUNDARY_FRAC * (peak - rest). The true movement
    # onset/offset are kept for reference only.
    u = boundary_u()
    truth = pd.DataFrame({
        "rep_idx": np.arange(n_reps),
        "t_start_ms": [T0_MS + r["start"] + u * r["dur"] for r in reps],
        "t_peak_ms": [T0_MS + r["start"] + r["dur"] / 2 for r in reps],
        "t_end_ms": [T0_MS + r["start"] + (1.0 - u) * r["dur"] for r in reps],
        "peak_deg": [r["peak"] for r in reps],
        "rest_deg": rest,
        "t_onset_ms": [T0_MS + r["start"] for r in reps],
        "t_offset_ms": [T0_MS + r["start"] + r["dur"] for r in reps],
    })
    return frames, truth


def check_session(variant, frames, truth):
    """Quick self-test; raises AssertionError if the mock is wrong."""
    t = frames["t_capture_ms"].to_numpy()
    assert np.all(np.diff(t) > 0), "timestamps must strictly increase"
    assert len(truth) == 10
    assert np.all(truth["t_start_ms"].to_numpy()[1:] > truth["t_end_ms"].to_numpy()[:-1]), \
        "reps overlap"
    assert truth["t_end_ms"].iloc[-1] < t[-1], "last rep runs past the clip"

    # The frame nearest each true peak should be near the planned peak angle.
    for _, r in truth.iterrows():
        i = int(np.argmin(np.abs(t - r["t_peak_ms"])))
        got = frames["theta_ht_deg"].iloc[i]
        assert abs(got - r["peak_deg"]) < 6.0, f"{variant}: peak {got:.1f} vs {r['peak_deg']:.1f}"

    if variant == "capped":
        assert frames["theta_ht_deg"].max() < 55.0, "capped mock must stay below 55 deg"
    if variant == "low_rom":
        assert truth["peak_deg"].max() < 30.0, "low_rom reps must not reach the 30 deg v0 threshold"
    if variant == "leaning":
        assert frames["trunk_lean_deg"].max() > 8.0
    else:
        assert frames["trunk_lean_deg"].max() < 3.0
    if variant == "elbow_30":
        assert frames["elbow_deg"].min() < 150.0, "elbow_30 must bend well below 175 deg"
    else:
        assert frames["elbow_deg"].min() > 168.0, "elbow must stay near straight"
    if variant == "hike":
        assert frames["hike_ratio"].max() > 0.17, "hike must rise past 0.17"
    else:
        assert frames["hike_ratio"].max() < 0.06, "hike must stay near 0"


def main():
    print(f"seed = {SEED}")
    print(f"{'variant':<9} {'frames':>6} {'fps':>6} {'reps':>4} {'mean dur s':>10} "
          f"{'max ht':>7} {'max lean':>8} {'min elb':>7} {'max hike':>8}")
    for variant in VARIANTS:
        frames, truth = make_session(variant)
        check_session(variant, frames, truth)

        p_frames = os.path.join(OUT_DIR, f"mock_frames_{variant}.parquet")
        p_truth = os.path.join(OUT_DIR, f"mock_frames_{variant}_truth.csv")
        frames.to_parquet(p_frames, index=False)
        truth.to_csv(p_truth, index=False)

        back = pd.read_parquet(p_frames)
        assert back.shape == frames.shape, "parquet round trip changed the table"

        t = frames["t_capture_ms"].to_numpy()
        fps = 1000.0 / np.median(np.diff(t))
        dur = (truth["t_end_ms"] - truth["t_start_ms"]).mean() / 1000.0
        print(f"{variant:<9} {len(frames):>6} {fps:>6.1f} {len(truth):>4} {dur:>10.2f} "
              f"{frames['theta_ht_deg'].max():>7.1f} {frames['trunk_lean_deg'].max():>8.1f} "
              f"{frames['elbow_deg'].min():>7.1f} {frames['hike_ratio'].max():>8.3f}")
    print(f"\nall checks passed; files in {OUT_DIR}")


if __name__ == "__main__":
    main()
