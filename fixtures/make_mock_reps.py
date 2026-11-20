"""
C1: mock reps.parquet generator (owner: Member C).

C can't wait for B's real reps, so this script writes a fake reps table with a
known structure. The scorer (C2-C4) and the evaluation code (C5-C8) are built
and tested on it until B's real table passes gate G2.

What it makes: fixtures/mock_reps.parquet
    18 subjects x 2 sessions x 13 blocks x 10 reps = 4,680 reps
    columns follow the reps.parquet contract (schema_version 1)

Blocks (13, from the protocol table in the research basis):
    normal_calib, normal_test, slow 2x/3x, band light/medium,
    trunk_lean small/large, elbow_bend 30/60, hike shrug, rom_cap 60deg/90deg
    -> 7 conditions, 13 condition-level blocks.

Design choices (mock choices, NOT literature values):
- Every subject has their own typical range, duration and small offsets on every
  feature (a "subject effect"). This matters: without it, a random split inside
  each subject and a leave-one-subject-out split would give the same answer, and
  P1 would have nothing to measure.
- Impairment effects are graded by level and follow the research brief:
    slow        duration x2 / x3; DTW distance barely changes (DTW is speed-blind)
    band        smaller range, less smooth, a bit slower
    trunk_lean  larger max trunk lean; theta_HT range unchanged
    elbow_bend  smaller min elbow angle
    hike        slightly larger hike ratio (weak, overlaps normal on purpose)
    rom_cap     range capped -> large DTW distance (positive control only)
- Feature conventions match B's code (Melchior-branch):
    rom_peak_deg      degrees moved from rest (peak - rest)
    rom_rel_personal  rom_peak_deg / personal calibration median - 1
                      (calibration = the subject's session-1 normal_calib block)
    sparc             negative; closer to 0 = smoother
    dtw_dist          DTW on theta_HT / personal peak, so it has no unit
- Times are ms on a capture clock that starts at 1,000,000 ms (not 0), so code
  that assumes time starts at zero breaks here and not on real data.
- About 0.5 % of reps have an unknown start boundary (NaN), and about 3 % have a
  low valid-frame fraction, so the scorer has to handle missing and bad reps.

Run (from the repo root):
    python fixtures/make_mock_reps.py
"""

import json
import os

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
SEED = 42
N_SUBJECTS = 18
SESSIONS = ("s1", "s2")
REPS_PER_BLOCK = 10
CLOCK_START_MS = 1_000_000.0
REST_BETWEEN_REPS_S = 1.0
REST_BETWEEN_BLOCKS_S = 30.0
OUT_PATH = os.path.join("fixtures", "mock_reps.parquet")

# block_id, condition, level
BLOCKS = [
    ("normal_calib", "normal", "none"),
    ("normal_test", "normal", "none"),
    ("slow_2x", "slow", "2x"),
    ("slow_3x", "slow", "3x"),
    ("band_light", "band", "light"),
    ("band_medium", "band", "medium"),
    ("lean_small", "trunk_lean", "small"),
    ("lean_large", "trunk_lean", "large"),
    ("elbow_30", "elbow_bend", "30deg"),
    ("elbow_60", "elbow_bend", "60deg"),
    ("hike", "hike", "shrug"),
    ("romcap_60", "rom_cap", "60deg"),
    ("romcap_90", "rom_cap", "90deg"),
]

# What each (condition, level) does to a normal rep. Mock choices.
#   dur    duration multiplier          rom   range multiplier
#   cap    max peak angle (deg); range is capped at cap - REST_DEG
#   sparc, lean, elbow, hike, dtw: added to the normal value
EFFECTS = {
    ("normal", "none"): {},
    ("slow", "2x"): {"dur": 2.0, "sparc": -0.25, "dtw": 0.02},
    ("slow", "3x"): {"dur": 3.0, "sparc": -0.45, "dtw": 0.03},
    ("band", "light"): {"dur": 1.4, "rom": 0.90, "sparc": -0.30, "dtw": 0.05},
    ("band", "medium"): {"dur": 1.8, "rom": 0.80, "sparc": -0.60, "dtw": 0.10},
    ("trunk_lean", "small"): {"lean": 5.0, "dtw": 0.03},
    ("trunk_lean", "large"): {"lean": 10.0, "dtw": 0.06},
    ("elbow_bend", "30deg"): {"elbow": -30.0, "dtw": 0.03},
    ("elbow_bend", "60deg"): {"elbow": -60.0, "dtw": 0.06},
    ("hike", "shrug"): {"hike": 0.04, "dtw": 0.01},
    ("rom_cap", "60deg"): {"cap": 60.0},
    ("rom_cap", "90deg"): {"cap": 90.0},
}

REST_DEG = 10.0          # resting arm angle; ranges are measured from here
DTW_ROM_GAIN = 1.0       # how much a range change raises the DTW distance

# Which rater label each condition is about (for the rater_labels subset)
RATER_KEYS = {"trunk_lean": "trunk_lean", "elbow_bend": "elbow_bend", "hike": "hike"}
RATER_FRACTION = 0.2     # share of reps in rated blocks that get a label
RATER_FLIP = 0.1         # chance the rater disagrees with the instruction

REP_COLUMNS = [
    "schema_version", "rep_id", "subject_id", "session_id", "block_id",
    "condition", "level", "side", "seg_method",
    "t_start_ms", "t_peak_ms", "t_end_ms", "duration_s",
    "rom_peak_deg", "rom_rel_personal", "time_to_peak_ratio", "sparc",
    "max_trunk_lean_deg", "min_elbow_deg", "max_hike_ratio",
    "dtw_dist", "dtw_band_frac", "ref_id",
    "valid_frame_frac", "rep_conf", "rater_labels",
]


def subject_params(rng):
    """One subject's typical values (their 'normal')."""
    return {
        "rom": rng.normal(90.0, 8.0),                        # deg from rest
        "dur": float(np.clip(rng.normal(2.0, 0.25), 1.4, 2.8)),  # s
        "ttp": rng.normal(0.5, 0.02),
        "sparc": rng.normal(-1.6, 0.08),
        "lean": abs(rng.normal(1.5, 0.5)),                   # deg
        "elbow": rng.normal(170.0, 3.0),                     # deg
        "hike": rng.normal(0.06, 0.015),
        "dtw": rng.normal(0.25, 0.05),
    }


def one_rep(rng, p, effect, session_shift):
    """Feature values of one rep for a subject p under an effect."""
    e = effect
    duration = p["dur"] * e.get("dur", 1.0) * np.exp(rng.normal(0.0, 0.07))
    rom = p["rom"] * e.get("rom", 1.0) + session_shift + rng.normal(0.0, 3.0)
    if "cap" in e:
        rom = min(rom, e["cap"] - REST_DEG + rng.normal(0.0, 1.5))
    dtw = (p["dtw"] * np.exp(rng.normal(0.0, 0.25))
           + DTW_ROM_GAIN * abs(rom / p["rom"] - 1.0)
           + e.get("dtw", 0.0))
    return {
        "duration_s": duration,
        "rom_peak_deg": rom,
        "time_to_peak_ratio": p["ttp"] + rng.normal(0.0, 0.04),
        "sparc": p["sparc"] + e.get("sparc", 0.0) + rng.normal(0.0, 0.08),
        "max_trunk_lean_deg": p["lean"] + e.get("lean", 0.0) + abs(rng.normal(0.0, 0.7)),
        "min_elbow_deg": p["elbow"] + e.get("elbow", 0.0) + rng.normal(0.0, 2.5),
        "max_hike_ratio": p["hike"] + e.get("hike", 0.0) + rng.normal(0.0, 0.025),
        "dtw_dist": dtw,
    }


def rater_label(rng, condition):
    """A JSON rater label for a subset of reps; NaN when the rep was not rated."""
    rated = condition in RATER_KEYS or condition == "normal"
    if not rated or rng.random() >= RATER_FRACTION:
        return np.nan
    labels = {}
    for cond, key in RATER_KEYS.items():
        truth = int(condition == cond)
        labels[key] = 1 - truth if rng.random() < RATER_FLIP else truth
    return json.dumps(labels)


def make_mock_reps(seed=SEED, n_subjects=N_SUBJECTS):
    """Build the mock reps table. Same seed -> same table."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        subject_id = f"sub{s + 1:02d}"
        p = subject_params(rng)
        for session_id in SESSIONS:
            session_shift = 0.0 if session_id == "s1" else rng.normal(0.0, 2.0)
            # calibration first, then the other blocks in a random order
            order = [BLOCKS[0]] + [BLOCKS[i] for i in 1 + rng.permutation(len(BLOCKS) - 1)]
            t = CLOCK_START_MS
            for block_id, condition, level in order:
                effect = EFFECTS[(condition, level)]
                for n in range(REPS_PER_BLOCK):
                    f = one_rep(rng, p, effect, session_shift)
                    dur_ms = f["duration_s"] * 1000.0
                    t_start = t
                    t_peak = t_start + f["time_to_peak_ratio"] * dur_ms
                    t_end = t_start + dur_ms
                    t = t_end + REST_BETWEEN_REPS_S * 1000.0

                    if rng.random() < 0.03:                    # a bad rep
                        vff = rng.uniform(0.5, 0.85)
                    else:
                        vff = 1.0 - abs(rng.normal(0.0, 0.02))
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "rep_id": f"{subject_id}_{session_id}_{block_id}_{n:03d}",
                        "subject_id": subject_id,
                        "session_id": session_id,
                        "block_id": block_id,
                        "condition": condition,
                        "level": level,
                        "side": "right",
                        "seg_method": "fsm",
                        "t_start_ms": t_start,
                        "t_peak_ms": t_peak,
                        "t_end_ms": t_end,
                        **f,
                        "dtw_band_frac": 0.15,
                        "ref_id": "pooled_mean",
                        "valid_frame_frac": vff,
                        "rep_conf": float(np.clip(vff - abs(rng.normal(0.0, 0.03)), 0.0, 1.0)),
                        "rater_labels": rater_label(rng, condition),
                    }
                    if rng.random() < 0.005:                   # unknown start boundary
                        row["t_start_ms"] = np.nan
                        row["duration_s"] = np.nan
                        row["time_to_peak_ratio"] = np.nan
                    rows.append(row)
                t += REST_BETWEEN_BLOCKS_S * 1000.0

    reps = pd.DataFrame(rows)

    # Personal centring: range relative to the subject's session-1 calibration median
    calib = reps[(reps["block_id"] == "normal_calib") & (reps["session_id"] == "s1")]
    rom_ref = calib.groupby("subject_id")["rom_peak_deg"].median()
    reps["rom_rel_personal"] = reps["rom_peak_deg"] / reps["subject_id"].map(rom_ref) - 1.0

    return reps[REP_COLUMNS]


def summary(reps):
    """Mean of the main features per condition and level."""
    cols = ["duration_s", "rom_peak_deg", "sparc", "max_trunk_lean_deg",
            "min_elbow_deg", "max_hike_ratio", "dtw_dist"]
    order = list(dict.fromkeys((c, l) for _, c, l in BLOCKS))
    return reps.groupby(["condition", "level"])[cols].mean().reindex(order).round(2)


def check(reps):
    """Sanity checks; raises AssertionError if the mock is not what we planned."""
    assert list(reps.columns) == REP_COLUMNS
    assert reps["rep_id"].is_unique
    assert reps["subject_id"].nunique() == N_SUBJECTS
    assert len(reps) == N_SUBJECTS * len(SESSIONS) * len(BLOCKS) * REPS_PER_BLOCK
    assert (reps["t_start_ms"].dropna() >= CLOCK_START_MS).all()

    m = reps.groupby(["condition", "level"])[["duration_s", "dtw_dist", "rom_peak_deg"]].mean()
    normal = m.loc[("normal", "none")]
    # slow is about 2x / 3x longer but its DTW distance stays close to normal
    assert 1.8 < m.loc[("slow", "2x"), "duration_s"] / normal["duration_s"] < 2.2
    assert 2.7 < m.loc[("slow", "3x"), "duration_s"] / normal["duration_s"] < 3.3
    assert m.loc[("slow", "3x"), "dtw_dist"] - normal["dtw_dist"] < 0.1
    # range caps hold and give a much larger DTW distance
    cap60 = reps[(reps["condition"] == "rom_cap") & (reps["level"] == "60deg")]
    assert cap60["rom_peak_deg"].max() < 60.0 - REST_DEG + 6.0
    assert m.loc[("rom_cap", "60deg"), "dtw_dist"] > normal["dtw_dist"] + 0.3
    # calibration block is centred on 0 by construction
    calib = reps[(reps["block_id"] == "normal_calib") & (reps["session_id"] == "s1")]
    assert abs(calib.groupby("subject_id")["rom_rel_personal"].median()).max() < 1e-9
    # missing is NaN, never 0
    assert (reps["duration_s"].dropna() > 0).all()


def main():
    print(f"seed = {SEED}")
    reps = make_mock_reps()
    check(reps)
    print(f"{len(reps)} reps, {reps['subject_id'].nunique()} subjects, "
          f"{reps['block_id'].nunique()} blocks, "
          f"{reps['condition'].nunique()} conditions")
    print(f"unknown start: {reps['t_start_ms'].isna().sum()} reps; "
          f"valid_frame_frac < 0.9: {(reps['valid_frame_frac'] < 0.9).sum()} reps; "
          f"rated: {reps['rater_labels'].notna().sum()} reps\n")
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(summary(reps))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    reps.to_parquet(OUT_PATH, index=False)
    back = pd.read_parquet(OUT_PATH)
    back["rater_labels"] = back["rater_labels"].where(back["rater_labels"].notna(), np.nan)
    pd.testing.assert_frame_equal(back, reps)
    print(f"\nsaved {OUT_PATH} (round trip OK)")


if __name__ == "__main__":
    main()
