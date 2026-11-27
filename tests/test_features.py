"""B6 tests: each feature validated on its own first, then rep_features() integration."""
import os

import numpy as np
import pandas as pd
import pytest

from neuroflow.segmentation.features import (
    peak_rom, rep_duration_s, rep_features, rep_speed, rep_window_extreme, sparc,
    time_to_peak_ratio)
from neuroflow.segmentation.segment import (
    BOUNDARY_FRAC, TOLERANCE_MS, calibration_refs, segment_reps, smooth)

FX = os.path.join(os.path.dirname(__file__), "..", "fixtures")
FS = 30.0
U0 = np.arccos(1 - 2 * BOUNDARY_FRAC) / (2 * np.pi)   # 5% point of a half-cosine


def _rep_speed(dur_s, amp=80.0, tremor=0.0):
    """Speed profile of one clean half-cosine rep (raise + lower), optional 6 Hz tremor."""
    t = np.arange(0, dur_s, 1 / FS)
    theta = amp * (1 - np.cos(2 * np.pi * t / dur_s)) / 2 + tremor * np.sin(2 * np.pi * 6 * t)
    return np.abs(np.gradient(theta, 1 / FS))


# ---- 1. duration ----

def test_duration_hand_example():
    assert rep_duration_s(1_000_000.0, 1_001_730.0) == pytest.approx(1.73)
    assert np.isnan(rep_duration_s(np.nan, 1_001_730.0))


@pytest.mark.parametrize("variant", ["normal", "slow", "capped", "leaning"])
def test_duration_matches_truth_on_mocks(variant):
    feats, truth = _mock_features(variant)
    true = (truth["t_end_ms"] - truth["t_start_ms"]).to_numpy() / 1000
    # each boundary is within +-67 ms, so a duration is within 2 x 67 ms
    assert np.abs(feats["duration_s"].to_numpy() - true).max() <= 2 * TOLERANCE_MS / 1000


# ---- 2. time-to-peak ratio ----

def test_ttp_hand_example():
    assert time_to_peak_ratio(0.0, 600.0, 2000.0) == pytest.approx(0.3)
    assert time_to_peak_ratio(0.0, 1000.0, 2000.0) == pytest.approx(0.5)


@pytest.mark.parametrize("peak_at", [0.3, 0.5, 0.7])
def test_ttp_recovers_asymmetric_rep(peak_at):
    # One rep warped so its peak lands at `peak_at` of the movement; the true ratio
    # between its 5% boundaries follows from the warp u -> u**a.
    a = np.log(0.5) / np.log(peak_at)
    dur = 2000.0
    t = np.arange(0, 6000, 1000 / FS)
    u = np.clip((t - 2000) / dur, 0, 1)
    frames = _block(t, 10 + 80 * (1 - np.cos(2 * np.pi * u ** a)) / 2)
    reps = segment_reps(frames, rom_ref=80.0, rest_ref=10.0)
    s, p, e = U0 ** (1 / a), 0.5 ** (1 / a), (1 - U0) ** (1 / a)
    true = (p - s) / (e - s)
    got = time_to_peak_ratio(reps["t_start_ms"], reps["t_peak_ms"], reps["t_end_ms"])
    assert len(reps) == 1 and abs(got[0] - true) < 0.03


# ---- 3. personal-centred peak range ----

def test_peak_rom_hand_example():
    t = np.arange(0, 3000, 1000 / FS)
    theta = 10 + 60 * (1 - np.cos(2 * np.pi * np.clip((t - 500) / 2000, 0, 1))) / 2   # peak 70
    rom, rel = peak_rom(t, smooth(t, theta), 500.0, 2500.0, rest_ref=10.0, rom_ref=80.0)
    assert rom == pytest.approx(60.0, abs=0.5)
    assert rel == pytest.approx(60 / 80 - 1, abs=0.01)                 # -0.25


def test_peak_rom_unknown_boundary():
    t = np.arange(0, 1000, 33.0)
    assert np.isnan(peak_rom(t, t, np.nan, 500.0, 10.0, 80.0)[0])


@pytest.mark.parametrize("variant", ["normal", "slow", "capped", "leaning", "low_rom"])
def test_peak_rom_matches_truth_on_mocks(variant):
    feats, truth = _mock_features(variant)
    true = (truth["peak_deg"] - truth["rest_deg"]).to_numpy()
    assert np.abs(feats["rom_peak_deg"].to_numpy() - true).mean() < 1.5   # degrees


# ---- 4. SPARC ----

def test_sparc_ignores_amplitude():
    assert sparc(_rep_speed(1.7, 40), FS) == pytest.approx(sparc(_rep_speed(1.7, 160), FS), abs=1e-9)


def test_sparc_robust_to_duration():
    vals = [sparc(_rep_speed(d), FS) for d in (1.7, 3.4, 6.8)]
    assert max(vals) - min(vals) < 0.02


def test_sparc_falls_with_tremor():
    vals = [sparc(_rep_speed(1.7, tremor=k), FS) for k in (0, 0.5, 1, 2, 4)]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_sparc_two_step_raise_is_less_smooth():
    t = np.arange(0, 2.0, 1 / FS)
    u = t / 2.0
    smooth_raise = 80 * (1 - np.cos(np.pi * u)) / 2
    two_step = np.where(u < 0.5, 40 * (1 - np.cos(4 * np.pi * u)) / 2,
                        40 + 40 * (1 - np.cos(4 * np.pi * (u - 0.5))) / 2)
    s1 = sparc(np.abs(np.gradient(smooth_raise, 1 / FS)), FS)
    s2 = sparc(np.abs(np.gradient(two_step, 1 / FS)), FS)
    assert s2 < s1 - 0.5


def test_sparc_rejects_bad_input():
    assert np.isnan(sparc([0.0, 0.0, 0.0], FS))
    assert np.isnan(sparc([1.0, np.nan, 1.0], FS))
    assert np.isnan(sparc([1.0], FS))


def test_rep_speed_uniform_grid():
    t = np.cumsum(np.r_[0, 33 + np.random.default_rng(0).normal(0, 3, 99)])   # jittered clock
    speed, fs = rep_speed(t, 2.0 * t / 1000.0, t[10], t[80])                  # 2 deg/s ramp
    assert fs == pytest.approx(1000 / np.median(np.diff(t)))
    assert np.allclose(speed, 2.0, atol=1e-6)


# ---- integration: rep_features ----

def test_rep_features_columns_and_values():
    cols = ["duration_s", "time_to_peak_ratio", "rom_peak_deg", "rom_rel_personal", "sparc"]
    normal, _ = _mock_features("normal")
    slow, _ = _mock_features("slow")
    capped, _ = _mock_features("capped")
    for f in (normal, slow, capped):
        assert set(cols) <= set(f.columns) and not f[cols].isna().any().any()
    assert slow["duration_s"].mean() / normal["duration_s"].mean() == pytest.approx(2.0, abs=0.1)
    assert normal["time_to_peak_ratio"].mean() == pytest.approx(0.5, abs=0.03)
    assert normal["rom_rel_personal"].mean() == pytest.approx(0.0, abs=0.05)
    assert capped["rom_rel_personal"].mean() == pytest.approx(35.1 / 79.6 - 1, abs=0.05)
    assert abs(slow["sparc"].mean() - normal["sparc"].mean()) < 0.2       # duration-robust


def test_rep_features_propagates_unknown_boundaries():
    frames = pd.read_parquet(os.path.join(FX, "mock_frames_normal.parquet"))
    reps = pd.DataFrame({"t_start_ms": [np.nan], "t_peak_ms": [1_003_000.0], "t_end_ms": [1_004_000.0]})
    out = rep_features(frames, reps, 10.0, 80.0)
    assert out[["duration_s", "time_to_peak_ratio", "rom_peak_deg", "sparc"]].isna().all().all()


# ---- 5. compensation helper (B7) ----

T5 = np.array([0.0, 100.0, 200.0, 300.0, 400.0])


def test_window_extreme_hand_examples():
    # the worked examples from the B7 briefing
    lean = [0.3, 4.1, 11.8, 6.0, 0.5]
    elbow = [176.0, 168.0, 151.0, 170.0, 174.0]
    assert rep_window_extreme(T5, lean, 0.0, 400.0, "max") == pytest.approx(11.8)
    assert rep_window_extreme(T5, elbow, 0.0, 400.0, "min") == pytest.approx(151.0)


def test_window_extreme_ignores_frames_outside_the_rep():
    sig = [99.0, 1.0, 2.0, 3.0, 99.0]          # spikes just outside [100, 300]
    assert rep_window_extreme(T5, sig, 100.0, 300.0, "max") == pytest.approx(3.0)
    assert rep_window_extreme(T5, sig, 100.0, 300.0, "min") == pytest.approx(1.0)


def test_window_extreme_nan_handling():
    sig = np.array([1.0, np.nan, 5.0, np.nan, 2.0])
    assert rep_window_extreme(T5, sig, 0.0, 400.0, "max") == pytest.approx(5.0)   # skips NaN frames
    assert np.isnan(rep_window_extreme(T5, sig, np.nan, 400.0, "max"))           # unknown boundary
    assert np.isnan(rep_window_extreme(T5, sig, 100.0, 100.0, "max"))            # only a NaN frame inside
    assert np.isnan(rep_window_extreme(T5, sig, 1000.0, 2000.0, "min"))          # no frame inside


def test_window_extreme_min_valid_frac():
    sig = np.array([170.0, np.nan, np.nan, np.nan, 160.0])        # 2 of 5 frames valid
    assert rep_window_extreme(T5, sig, 0.0, 400.0, "min") == pytest.approx(160.0)
    assert np.isnan(rep_window_extreme(T5, sig, 0.0, 400.0, "min", min_valid_frac=0.5))
    assert rep_window_extreme(T5, sig, 0.0, 400.0, "min", min_valid_frac=0.4) == pytest.approx(160.0)


def test_rep_features_needs_half_the_frames_valid():
    """Like the G1 clips: elbow valid on only a few frames of the rep -> NaN, not a 1-frame min."""
    frames = pd.read_parquet(os.path.join(FX, "mock_frames_elbow_30.parquet"))
    rest, rom = CALIB["mock01"]
    reps = segment_reps(frames, rom, rest)
    sparse = frames.copy()
    keep = np.zeros(len(frames), bool)
    keep[::3] = True                                               # ~33 % valid
    sparse.loc[~keep, "elbow_deg"] = np.nan
    assert rep_features(sparse, reps, rest, rom)["min_elbow_deg"].isna().all()
    half = frames.copy()
    keep = np.zeros(len(frames), bool)
    keep[::3] = True
    keep[1::3] = True                                              # ~67 % valid
    half.loc[~keep, "elbow_deg"] = np.nan
    assert not rep_features(half, reps, rest, rom)["min_elbow_deg"].isna().any()


def test_window_extreme_rejects_bad_how():
    with pytest.raises(ValueError):
        rep_window_extreme(T5, np.ones(5), 0.0, 400.0, "mean")


@pytest.mark.parametrize("variant,col,how,expect,normal_expect", [
    ("leaning", "trunk_lean_deg", "max", 12.0, 0.0),
    ("elbow_30", "elbow_deg", "min", 145.0, 175.0),
    ("hike", "hike_ratio", "max", 0.20, 0.0),
])
def test_window_extreme_on_mocks(variant, col, how, expect, normal_expect):
    """On the truth boundaries, each compensation mock gives its planted size; normal does not."""
    def per_rep(v):
        f = pd.read_parquet(os.path.join(FX, f"mock_frames_{v}.parquet"))
        tr = pd.read_csv(os.path.join(FX, f"mock_frames_{v}_truth.csv"))
        t = f["t_capture_ms"].to_numpy()
        sig = smooth(t, f[col].to_numpy())
        return np.array([rep_window_extreme(t, sig, r.t_start_ms, r.t_end_ms, how)
                         for r in tr.itertuples()])
    tol = 0.1 * abs(expect - normal_expect)   # 10% of the planted effect
    assert per_rep(variant).mean() == pytest.approx(expect, abs=tol)
    assert per_rep("normal").mean() == pytest.approx(normal_expect, abs=tol)


# ---- 6. compensation integration (B7) ----

COMP = ["max_trunk_lean_deg", "min_elbow_deg", "max_hike_ratio"]


def test_rep_features_compensation_on_mocks():
    """Each compensation mock moves only its own feature (the plan's B7 acceptance check)."""
    normal, _ = _mock_features("normal")
    leaning, _ = _mock_features("leaning")
    elbow, _ = _mock_features("elbow_30")
    hike, _ = _mock_features("hike")
    for f in (normal, leaning, elbow, hike):
        assert not f[COMP].isna().any().any()
    assert leaning["max_trunk_lean_deg"].mean() == pytest.approx(12.0, abs=1.0)
    assert normal["max_trunk_lean_deg"].mean() < 2.0
    assert elbow["min_elbow_deg"].mean() == pytest.approx(145.0, abs=1.5)
    assert normal["min_elbow_deg"].mean() > 172.0
    assert hike["max_hike_ratio"].mean() == pytest.approx(0.20, abs=0.02)
    assert normal["max_hike_ratio"].mean() < 0.03
    # isolation: a compensation leaves the other two at their normal values
    for col in ("min_elbow_deg", "max_hike_ratio"):
        assert leaning[col].mean() == pytest.approx(normal[col].mean(), rel=0.05)
    assert elbow["max_trunk_lean_deg"].mean() == pytest.approx(normal["max_trunk_lean_deg"].mean())
    assert hike["min_elbow_deg"].mean() == pytest.approx(normal["min_elbow_deg"].mean())


def test_rep_features_lean_is_absolute():
    frames = pd.read_parquet(os.path.join(FX, "mock_frames_leaning.parquet"))
    flipped = frames.assign(trunk_lean_deg=-frames["trunk_lean_deg"])
    rest, rom = CALIB["mock01"]
    reps = segment_reps(frames, rom, rest)
    a = rep_features(frames, reps, rest, rom)["max_trunk_lean_deg"]
    b = rep_features(flipped, reps, rest, rom)["max_trunk_lean_deg"]
    assert np.allclose(a, b)


def test_rep_features_compensation_missing_is_nan():
    frames = pd.read_parquet(os.path.join(FX, "mock_frames_normal.parquet"))
    rest, rom = CALIB["mock01"]
    reps = segment_reps(frames, rom, rest)
    # column absent (A4 v0) -> NaN
    out = rep_features(frames.drop(columns=["hike_ratio"]), reps, rest, rom)
    assert out["max_hike_ratio"].isna().all() and not out["min_elbow_deg"].isna().any()
    # every elbow frame of rep 0 missing -> NaN for that rep only, not an interpolated value
    t = frames["t_capture_ms"]
    r0 = reps.iloc[0]
    gap = frames.copy()
    gap.loc[(t >= r0["t_start_ms"]) & (t <= r0["t_end_ms"]), "elbow_deg"] = np.nan
    out = rep_features(gap, reps, rest, rom)
    assert np.isnan(out["min_elbow_deg"].iloc[0]) and not out["min_elbow_deg"].iloc[1:].isna().any()
    # unknown boundary -> NaN
    unk = pd.DataFrame({"t_start_ms": [np.nan], "t_peak_ms": [1_003_000.0], "t_end_ms": [1_004_000.0]})
    assert rep_features(frames, unk, rest, rom)[COMP].isna().all().all()


# ---- helpers ----

CALIB = calibration_refs()


def _mock_features(variant):
    frames = pd.read_parquet(os.path.join(FX, f"mock_frames_{variant}.parquet"))
    truth = pd.read_csv(os.path.join(FX, f"mock_frames_{variant}_truth.csv"))
    rest, rom = CALIB[frames["subject_id"].iloc[0]]
    return rep_features(frames, segment_reps(frames, rom, rest), rest, rom), truth


def _block(t, theta):
    """Minimal frames table with the columns the segmenter needs."""
    return pd.DataFrame({"subject_id": "syn", "session_id": "s1", "block_id": "b", "condition": "x",
                         "level": "x", "t_capture_ms": t, "theta_ht_deg": theta})


# ---- 7. shoulder-line tilt, rest-corrected (exploratory, #6 / #14) ----
from neuroflow.segmentation.features import (  # noqa: E402
    TILT_REST_MIN_FRAMES, rest_corrected_tilt, resting_tilt, wrap_deg)


def test_wrap_deg():
    assert list(wrap_deg([0, 89, 90, -91, 179])) == [0, 89, -90, 89, -1]


def test_resting_tilt_window_and_minimum():
    t = np.arange(0, 2000, 40.0)                       # 25 fps
    tilt = np.where(t < 1000, 3.0, 10.0)               # 3 deg at rest, 10 during the rep
    assert resting_tilt(t, tilt, 1000.0) == 3.0        # the 0.5 s before the start
    assert np.isnan(resting_tilt(t, tilt, np.nan))
    # previous rep ended 100 ms before this start: too few rest frames left
    assert np.isnan(resting_tilt(t, tilt, 1000.0, t_prev_end_ms=900.0))
    gappy = tilt.copy()
    gappy[(t >= 500) & (t < 920)] = np.nan             # only 2 valid rest frames left
    assert (~np.isnan(gappy[(t >= 500) & (t < 1000)])).sum() < TILT_REST_MIN_FRAMES
    assert np.isnan(resting_tilt(t, gappy, 1000.0))


def test_rest_corrected_tilt_ignores_resting_offset():
    t = np.arange(0, 3000, 40.0)
    tilt = np.full(len(t), 4.0)                        # posture/camera: 4 deg at rest
    rep = (t >= 1000) & (t <= 2000)
    tilt[rep] += 6.0 * np.sin(np.pi * (t[rep] - 1000) / 1000)   # +6 deg shrug mid-rep
    v = rest_corrected_tilt(t, tilt, tilt, 1000.0, 2000.0)
    assert v == pytest.approx(6.0, abs=0.1)            # not 10: the 4 deg offset is removed
    assert np.isnan(rest_corrected_tilt(t, tilt, tilt, 1000.0, np.nan))


def test_rep_features_tilt_column():
    frames = pd.read_parquet(os.path.join(FX, "mock_frames_normal.parquet"))
    rest, rom = CALIB[frames["subject_id"].iloc[0]]
    reps = segment_reps(frames, rom, rest)
    # no shoulder_tilt_deg column (as in G1's frames.parquet v1) -> NaN, never 0
    assert rep_features(frames, reps, rest, rom)["max_shoulder_tilt_deg"].isna().all()
    f = frames.sort_values("t_capture_ms").copy()
    t = f["t_capture_ms"].to_numpy()
    tilt = np.full(len(f), 3.0)                         # resting offset
    for _, r in reps.iterrows():                        # +5 deg tilt peak inside every rep
        inside = (t >= r["t_start_ms"]) & (t <= r["t_end_ms"])
        tilt[inside] += 5.0 * np.sin(np.pi * (t[inside] - r["t_start_ms"]) / (r["t_end_ms"] - r["t_start_ms"]))
    f["shoulder_tilt_deg"] = tilt
    v = rep_features(f, reps, rest, rom)["max_shoulder_tilt_deg"]
    assert v.notna().all()
    assert v.to_numpy() == pytest.approx(5.0, abs=0.3)  # offset removed, shrug kept
