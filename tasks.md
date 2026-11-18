# Member B: What We Did and Why

Last updated: 2026-11-18. Status of every task is in `task_status.md`; this file explains the work behind it.

## Setup

- **Branch.** `b1-mock-frames`, created from `main` (`d3aa726`) and pushed. The `origin` remote is HTTPS with no stored login, so push over SSH: `git push git@github.com:AdastraJAP/NeuroFlow.git <branch>`.
- **Environment.** `.venv` with Python 3.12, numpy, pandas, scipy, pyarrow, matplotlib, OpenCV, mediapipe and pytest. Activate with `source .venv/bin/activate`. MediaPipe fails on WSL until `sudo apt install -y libgles2` is run; B's tasks don't need it.
- **Legacy code rule.** From Member A's prototype (`nf_core.py`, `step*.py`) we reuse only parts that are sound, and fix a faulty part only when we actually use it. Sound: `angle_at`, OpenCV clip reading. Not reused: `dtw_distance` (no band, raw degrees), visibility gating, dropping unusable frames, frame-index timing.
- **Old traces.** `traces/*.csv` are real single-rep angle traces. Their `frame` column counts only usable frames, so it is not true time. They can serve as a rough real-data check for B2/B3.

## B1: mock frames generator

**File:** `fixtures/make_mock_frames.py` · **Run:** `python fixtures/make_mock_frames.py`

**Why:** B can't wait for A's real data. A mock with known rep times lets us build and test the segmenter now.

**What it makes:** four files, `fixtures/mock_frames_<variant>.parquet`, each with a `_truth.csv` listing the true start, peak and end of all 10 reps. The truth CSVs are B2's golden expected output.

| Variant | What changes | Tests |
| --- | --- | --- |
| normal | ~2 s reps, peak ~90° | baseline |
| slow | reps 2× longer | tempo |
| capped | peak ~45° | low ROM (fixed thresholds like Pavlikov's 55° fail here) |
| leaning | trunk lean rises to ~12° in each rep | B7 lean feature |

**How a rep is made:** a raised half-cosine, `rest + (peak − rest)·(1 − cos 2πu)/2`, where u goes from 0 to 1 over the rep. For u = 0, .25, .5, .75, 1 with rest 10° and peak 90° it gives 10, 50, 90, 50, 10.

**Details that matter:**
- Columns follow the `frames.parquet` contract (schema version 1, 78 columns).
- The clock starts at 1,000,000 ms, not 0, so code that assumes time starts at zero breaks here rather than on real data.
- Frame times jitter by about ±3 ms around 30 fps, so code must use `t_capture_ms`, not frame counts.
- Landmark columns exist but are `NaN`: a mock has no pose, and the contract says missing is `NaN`, never 0.
- Reps are symmetric, so the true time-to-peak ratio is exactly 0.5, which makes B6 easy to check by hand.
- Truth `t_start_ms` / `t_end_ms` follow the 5 % boundary definition below. For a half-cosine that point is at u = 0.0718 (and 1 − 0.0718), about 0.29 of a second inside a 2 s rep. `t_onset_ms` / `t_offset_ms` (first and last movement) are kept for reference only.
- θ_V = θ_HT + lean is a first-order inference from the research brief, used only in the mock.
- Noise 1°, rest 10°, seed 42 (printed). These are mock choices, not literature values.

**Check:** `main()` asserts timestamps strictly increase, reps don't overlap, the angle near each true peak matches the planned peak, capped stays below 55°, and only the leaning variant has lean above 3°. It also checks the parquet round trip.

## Rep boundary definition (applies to B1, B2, B9)

**Rep start** = the moment the angle rises through **rest + 5 % of that rep's range**; **rep end** = the moment it falls back through the same level. Range = peak angle − rest angle, per rep. The value lives in one place, `BOUNDARY_FRAC = 0.05` in `neuroflow/segmentation/segment.py`, and the B1 generator imports it from there.

**Why:** the very first movement can't be timed. A smooth rep starts flat: on a 2 s, 80° rep the angle needs 101 ms to rise even 0.5° (half the noise), and 202 ms on a slow rep. Our first attempt at B2, measured against first-movement truth, was always 100–270 ms late on starts and early on ends. The 5 % level (about 4° on a normal rep) sits clear of the noise, so it can be timed and the ±67 ms tolerance stays a real test. 5 % is our own choice, not a literature value.

**B9 annotations:** manual labels must mark the same 5 % point, not "first visible movement", or B8 would compare two different definitions.

## B2: fixed-threshold segmenter

**File:** `neuroflow/segmentation/segment.py` · **Run:** `python -m neuroflow.segmentation.segment` · **Test:** `python -m pytest tests/`

**Idea:** hysteresis means two thresholds. A rep opens when the angle rises above `T_high = 30°` and closes when it falls below `T_low = 20°`. The 10° gap stops noise near one threshold from opening and closing reps repeatedly.

**How it works:**
1. **Smooth** θ_HT with Savitzky-Golay (0.25 s window, order 2, set in seconds from real timestamps). This stands in for A's `smooth_savgol` until it exists.
2. **Detect** reps with a 4-state machine: REST → RISING (above 30°) → PEAK (stopped rising) → RETURNING (falling) → REST (below 20°). The peak is the highest frame of the rep.
3. **Place boundaries.** Estimate the rest level as the median smoothed angle of the rest frames before the rep (and after it, for the end). Walk outward from the threshold crossing until the angle drops to rest + 5 % of range, then interpolate between the two frames either side, so times are not locked to the frame grid. If no rest is visible, the boundary is `NaN`.

**Why 30°/20° works on the capped mock:** capped reps peak at about 45–49°, above 30°. A fixed 55° threshold (Pavlikov) would miss all of them. B3 replaces fixed thresholds with fractions of the user's calibrated ROM.

**Validation on the four committed mocks (tolerance ±67 ms, signed error = detected − truth):**

| Variant | Reps found | Start error mean / SD / max abs | End error mean / SD / max abs | Pass |
| --- | --- | --- | --- | --- |
| normal | 10/10 | +6.6 / 13.2 / 37.8 ms | −0.1 / 14.6 / 29.1 ms | yes |
| slow | 10/10 | −0.6 / 23.2 / 43.5 ms | −5.3 / 19.7 / 38.1 ms | yes |
| capped | 10/10 | +14.8 / 23.1 / 59.8 ms | −2.4 / 27.4 / 50.6 ms | yes |
| leaning | 10/10 | +6.6 / 13.2 / 37.8 ms | −0.1 / 14.6 / 29.1 ms | yes |

For reference, raw threshold crossings (no boundary step) are 200–430 ms late on starts.

Leaning gives the same numbers as normal because θ_HT doesn't change with trunk lean, which is the reason we use θ_HT. Lean only shows up in `theta_v_deg` and `trunk_lean_deg`, which B7 uses.

**Robustness check (50 random mocks per variant):** a block passes only if *every* rep is within ±67 ms.

| Variant | Count correct | Blocks passing | Worst error: median / 90th pct / max |
| --- | --- | --- | --- |
| normal | 50/50 | 50/50 | 22 / 26 / 38 ms |
| slow | 50/50 | 49/50 | 43 / 55 / 77 ms |
| capped | 50/50 | 46/50 | 52 / 64 / 82 ms |

**Known limit:** timing error grows when the angle crosses the 5 % level slowly: slow reps, and small-range reps where 5 % of 35° is only 1.75°. Rep counting is never wrong. Recheck this on real data in B3 and B8.

## Decisions log

| Date | Decision |
| --- | --- |
| 2026-11-18 | Golden criterion: start and end within ±2 frames (±67 ms at 30 fps) |
| 2026-11-18 | Reuse legacy code only where sound, fix only when used; don't post the review of A's code |
| 2026-11-18 | B2 starts with T_high = 30°, T_low = 20° (to be tuned on training subjects only) |
| 2026-11-18 | B1 stays uncommitted until B2 validates against all four truth files |
| 2026-11-18 | Rep start/end = crossing of rest + 5 % of the rep's range; ±67 ms tolerance kept; same definition for B1 truth, B2 and B9 |
| 2026-11-18 | B2 validated: all four mocks pass (10/10 reps, max error 59.8 ms) |
