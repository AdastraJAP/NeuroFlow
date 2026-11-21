# Member B: What We Did and Why

Last updated: 2026-11-21. Status of every task is in `task_status.md`; this file explains the work behind it.

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

## B3: ROM-normalised segmenter (S3 core)

**Branch:** `b3-segmenter-v1` (from `b1-mock-frames`) · **File:** `neuroflow/segmentation/segment.py`, functions `calibrate()` and `segment_reps()` · **Test:** `python -m pytest tests/`

**Why:** fixed thresholds fail for a user whose whole range sits below them. With v0's 30° (or Pavlikov's 55°), someone who can only lift to 25° never opens a rep. Fractions of the user's own range always can.

**How it works:**
1. **Calibrate** on a separate Normal block (in the protocol, session 1's calibration block), giving `rest_ref` and `rom_ref`.
   - Rest = 10th percentile of the smoothed angle; rough range = 98th percentile − rest.
   - Detect reps at 50 % / 40 % of that rough range.
   - `rom_ref` = median rep peak − rest. This self-scales, so it works for small ranges too.
2. **Thresholds:** `T_high = rest_ref + 20 % of rom_ref`, `T_low = rest_ref + 15 % of rom_ref`. Same state machine as B2.
3. **Filters:** drop a detected rep if its peak rises less than 30 % of `rom_ref` above rest, or if it lasts under 0.4 s between its boundaries.
4. **Boundaries:** the same 5 % rule as B2, unchanged. Rest frames are now those below the scaled `T_low`.

The values 20 % / 15 % / 30 % / 0.4 s are design inferences, not literature values; tune them on training subjects only (sensitivity goes in B8).

**New mock (B1 generator):** `low_rom`, a different user (`mock02`) whose reps peak at about 25° (rest 10°, range about 15°). The four existing mock files are byte-identical; the generator change only adds this variant.

**Calibration in the tests:** a Normal block generated with a different seed (7), never the block under test. Results: `mock01` rest 9.6°, rom 79.6° → T_high 25.5°, T_low 21.5°. `mock02` rest 9.6°, rom 15.3° → T_high 12.6°, T_low 11.9°.

**Results on the committed mocks:**

| Variant | v0 fixed: reps / pass | v1 ROM: reps / pass | v1 worst start / end error |
| --- | --- | --- | --- |
| normal | 10/10 / yes | 10/10 / yes | 37.2 / 29.1 ms |
| slow | 10/10 / yes | 10/10 / yes | 43.5 / 38.1 ms |
| capped | 10/10 / yes | 10/10 / yes | 59.4 / 47.3 ms |
| leaning | 10/10 / yes | 10/10 / yes | 37.2 / 29.1 ms |
| low_rom | **0/10** / no | **10/10** / no (timing) | 110.3 / 82.7 ms |

**Robustness (50 random mocks per variant):**

| Variant | Count correct | Boundary pass |
| --- | --- | --- |
| normal | 50/50 | 50/50 |
| slow | 50/50 | 49/50 |
| capped | 50/50 | 46/50 |
| leaning | 50/50 | 50/50 |
| low_rom | 50/50 | 1/50 (worst error median 112 ms, max 641 ms) |

**False alarms:** 0 reps in 50 rest-only blocks (1° noise, about 37 s each), with either calibration.

**Known limit (extends B2's):** on a very small range, the 5 % boundary level (about 0.8° on 15°) is below the noise, so boundary timing is unreliable. Rep counting still works. The 5 % definition stays as approved; recheck on real low-range clips in B8.

**Tests added:** v1 rep count on all five mocks; v1 boundaries within ±67 ms on the four original mocks (low_rom excluded on purpose, see the known limit); v0 finds 0 reps on low_rom; no false reps on a rest-only block.

## B4: DTW with a Sakoe-Chiba band

**File:** `neuroflow/segmentation/dtw.py`, function `dtw_distance(x, y, band_frac=0.15)` · **Run:** `python -m neuroflow.segmentation.dtw` · **Test:** `python -m pytest tests/test_dtw.py`

**Idea:** DTW lines up two series when one runs faster or slower in places. It fills a cost table `D[i, j] = (x[i] − y[j])² + min(D[i−1, j−1], D[i−1, j], D[i, j−1])`, and the distance is `sqrt(D[n, m])`. This is the same convention as the `dtaidistance` library, so the two can be compared exactly.

**Band:** cells must stay near the diagonal, so a point can shift by at most `w = round(band_frac × longer length)` samples. For unequal lengths, the length difference is added on the side where the longer series needs room. `band_frac=None` means unconstrained; `0` means diagonal only, which equals the Euclidean distance.

**Worked example:** x = [0, 1, 2, 1, 0], y = [0, 0, 1, 2, 1] (y is x shifted one sample to the right).
- Diagonal only: differences 0, 1, 1, −1, −1, so the distance is √4 = 2.
- 20 % band (w = 1) or unconstrained: the path (0,0)(0,1)(1,2)(2,3)(3,4)(4,4) costs only 1.

**Checks (all pass):**
- The hand example above.
- Identical series give 0; band 0 equals the Euclidean distance.
- A wider band never increases the distance (20 random pairs × 6 band settings).
- Matches `dtaidistance` 2.5.1 exactly (max difference 0.0) on 200 random pairs of unequal length × 4 band settings. Note that `dtaidistance`'s `window` equals our `w + 1`.
- NaN or empty input raises an error, so gaps must be filled or dropped first.

**What the band does on a rep:** a rep with its peak at 30 % instead of 50 % (same range, 60 samples) scores 0.135 unconstrained or with a 20 % band, because the warping absorbs a 12-sample shift. With a 10 % band it scores 0.957. This is why band width is a P2 factor: too wide and DTW is blind to timing.

**Bug found and fixed during B4:** my first band rule widened the band symmetrically for unequal lengths. It disagreed with `dtaidistance` on unequal-length pairs; adopting its rule (extra room only on the longer side) gives an exact match.

**Not decided here, left for B5/B6:** DTW runs on θ_HT divided by the personal peak so amplitude doesn't dominate (plan pitfall), with amplitude kept as its own feature. Reps should be resampled to a common length first; the band then limits within-rep timing distortion, and overall slowness goes into the duration feature.

**Speed:** about 1 ms per 60 × 60 pair with a 15 % band (plain Python loops).

**Dependency:** `dtaidistance` is used only by the test; it's skipped if not installed.

## B5: rep templates

**File:** `neuroflow/segmentation/templates.py` (plus `dtw_path` added to `dtw.py`) · **Run:** `python -m neuroflow.segmentation.templates` · **Test:** `python -m pytest tests/test_templates.py`

### Step 1: linear-stretch mean (RehabHub-style reference)

**How it works** (`rep_matrix`, `mean_template`):
1. Cut each rep from the smoothed θ_HT at its 5 % boundaries (from the B3 segmenter).
2. Normalise to the user's range: `(θ − rest_ref) / rom_ref`, so rest = 0 and the user's typical peak = 1. This is the plan's guard against amplitude dominating DTW. It's per person, not per rep, so a capped rep still looks smaller.
3. Stretch each rep to 60 samples with `np.interp` on real timestamps (`resample_rep`). One valid frame either side of the rep is included, so the first and last samples are interpolated, not clamped.
4. Template = sample-by-sample mean.

`TEMPLATE_LEN = 60` (about one 2 s rep at 30 fps) is a design choice. Templates must come from training subjects only.

**Validation against the exact true shape** (known for the mocks; error in units of the user's range):

| Mock | Error vs true shape | Correlation | Peak position (ideal 29.5) | Ends (expected) |
| --- | --- | --- | --- | --- |
| normal | 0.0048 (≈0.4°) | 0.9999 | 29 | 0.056 (0.055) |
| slow | 0.0020 | 1.0000 | 29 | 0.059 |
| capped | 0.0039 | 0.9998 | 29, height 0.44 | 0.028 |
| low_rom | 0.0322 (≈0.5°) | 0.996 | 27 | 0.076 (0.076) |

- Slow and normal templates are almost identical (DTW 0.036): stretching removes overall tempo, so duration must be a separate feature (B6).
- The capped template stays clearly different (DTW 2.46): per-person normalisation keeps the smaller range visible.

**Bug found and fixed:** `resample_rep` first used only frames inside the rep, so the first and last samples were clamped to the nearest inside frame rather than interpolated at the boundary. A unit test on a straight line caught it. After the fix, the template ends match the expected 5 % level.

### Step 2: DBA (DTW Barycenter Averaging)

**How it works** (`dba_template`): start from the mean template, then repeat until it stops improving (at most 20 rounds):
1. Align every rep to the current template with banded DTW (`dtw_path`, band 15 %).
2. Replace each template sample with the mean of all rep samples aligned to it.

Each round can only lower or keep the total squared DTW distance, so DBA is never worse than the mean it starts from on that measure. The tests check this.

**Results** (spread = mean DTW distance from each rep to the template):

| Data | Mean spread | DBA spread | Gain | Mean peak | DBA peak |
| --- | --- | --- | --- | --- | --- |
| normal | 0.102 | 0.088 | 14 % | 1.008 | 1.020 |
| slow | 0.106 | 0.089 | 16 % | 1.010 | 1.027 |
| capped | 0.083 | 0.063 | 24 % | 0.444 | 0.460 |
| low_rom | 0.163 | 0.132 | 19 % | 1.002 | 1.043 |
| timing-varied (synthetic) | 0.199 | 0.104 | 48 % | 0.929 | 1.000 |

The synthetic set has 20 unit-height reps whose peak falls anywhere between 35 % and 65 % of the rep. The committed mocks all peak at exactly 50 %, so they can't show what DBA is for.

**Finding: DBA's advantage depends on the data.**
- **Timing varies between reps** (the realistic case): DBA keeps the true peak height (1.00, where the mean blurs it to 0.93) and is closer to the typical rep.
- **All reps share the same timing** (our mocks): DBA fits the noise. Its error vs the true shape is 0.9–1.2° against 0.2–0.4° for the mean, with peaks 1–3 % too high. This overshoot grows with noise (synthetic, noise 0.03: peak 1.014).
- So "DBA beats mean on distance to members" holds everywhere, but that doesn't make it more correct. Which template scores better is P2's "reference" factor, decided on real data; B5 provides both.

**Speed:** mean template is instant; DBA on 10 reps takes under a second, but on a pooled-size set of 240 reps about 18 s (hits the 20-round cap). In C's leave-one-subject-out P2 grid that's about 5 minutes per configuration (18 folds), worth telling C.

**`dtw_path`** (new in `dtw.py`): returns the distance and the warping path, preferring the diagonal on ties. On the B4 hand example it gives exactly the hand-worked path. On 50 random pairs the path's cost equals the distance and every step is valid. `dtw_distance` is unchanged (B4 tests pass).

## B6: per-rep features

**File:** `neuroflow/segmentation/features.py` · **Run:** `python -m neuroflow.segmentation.features` · **Test:** `python -m pytest tests/test_features.py`

Each feature is a separate function, validated on its own (on hand examples and synthetic reps with known answers) before `rep_features(frames, reps, rest_ref, rom_ref)` combines them into `reps.parquet` columns. Note that the contract signature was `rep_features(frames, reps)`; `rest_ref` and `rom_ref` (the user's calibration) were added because the personal-centred range needs them.

| Column | Definition | Choice to note |
| --- | --- | --- |
| `duration_s` | (end − start) / 1000, between the 5 % boundaries | Kept absolute; dividing by the user's calibration median duration is left to the scorer (C) |
| `time_to_peak_ratio` | (peak − start) / (end − start) | 0.5 = symmetric rep |
| `rom_peak_deg` | peak smoothed θ_HT inside the rep − rest_ref | Range = degrees moved from rest, not the absolute peak angle |
| `rom_rel_personal` | rom_peak_deg / rom_ref − 1 | Ratio form, so 0 = the user's typical range, −0.5 = half; comparable across users |
| `sparc` | spectral arc length of the absolute angular speed (Balasubramanian et al. 2015) | Whole rep (raise + lower), on the smoothed angle resampled to an even grid; zero-padding 4, cut-off 10 Hz, threshold 0.05 as in the authors' reference code from memory, **to verify against the original before citing** |

### Validation, feature by feature

**Duration:**
- Hand example: 1,730 ms gives 1.73 s; an unknown boundary gives NaN.
- On the four mocks, every rep is within 2 × 67 ms of the truth, since each boundary is within ±67 ms.

**Time-to-peak ratio:**
- Hand examples: 600 / 2000 = 0.3, 1000 / 2000 = 0.5.
- On synthetic asymmetric reps with the peak at 30 %, 50 % or 70 % of the movement, run through the B3 segmenter, it recovers the true ratio between the 5 % boundaries within 0.03.

**Peak range:**
- Hand example: a rep from 10° to 70° gives 60° (within 0.5°) and −0.25 personal.
- On all five mocks, the mean error vs the true range is under 1.5°. Mock results: normal 80.5° (true 80.1°), slow 80.8° (80.1°), capped 35.7° (35.1°), low_rom 15.9° (15.0°). The small upward bias comes from the slightly low rest estimate (9.6° vs 10°) and noise at the peak.

**SPARC** (on clean synthetic reps):
- Ignores amplitude: identical at ½× and 2× height.
- Duration-robust: −2.342, −2.350 and −2.350 at 1.7 s, 3.4 s and 6.8 s.
- Falls steadily with 6 Hz tremor: 0 → 0.5 → 1 → 2 → 4° gives −2.34, −2.70, −2.85, −3.18, −3.57.
- A two-step raise scores −2.76 against −1.55 for a smooth raise.
- Flat, NaN or too-short input returns NaN.
- `rep_speed` resamples to an even grid correctly on a jittered clock (a 2°/s ramp gives exactly 2°/s).

### Integration (`rep_features` on the mocks)

| Mock | Duration (true) | Time to peak | Range (true) | Personal range | SPARC |
| --- | --- | --- | --- | --- | --- |
| normal | 1.73 s (1.73) | 0.493 | 80.5° (80.1) | +0.01 | −2.68 |
| slow | 3.46 s (3.46) | 0.502 | 80.8° (80.1) | +0.01 | −2.82 |
| capped | 1.72 s (1.73) | 0.501 | 35.7° (35.1) | −0.55 | −2.97 |
| low_rom | 1.69 s (1.73) | 0.498 | 15.9° (15.0) | +0.04 | −3.54 |

Integration tests check: all five columns are present with no NaN on the mocks; slow / normal duration ≈ 2; time to peak ≈ 0.5; personal range ≈ 0 (normal) and ≈ −0.56 (capped); SPARC normal ≈ slow; unknown boundaries propagate NaN.

### Known limit: SPARC confuses noise with jerkiness on small movements

Without noise, every mock scores exactly −2.421, whatever its range or duration. With 1° of angle noise: normal −2.68, capped −2.97, low_rom −3.54. Noise is a larger share of a small movement, and SPARC can't tell noise from tremor. On a clean full rep, 2° of noise shifts SPARC about as much as 0.5° of real tremor.

So a capped or low-range rep will look less smooth from noise alone. This matters for P2/P3 (the band condition is meant to show up as lower smoothness). The definition is unchanged; options for later are stronger smoothing before SPARC, comparing SPARC only between conditions with similar range, or letting the scorer account for range. Check on real webcam noise first, as the plan says.

### Not in B6 (for later tasks)

- `dtw_dist`, `dtw_band_frac`, `ref_id`: need a template fitted on training subjects only, so they're computed inside C's leave-one-subject-out loop using B4/B5 functions.
- `valid_frame_frac`, `rep_conf`: need A's confidence flags.
- Compensation features: B7 (done, below).

## B7: compensation features

**File:** `neuroflow/segmentation/features.py` (`rep_window_extreme`, `COMPENSATION`, added to `rep_features()`) · **Test:** `python -m pytest tests/test_features.py`

**Idea:** no geometry here. A4 writes one value per frame for `trunk_lean_deg`, `elbow_deg` and `hike_ratio`. B7 reduces each one to a single number per rep between the 5 % boundaries, the same windowing as `peak_rom()`.

| Column | Per rep | Note |
| --- | --- | --- |
| `max_trunk_lean_deg` | max of \|lean\| | absolute value: how A4 signs lean is still open (ask A) |
| `min_elbow_deg` | min | 180° = straight arm |
| `max_hike_ratio` | max | exploratory (weak evidence) |

**Details:** each signal is smoothed first (`segment.smooth`), because the max of raw noise is biased upward. NaN frames are put back after smoothing, so a rep with no valid frames gives NaN instead of an interpolated value. A column missing from frames (e.g. A4 v0) gives NaN, never 0. The literature flags (5.2°, 153.4°, 0.174) belong to C4, not here.

**New mocks (B1 extension):** `elbow_30` (elbow 175° → ~145° at each peak) and `hike` (hike → ~0.20), named after C's blocks. Same seed and no extra random draws, so the five older mocks are byte-identical and θ_HT is identical to normal.

**Results (B3 segmenter):** normal 0.8° / 173.7° / 0.012; leaning lean 12.3°; elbow_30 elbow 144.6°; hike 0.203. Each mock moves only its own feature.

## B9: manual labelling helper

**File:** `neuroflow/segmentation/label_tool.py` · **Run:** `python -m neuroflow.segmentation.label_tool CLIP --frames FRAMES.parquet --labeler NAME` (no clip: self-test) · **Test:** `python -m pytest tests/test_label_tool.py`

**How:** OpenCV window. a/d step 1 frame, j/l 10 frames, space play, m marks the next point (start → peak → end), u undo, s save, q save and quit. Click the timeline strip at the bottom to jump. The strip shows θ_HT if the frames file has it. Re-opening a clip resumes from its labels CSV.

**Labelling rule:** start/end at rest + 5 % of the rep's range, not first visible movement (same as B1/B2). Leave out a rep you can't judge.

**Times:** with the clip's `frames.parquet`, video frame i gets `t_capture_ms` of `frame_idx` i, the segmenter's clock. Without it, i × 1000 / fps, flagged `time_source = video_fps` and not comparable with segmenter output.

**Output:** CSV with `rep_idx, t_start_ms, t_peak_ms, t_end_ms, subject_id, session_id, block_id, labeler, time_source`. `load_labels()` drops unfinished reps, checks start < peak < end and no overlap, and loads straight into `segment.validate(frames, labels, reps=...)` for B8.

**Not yet tried:** the interactive window. On this WSL box OpenCV's Qt window needs `sudo apt install -y libsm6 libice6`. Logic, timing, drawing and the B8 hand-off are tested.

## Angle input: theta_ht_deg or theta_v_deg (issue #4)

**Why:** A's seated V4 clips have no usable hip landmarks, so `theta_ht_deg` (arm vs trunk) is NaN on every frame and only `theta_v_deg` (arm vs image vertical) has values. The unchanged segmenter found 0 reps on them.

**What changed:** `angle_col` parameter (default `theta_ht_deg`, allowed `theta_v_deg`) on `segment_fixed`, `calibrate`, `segment_reps`, `rep_features`, `rep_matrix`, and `--angle` on the label tool.
- The reps table gets an `angle_col` column (new reps.parquet column, C needs to add it to the contract).
- `rep_features` / `rep_matrix` read the angle from the reps; passing a different one raises, so two angle definitions can't be mixed.
- A missing or all-NaN angle column raises a clear error instead of returning 0 reps.
- Calibration must be done on the same angle.

**Known limit:** trunk lean adds to `theta_v_deg`. On the leaning mock, range reads ~12° higher than with `theta_ht_deg` (tested). Without hips there's nothing to correct it with.

**Threshold check** (`docs/v4_threshold_check.py`, local): A's six V4 **practice** clips (not the held-out T clips), one rep each, 24–29 fps. Calibrating on the three complete ones gives rest 11.6°, range 79.8° (mock: 10°, 79.6°). B2 (30°/20°) and B3 (20 %/15 %) each find exactly 1 rep in all six, including the three short raises (peaks 51–75°). No threshold change needed. Our start is 250–430 ms before A's detector start because A's rule is 15° above rest held for 180 ms, ours is 5 % of range; the two aren't comparable.

**Note:** real fps is 24–29, not 30, so ±67 ms is about 1.7 frames, not 2.

## Decisions log

| Date | Decision |
| --- | --- |
| 2026-11-18 | Golden criterion: start and end within ±2 frames (±67 ms at 30 fps) |
| 2026-11-18 | Reuse legacy code only where sound, fix only when used; don't post the review of A's code |
| 2026-11-18 | B2 starts with T_high = 30°, T_low = 20° (to be tuned on training subjects only) |
| 2026-11-18 | B1 stays uncommitted until B2 validates against all four truth files |
| 2026-11-18 | Rep start/end = crossing of rest + 5 % of the rep's range; ±67 ms tolerance kept; same definition for B1 truth, B2 and B9 |
| 2026-11-18 | B2 validated: all four mocks pass (10/10 reps, max error 59.8 ms) |
| 2026-11-18 | B1 + B2 committed and pushed (`e716e9e`), tracking files committed (`1fba1f2`) |
| 2026-11-18 | B3 branched from `b1-mock-frames`; thresholds 20 % / 15 % of rom_ref, min prominence 30 %, min duration 0.4 s (to tune); `low_rom` mock added for the v0-fails case |
| 2026-11-18 | One branch for all B work (`Melchior-branch`); PR into main deferred until B6 |
| 2026-11-18 | B4 DTW: squared cost + sqrt (dtaidistance convention), band = round(band_frac × longer length), default 15 % |
| 2026-11-18 | B4 committed and pushed (`2023cad`, tracking `2da9475`) |
| 2026-11-18 | B5: templates on (θ − rest_ref) / rom_ref, 60 samples; both mean and DBA kept, choice left to P2 |
| 2026-11-18 | B5 committed and pushed (`14fb690`, tracking `e16dacc`) |
| 2026-11-18 | B6: range = excursion from rest; personal range = rom / rom_ref − 1; SPARC on whole rep; duration kept absolute |
| 2026-11-19 | PR #1 (B1–B6) approved by A and merged |
| 2026-11-20 | PR #2 (C1–C2) approved and merged by B without comments; known issues kept locally |
| 2026-11-20 | B7: one windowed max/min helper; lean taken as absolute until A confirms its sign |
| 2026-11-21 | B9: labels on the capture clock from frames.parquet; video fps fallback flagged |
| 2026-11-21 | Issue #4: agreed with A on a configurable angle (`theta_v_deg` for V4 clips) and a separate rest-corrected shoulder-tilt feature; data shared outside git |
| 2026-11-21 | `angle_col` added and recorded in reps; thresholds unchanged after checking on A's practice clips |
