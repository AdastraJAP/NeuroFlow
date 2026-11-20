# Member C: Wilkend's Work Plan

Branch: `Wilkend` (from `main`, `d3aa726`) · Owner: Wilkend Auguste · Last updated: 2026-11-18

Source documents (from Melchior, Nov 17): *Implementation Plan: Tasks, Workflow and Member Prompts*, *Research Brief: Evidence, Datasets and Architecture*, *Research Basis (3 + 3 contributions)*, and Prof. Huang's email of Nov 17 ("Re: Next meeting").

---

## 1. My role

I am **Member C: scoring and evaluation**. I own the project's claims.

| Contribution | What it asks | Mine? |
| --- | --- | --- |
| **P1** | Does RehabHub-style DTW scoring hold up without subject leakage? (its within-participant split vs leave-one-subject-out, per condition) | **Owner** |
| **P2** | Which design choices fix DTW's blindness to slow, full-range reps? (factorial ablation) | **Owner** |
| **P3** | Do compensation sub-scores (lean, elbow, hike) catch reps DTW misses? | **Sub-scores and combined score** (B computes the features) |
| Integration and demo | Live webcam loop with angle, rep count and score | **Owner** |
| S1, S2 (A) · S3 (B) | Robustness, abstention, segmentation | I use or review them |

**Team**

| Member | Area | Branch |
| --- | --- | --- |
| A: Piard (Alan) | Perception and geometry: capture, pose, angles, filtering, confidence | (his own) |
| B: Melchior | Segmentation and features: reps, DTW, templates, SPARC, lean | `Melchior-branch` |
| C: Wilkend | Scoring and evaluation: folds, scorers, statistics, P1/P2, demo | `Wilkend` |

**Review rotation:** C reviews A, A reviews B, **B (Melchior) reviews C**. A review means running the other member's `main()` and tests and reading the diff, not rewriting it.

**Team rule (Melchior, Nov 18):** one branch per person (not per task). After review, branches are merged into `main`.

---

## 2. Setup

**Python packages:** `numpy pandas pyarrow scipy scikit-learn pytest` (plus `matplotlib` later for plots).

```bash
# from the repo root (Desktop\Neuro_Flow)
python -m venv .venv
.venv\Scripts\activate            # Windows  (WSL/Linux: source .venv/bin/activate)
pip install numpy pandas pyarrow scipy scikit-learn pytest matplotlib
```

**Run everything of mine:**

```bash
python fixtures/make_mock_reps.py        # C1: writes fixtures/mock_reps.parquet
python -m neuroflow.scoring.folds        # C1: writes fixtures/mock_folds.csv
python -m neuroflow.scoring.baseline     # C2: RehabHub baseline under LOSO
python -m pytest tests/ -q               # all tests
```

**Pushing:** the `origin` remote is HTTPS with no stored login. Melchior pushes over SSH:
`git push git@github.com:AdastraJAP/NeuroFlow.git Wilkend`

---

## 3. Data contract (what I read and what I write)

Stages talk **only through tables** (`schema_version = 1`). Never rename a column quietly; contract changes go through a pull request that bumps `schema_version` and tells all three members.

| Table | Owner | One row per | I… |
| --- | --- | --- | --- |
| `frames.parquet` | A | video frame | rarely touch it (demo only) |
| `reps.parquet` | B | repetition | **read** (my input) |
| `scores.parquet` | **C** | rep × scorer | **write** |
| `folds.csv` | **C** | subject | **write** (everyone uses it) |

**`reps.parquet` columns (B's output):** `rep_id, subject_id, session_id, block_id, condition, level, side, seg_method, t_start_ms, t_peak_ms, t_end_ms, duration_s, rom_peak_deg, rom_rel_personal, time_to_peak_ratio, sparc, max_trunk_lean_deg, min_elbow_deg, max_hike_ratio, dtw_dist, dtw_band_frac, ref_id, valid_frame_frac, rep_conf, rater_labels`

**B's conventions I must respect (from `Melchior-branch`):**
- `rep_id` = `{subject}_{session}_{block}_{nnn}`
- `rom_peak_deg` = degrees moved **from rest** (peak − rest), not the absolute angle.
- `rom_rel_personal` = `rom_peak_deg / personal calibration median − 1` (0 = the user's usual range).
- `sparc` is negative; **closer to 0 = smoother**.
- `duration_s` is absolute. **Dividing it by the user's calibration median is MY job** (scorer).
- `dtw_dist` = sqrt of summed squared differences on θ_HT ÷ personal peak (no unit).
- Rep start/end = where the angle crosses rest + 5 % of the rep's range.

**`scores.parquet` columns (mine):** `schema_version, rep_id, scorer_id, ref_type, mapping, fold_id, score_0_100, percentile, subscores (json: rom, tempo, shape, smooth, trunk, elbow, hike), abstain, abstain_reason, feedback_code`

**`folds.csv` columns (mine):** `subject_id, fold_id, split` (`train` / `val` / `locked_test`; locked test has `fold_id = -1`)

**General rules:** times in **milliseconds** from the capture clock (never frame counts); missing values are **NaN, never 0**; fix and print every random seed.

---

## 4. My tasks (C1–C9) and status

Legend: ✅ done · 🟡 code written, not yet reviewed/committed · ⬜ not started. Size: S = 1, M = 2, L = 3 points (17 total).

| ID | Task | Size | Status | Done when |
| --- | --- | --- | --- | --- |
| C1 | Mock `reps.parquet` generator + subject-wise `folds.csv` builder (locked test subset) | S | 🟡 | Mock with 18 subjects; no subject in two folds |
| C2 | RehabHub baseline: Z of DTW distance vs normal reps, score = 100 / max(Z, 1) | S | 🟡 | Reproduces the saturation: many reps with Z < 1 all get 100 |
| C3 | Scorer v1: standardise features → Ledoit-Wolf covariance → Mahalanobis distance → pooled ECDF → 0–100 | M | ⬜ | A shifted mock group scores lower than normal; ECDF of training normals ≈ uniform |
| C4 | P3 sub-scores: per-feature ECDF for lean, elbow, hike (and rom, tempo, shape, smooth); feedback code from the worst sub-score | M | ⬜ | A lean-heavy mock rep returns feedback code "trunk lean" |
| C5 | LOSO harness: fit on train subjects, score the held-out subject, AUROC per condition | M | ⬜ | AUROC ≈ 0.5 on shuffled labels; high on a clearly shifted mock |
| C6 | Statistics: subject-level bootstrap CI, paired score difference, ICC(2,1), SEM, MDC | L | ⬜ | CI on a known synthetic effect covers the truth ≈ 95 % of the time |
| C7 | **P2 ablation grid runner** (reference × band × tempo term × mapping) | L | ⬜ | One command reproduces the table; each cell has a CI |
| C8 | **P1 reproduction**: RehabHub within-participant split vs LOSO, per condition | M | ⬜ | Table showing the gap between the two splits, per condition |
| C9 | Demo v0: live webcam loop with angle, rep count and last rep's score | S | ⬜ | Runs on the laptop webcam; prints a score after each rep |

### What exists now (C1 + C2)

| File | What it does |
| --- | --- |
| `fixtures/make_mock_reps.py` | 18 subjects × 2 sessions × 13 blocks × 10 reps = 4,680 mock reps. Subjects differ from each other (needed so leakage is visible). Impairments are graded. Slow reps are longer but DTW barely changes (speed-blind, on purpose). About 0.5 % of reps have an unknown start (NaN) and 3 % have low valid-frame fractions. |
| `neuroflow/scoring/contract.py` | Column lists for `scores.parquet` and `folds.csv` |
| `neuroflow/scoring/folds.py` | `make_folds()` (LOSO by default, 4 locked-test subjects, seed 0), `check_folds()`, `iter_folds()` (never puts a subject in both train and test, never uses locked subjects), save/load |
| `neuroflow/scoring/baseline.py` | `fit_baseline()` (μ, σ from normal training reps only), `score_baseline()`, refuses to score a training subject inside a fold |
| `tests/test_mock_reps.py`, `tests/test_folds.py`, `tests/test_baseline.py` | 28 tests, all pass |

**First result (mock data, LOSO):** 83 % of normal reps and 75 % of slow reps score exactly 100, so the RehabHub score cannot tell slow reps from normal ones. That is the flaw P2 must fix. (Theory: Z < 1 covers ≈ 84 % of a normal distribution.)

**Mock blocks:** `normal_calib, normal_test, slow 2x/3x, band light/medium, trunk_lean small/large, elbow_bend 30deg/60deg, hike shrug, rom_cap 60deg/90deg` (7 conditions, 13 blocks). Naming follows B's mock (`condition` + `level`).

---

## 5. How to do the next tasks

### C3: Scorer v1
- **Learn first:** Mahalanobis distance; `sklearn.covariance.LedoitWolf`; what an ECDF is.
- **Features (about 6–10, low-dimensional):** `dtw_dist`, duration ratio (`duration_s` ÷ the user's calibration median), `rom_rel_personal`, `time_to_peak_ratio`, `sparc`, `max_trunk_lean_deg`, `min_elbow_deg`, (`max_hike_ratio`, exploratory).
- **Steps:** (1) standardise with mean/SD of **normal training reps**; (2) fit Ledoit-Wolf on them; (3) Mahalanobis distance of every rep; (4) score = 100 × (1 − ECDF of the normal training distances), so a typical normal rep is around 50 and an unusual rep is near 0.
- **Personalisation = location only:** centre ROM and duration on the user's calibration median. The distribution shape always comes from pooled training subjects (15 personal reps give a ±0.35 ECDF band, which is too wide).
- **Code:** `neuroflow/scoring/scorer.py` with `fit_scorer(train_reps) -> dict` and `score_reps(model, reps) -> DataFrame`.

### C4: Sub-scores (P3)
- Same pooled-ECDF idea, one feature at a time: rom, tempo, shape (DTW), smooth (SPARC), trunk, elbow, hike.
- Pick the direction for each feature (a bigger lean is worse; a smaller elbow angle is worse; a more negative SPARC is worse).
- `feedback_code` = the name of the worst sub-score (for example `trunk_lean`), stored in `subscores` as JSON.
- Hike stays **exploratory** (F1 ≈ 0.55 in Pavlikov 2026; it's weak everywhere).

### C5: LOSO harness
- Use `iter_folds()`: fit on `train`, score `test`, collect all folds.
- **AUROC per condition vs normal, never pooled** (`sklearn.metrics.roc_auc_score`; label 1 = impaired, use −score so a lower score means more impaired).
- **Leak test:** shuffle condition labels → AUROC must be ≈ 0.5.
- `rom_cap` is a **positive control**: report it separately and never pool it into a headline number.

### C6: Statistics
- **Bootstrap resamples subjects, not reps** (1,000–2,000 resamples, fixed seed).
- v0: percentile bootstrap of a mean → v1: subject resample of AUROC → v2: paired difference between two scorers on the same resamples, plus ICC(2,1), SEM and MDC between session 1 and session 2.
- Call something "a difference" only if the paired interval excludes 0. With ~18 subjects, AUROC differences below ~0.1 cannot be resolved (SE ≈ 0.062 at AUROC 0.85).

### C7: P2 ablation grid

| Factor | Levels |
| --- | --- |
| Reference | pooled linear-stretch mean / DBA template / own-session mean (leaky control) |
| Alignment | unconstrained DTW / Sakoe-Chiba band 10 % / 20 % |
| Tempo term | none / duration ratio / duration + time-to-peak ratio |
| Score mapping | RehabHub Z → 100/max(Z,1) / LW Mahalanobis + pooled ECDF / GMM score |
| Fuser (ablation only) | none / logistic regression / gradient boosting |

- Run the full grid on the **slow** and **band** conditions, and a reduced grid elsewhere.
- Ladder: v0 = two cells by hand → v1 = loop with fixed seeds → v2 = CI per cell + plot of AUROC by factor level.
- Reference and alignment need **B's code** (`neuroflow/segmentation/dtw.py`, `templates.py` on `Melchior-branch`).

### C8: P1 reproduction
- RehabHub split: the normal reps of **each** subject are split at random between "template/baseline" and "test-normal" (every subject is in both sets, so it leaks).
- My split: LOSO.
- Same scorer, both splits, per condition → the AUROC gap is the result. `score_baseline(..., fold_id=-1)` allows in-sample scoring for exactly this leaky case.
- Exact RehabHub recipe (for the faithful version): MA-5 smoothing, single shoulder angle, linear-stretch mean template, unconstrained DTW, Z, 100/max(Z,1).

### C9: Demo v0
- Reuse A's angle and filter functions plus B's segmenter; overlay with `cv2.putText`; print a score after each rep.
- The old `step6_demo.py` shows the OpenCV loop (mirror the display, detect on the unmirrored image).

---

## 6. Rules I must never break

1. **Fit everything inside the training fold only:** template, covariance, ECDF, thresholds, μ/σ. A single global fit before splitting is the leak this project exists to measure.
2. **Folds are by subject.** Nothing from a test subject enters fitting.
3. **Per-condition AUROC, never pooled**, always with subject-bootstrap intervals.
4. **Locked test** (4 subjects): run **once**, at gate G5, by a member who did not tune on it.
5. Fix the analysis plan (conditions, metrics, grid) **before** looking at test scores.
6. Do not invent numbers from papers. Keep what we computed separate from what we cite.

**Self-review checklist before every pull request**
1. Does `main()` run from a clean checkout and print something checkable?
2. Do the tests pass (`python -m pytest tests/`)?
3. Are times in ms and missing values NaN?
4. Is anything fitted before the train/test split?
5. Is every number traceable to a seed and a file name?

---

## 7. Gates (no calendar; work runs on these)

| Gate | Passes when | What it means for me |
| --- | --- | --- |
| G1 Signal | A's `frames.parquet` from 3 real clips passes, **reviewed by me** | I review A's work |
| G2 Reps | B's `reps.parquet` from those clips has the right rep count, reviewed by A | I switch from mock to real reps |
| G3 Scorer | My LOSO harness runs end to end on real tables, **reviewed by B** | P1, P2, P3 start on real data |
| G4 Data | Protocol fixed, ≥ 6 subjects recorded in both sessions | Full runs |
| G5 Freeze | Analysis plan written, folds and locked test fixed | Locked test run once |

Until G2, I build everything on `fixtures/mock_reps.parquet`. Because the schema is fixed, real data should swap in without code changes.

---

## 8. Open points the team must decide (they affect my scorer)

1. **Which angle feeds DTW?** Prof. Huang (Nov 17) proposed **hip-free signals**: *arm elevation* (upper arm vs image vertical, θ_V) and *shoulder-line tilt* (vs start of the rep), because the hip has visibility 0.25–0.28 in our camera view. The Research Brief makes **θ_HT** (needs the hip) the primary DTW signal. My code only reads `dtw_dist`, so it works either way, but A and B must choose. The reference clip and thresholds must then be recorded again (6 clips).
2. **Angle tolerance:** the professor says 10° is fine for now (not 5°). The goniometer check is at 0, 45, 90 and 120° (a phone angle app is OK).
3. **REHAB24-6 dataset** (https://zenodo.org/records/13305826): RGB videos, arm abduction, mocap ground truth, rep boundaries and correct/incorrect labels. It could be a real external test for my LOSO harness. The professor wants our opinion.
4. **Block list:** 12 or 13 blocks (the report says both). My mock uses 13; it's one constant (`BLOCKS`) to change.
5. **3D side experiment** (professor): world-landmark angle next to the 2D angle, front vs ~45° view. That's A's area.
6. **Professor wants** one screenshot with the skeleton overlay (to see where the hip is).
7. Next steps after Foundations: a small pilot with 3–5 classmates; the squat only after the shoulder workflow is stable.

---

## 9. Key numbers to remember

| Quantity | Value | Meaning |
| --- | --- | --- |
| RehabHub score | 100 / max(Z, 1) | Every rep with Z < 1 gets 100 (≈ 84 % of normal reps) |
| RehabHub failure | 10 closest abnormal shoulder-flexion reps: mean Z −0.0123, all scored 100 | DTW ignores speed |
| ECDF 95 % band | ±0.35 (n = 15), ±0.25 (n = 30), ±0.09 (n = 240) | Personal reps set location only |
| AUROC 0.85, 20 vs 20 subjects | SE ≈ 0.062, 95 % CI ≈ ±0.12 | Differences < 0.1 not resolvable |
| ICC 0.80, 20 subjects × 2 sessions | 95 % CI width ≈ 0.32 | Report it, don't rank on it |
| Trunk lean detection | F1 0.92 (Pavlikov 2026) | Strong sub-score |
| Shoulder hike detection | F1 0.55 (Pavlikov), 0.07 with Kinect (Zhi 2017) | Exploratory only |

(The ECDF, AUROC-SE and ICC widths were computed by the report, not taken from papers. Check before citing.)

---

## 10. Reading list for me (in order)

1. Labchurat et al. 2026, **RehabHub**, Sensors 26(4):1219: the method I re-evaluate (split, template, Z, 100/max(Z,1)).
2. Pavlikov et al. 2026, Sensors 26(16):5054: the closest prior system (angles, state machine, flags).
3. Unger et al. 2025, Frontiers in Medicine: evidence of the subject-leakage gap.
4. Liao, Vakanski & Xian 2020, IEEE TNSRE: Mahalanobis vs GMM vs DTW scoring.
5. Ledoit & Wolf 2004: shrinkage covariance.
6. Bonett 2002: ICC sample size.

---

## 11. Next actions

- [ ] Review C1 + C2 together (code walk-through), then commit on `Wilkend`
- [ ] Mentor mode from C3 on: idea + hint first, code after
- [ ] C3 scorer v1 → C4 sub-scores → C5 LOSO harness (C3–C5 make gate G3 possible)
- [ ] Reply in the group about the angle decision (section 8, point 1)
- [ ] Look at REHAB24-6 and give the professor an opinion
