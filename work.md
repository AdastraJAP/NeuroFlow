# Member C: Wilkend's Work Plan

Branch: `Wilkend` (rebased on `main`, `3a117d2`) · Owner: Wilkend Auguste · Last updated: 2026-11-23

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

**Pushing:** `gh` is logged in as `7willA`, so plain HTTPS works: `git push origin Wilkend`. (Melchior pushes over SSH.)

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

**`scores.parquet` columns (mine):** `schema_version, rep_id, scorer_id, ref_type, mapping, fold_id, score_0_100, percentile, subscores (json: d, rom, tempo, shape, smooth, trunk, elbow, hike; `None` when abstained), abstain, abstain_reason, feedback_code`

**`folds.csv` columns (mine):** `subject_id, fold_id, split` (`train` / `val` / `locked_test`; locked test has `fold_id = -1`)

**General rules:** times in **milliseconds** from the capture clock (never frame counts); missing values are **NaN, never 0**; fix and print every random seed.

---

## 4. My tasks (C1–C9) and status

Legend: ✅ done (merged into `main`) · 🟡 in progress or in review · ⬜ not started. Size: S = 1, M = 2, L = 3 points (17 total).

| ID | Task | Size | Status | Done when |
| --- | --- | --- | --- | --- |
| C1 | Mock `reps.parquet` generator + subject-wise `folds.csv` builder (locked test subset) | S | ✅ merged (PR #2) | Mock with 18 subjects; no subject in two folds |
| C2 | RehabHub baseline: Z of DTW distance vs normal reps, score = 100 / max(Z, 1) | S | ✅ merged (PR #2) | Reproduces the saturation: many reps with Z < 1 all get 100 |
| C3 | Scorer v1: standardise features → Ledoit-Wolf covariance → Mahalanobis distance → pooled ECDF → 0–100 | M | ✅ merged (PR #8) | A shifted mock group scores lower than normal; ECDF of training normals ≈ uniform |
| C4 | P3 sub-scores: per-feature ECDF for lean, elbow, hike (and rom, tempo, shape, smooth); feedback code from the worst sub-score | M | ✅ merged (PR #8) | A lean-heavy mock rep returns feedback code "trunk" |
| C5 | LOSO harness: fit on train subjects, score the held-out subject, AUROC per condition | M | 🟡 in review (PR #13) | AUROC ≈ 0.5 on shuffled labels; high on a clearly shifted mock |
| C6 | Statistics: subject-level bootstrap CI, paired score difference, ICC(2,1), SEM, MDC | L | 🟡 written, tests pass, not pushed | CI on a known synthetic effect covers the truth ≈ 95 % of the time |
| C7 | **P2 ablation grid runner** (reference × band × tempo term × mapping) | L | 🟡 v0 (tempo × mapping) written, not pushed | One command reproduces the table; each cell has a CI |
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

**C3 result (mock data, LOSO, session-2 reps):** 0.4 % of normal reps score 100 (RehabHub: 83 %); slow 2x/3x median 0 (RehabHub: 100). Two things to remember:
- **New floor at 0.** Every rep beyond the most unusual normal training rep scores 0, so band light and medium both get 0. The raw Mahalanobis distance is kept in `subscores` as `d` so C7 can still grade severity.
- **Normal median is 40.5, not 50.** μ, Σ and the ECDF reference are fitted on the training subjects, so their own normal reps fit the model by construction; a held-out subject moves in their own way. That 50 → 40.5 drop is a small example of the P1 leakage gap.

**C4 result (mock data, LOSO, session-2 reps):** the expected feedback code is the most common one for every condition: trunk_lean → `trunk`, elbow_bend → `elbow`, rom_cap → `rom`, slow and band → `tempo` (feedback on 99–100 % of reps; rom_cap 90° on 51 %, a mild case). **False feedback on held-out normal reps: 6.8 %**, above the naive 5 % because held-out subjects score lower than training subjects. Hike shrug gets feedback on 9.3 %, mostly `shape`, because hike is never allowed to speak.

**C5 result (mock data, LOSO, s2, `normal_test` vs each condition, AUROC):**

| Condition | RehabHub (C2) | C3 |
| --- | --- | --- |
| slow 2x / 3x | 0.548 / 0.545 | 1.000 / 1.000 |
| trunk_lean small / large | 0.547 / 0.611 | 1.000 / 1.000 |
| elbow_bend 30° / 60° | 0.572 / 0.629 | 1.000 / 1.000 |
| band light / medium | 0.767 / 0.980 | 0.999 / 1.000 |
| hike shrug | 0.559 | 0.544 (hike is not a C3 feature) |
| *rom_cap 60° / 90° (positive control)* | *0.999 / 0.667* | *1.000 / 0.768* |

C3 coverage is 0.986–0.996 in every row. **The mock is too easy:** C3 sits at the 1.000 ceiling almost everywhere, so C7 variants cannot be told apart on it. Meaningful P2 differences will need real data (or a harder mock).

**Leak test (labels shuffled within subject, before fitting; pass = mean 0.5 ± 0.05, fixed in advance):** C3 LOSO 0.501 ✅, 1-NN LOSO 0.492 ✅, **1-NN with a deliberate leak 1.000 → caught ✅**, but **C3 with the same deliberate leak 0.506 → passes**. Lesson: the test is sensitive to a scorer that memorises reps, but blind to a leak into a smooth model like C3 (a few means and one covariance cannot memorise a subject). So for C3, passing the shuffle test is weak evidence; its real protection is structural (the scorer refuses to score a training subject, and tests prove impaired training reps do not change the fit).

**C6 result (mock, LOSO, s2, 14 subjects, 2,000 subject resamples, seed 0):** C3 − RehabHub paired AUROC differences exclude 0 for slow (+0.45), trunk_lean (+0.39 / +0.45), elbow_bend (+0.43 / +0.37) and band (+0.23 / +0.02); not for hike (−0.015 [−0.072, +0.036]). Test-retest of subject mean normal scores: RehabHub ICC(2,1) 0.67, MDC95 11.6 points; C3 ICC 0.75, MDC95 22.0 points. Synthetic check: the 95 % subject-bootstrap CI covered a known AUROC 94.7 % of the time (150 simulated studies, 14 subjects).

**C7 v0 result (mock; reference and alignment held fixed):** the tempo term is the factor that matters. Slow 2x AUROC goes from 0.52–0.56 (shape only, any mapping) to 1.000 with `duration_ratio`; adding time-to-peak adds nothing on the mock. The three mappings (z100, LW + ECDF, GMM) are indistinguishable here because the mock is at the ceiling. Reference and alignment wait for the per-fold DTW template; the AUROC-by-factor plot (v2) is still to do.

**Real data (G1, P001, 3 clips) – known gap:** `trunk_lean_deg` is NaN on 100 % of frames and `elbow_deg` is valid on only 0–5 %, so the C3 scorer as written would abstain on every real rep. The feature set must depend on what the pipeline provides (and be recorded in the model). To do before G3.

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
- `feedback_code` (its own column, not inside `subscores`) = the short name of the worst sub-score, for example `trunk`. Allowed values: `contract.FEEDBACK_CODES` (no `d`, no `hike`); `None` when there is no feedback.
- **Directions:** high = worse for trunk, shape (DTW), hike; low = worse for elbow, smooth (SPARC), **rom**; two-sided for tempo (|duration_ratio − train median|). One-sided-low uses ECDF(−x).
- **When to give feedback:** only if the C3 score is < 5 **and** the worst sub-score is low too; otherwise `None`, so a normal rep never gets a false message. Thresholds are set on training subjects only.
- **Ties at 0:** break them with the robust deviation (x − train median) / train MAD, on the bad side only. These values are used to pick the code and are not stored.
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
- Same scorer, both splits, per condition → the AUROC gap is the result. `score_baseline(..., allow_in_sample=True)` allows in-sample scoring for exactly this leaky case and marks the rows `ref_type="in_sample"`; `fold_id` no longer switches the check off (Melchior's review of PR #8, point 3).
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
7. **Calibration session:** session-1 calibration only, or each session's own `normal_calib`? Own-session is more realistic (camera and day change). It affects `rom_rel_personal` (my mock, `make_mock_reps.py`), `duration_ratio` (my scorer) and `rest_ref`/`rom_ref` on real data (B). It also matters for C6 test-retest. For now everything uses session 1.
8. **ROM is one-sided (C4):** only "not high enough" loses points. Going above the calibration range is progress, and overshoot through leaning is already caught by the trunk sub-score. Tempo stays two-sided (too fast = swinging the arm). Team to confirm.
9. **Feedback threshold (C4):** give a `feedback_code` only when `score_0_100` < 5. Team to confirm the value once we see how often held-out normal reps cross it.
10. Next steps after Foundations: a small pilot with 3–5 classmates; the squat only after the shoulder workflow is stable.

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

- [x] Commit C1 + C2 on `Wilkend` (rebased on `main` after PR #1), push, open PR for Melchior's review
- [x] Melchior reviewed and merged C1 + C2 (PR #2, 2026-11-20)
- [ ] Melchior reviews C3 (PR #8)
- [ ] Mentor mode from C3 on: idea + hint first, code after
- [ ] ~~C3 scorer v1~~ (done) → C4 sub-scores → C5 LOSO harness (C3–C5 make gate G3 possible)
- [ ] Reply in the group about the angle decision (section 8, point 1)
- [ ] Look at REHAB24-6 and give the professor an opinion
