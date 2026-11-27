# Member B: Task Status

Last updated: 2026-11-27 · Branch: `Melchior-branch` · B1–B6 merged (PR #1); B7, B8, B9 merged (PR #3, approved by Adastra 2026-11-24); tilt, `rep_curves` angle and #12 gap fix in PR #15, waiting on review; reps schema v2 + G1 calibration reps on `b-schema-v2-draft` (after #15)

Legend: ✅ done · 🟡 in progress / blocked on a decision · ⬜ not started

## Setup

| Item | Status | Note |
| --- | --- | --- |
| Read project docs (plan, research brief, research basis) | ✅ | |
| Branch `Melchior-branch` (one branch for all B work) | ✅ | Push over SSH (see tasks.md) |
| Python environment `.venv` | ✅ | `requirements.txt` added (incl. scikit-learn for 7willA's scorer). MediaPipe needs `libgles2` on WSL; B9's window needs `libsm6 libice6`. If `gh` / `git fetch` time out on WSL: `sudo ip link set dev eth0 mtu 1350` (resets on WSL restart). Fetching `main` / `Wilkend` over SSH can stall on the videos: for a PR, use a shallow clone without large files (`git clone --depth 1 --filter=blob:limit=1m --no-checkout -b <branch>` + sparse checkout without videos) |
| Review of Member A's prototype code | ✅ | Kept local, not posted |

## B tasks

| ID | Task | Size | Status | Note |
| --- | --- | --- | --- | --- |
| B1 | Mock `frames.parquet` generator (normal, slow, capped, leaning) | S | ✅ | Truth files use the 5 % boundary definition. B2 validated against all four. Committed `e716e9e` |
| B2 | Segmenter v0: fixed-threshold hysteresis state machine | M | ✅ | 10/10 reps and start/end within ±67 ms on all four mocks; `pytest tests/` passes. Known limit: across 50 random mocks, slow passes 49/50 and capped 46/50 (known limit). Committed `e716e9e` |
| B3 | Segmenter v1: ROM-normalised thresholds (S3 core) | M | ✅ | Finds 10/10 reps on new `low_rom` mock where v0 finds 0; all 15 tests pass. Known limit: low_rom boundary timing misses ±67 ms. Committed |
| B4 | DTW with Sakoe-Chiba band | M | ✅ | Matches `dtaidistance` exactly (200 random pairs × 4 bands); band behaves as expected; 7 tests. Committed `2023cad` |
| B5 | Templates: linear-stretch mean, then DBA | M | ✅ | Mean template recovers the true shape (≈0.4°); DBA beats mean on distance to members everywhere; 13 tests (35 total). Known: DBA fits noise when timing doesn't vary. Committed `14fb690` |
| B6 | Features: duration, time-to-peak ratio, personal-centred ROM, SPARC | M | ✅ | Each feature validated on its own, then integrated in `rep_features()`; 24 tests (59 total). Known: SPARC reads small movements as less smooth when noise is present. Committed `7260c6e` |
| B7 | Compensation features: max lean, min elbow, max hike | M | ✅ | `rep_window_extreme` + 3 columns in `rep_features()`; new `elbow_30` / `hike` mocks; each mock moves only its own feature; 10 tests. Open: lean sign (absolute for now, ask A). Committed `0589f8e` |
| B8 | S3 study: automatic vs manual segmentation | L | ✅ | Harness `s3_study.py` (IoU matching, count MAE, P/R/F1, boundary error); label clock/identity/in-clip checks; calibration only via `--calib` manifest; unknown boundaries matched at the clip edge, errors split start/end (`1c2e6ce`). **G1 result (2026-11-24, final labels):** count right on all 3 clips; 4 boundaries known on both sides, all within ±67 ms (starts +54, +11, −11; T_01 end −66 ms). T_02/T_05 ends outside the clip, so ends tested on 1 rep only. Package sent privately to Adastra; Adastra reproduced it and approved (PR #3, merged 2026-11-25). Preliminary: rule-aided, single participant |
| B9 | Manual labelling helper | S | ✅ | `label_tool.py`: labels on the capture clock, `load_labels()` feeds `segment.validate` for B8; 8 tests. Committed `13df5fe`. Real G1 clips labelled 2026-11-22 (`data/g1/T_0x_labels.csv`, gitignored): **rule-aided** (points placed at the 5 % crossing, then checked), T_02 end = last frame (rep cut off by the clip), peaks loose (flat tops). Strict frame-to-time mapping + finite marks (`6b915ed`); save and quit also with Esc / Enter / closing the window |

## Follow-ups from A and C

| Item | Status | Note |
| --- | --- | --- |
| Configurable angle input (`theta_ht_deg` / `theta_v_deg`), issue #4 | ✅ | `angle_col` on segmenter, calibration, features, templates, label tool; recorded in reps; thresholds hold on A's 6 V4 practice clips (1 rep found in each, rest 11.6°, rom 79.8°). 7 tests (112 total) |
| Shoulder-line tilt feature (`max_shoulder_tilt_deg`, rest-corrected) | ✅ | `f06ee1e`, PR #15: median of the 0.5 s before the start as resting tilt; wording posted in #14. Goes into reps with `schema_version` 2 |
| Adastra's PR #3 review (3 blocking) | ✅ | Zero-rep angle kept (`126dade`), B9 strict mapping (`6b915ed`), B8 label checks (`ce764a2`); replied 2026-11-22 |
| Adastra's PR #3 re-review (2026-11-22) | ✅ | Label times finite, ordered, inside the clip (`569fe59`); calibration manifest `calib_manifest.py`, B8 takes `--calib` (`c548f59`, 12 tests, 156 total); tracking gaps moved to issue #12. Labels + manifest + README sent privately to Adastra (`data/g1/G2_labels_MelchiorRM.zip`). Replied; asked whether practice clips are P001 and which session |
| Adastra's G2 package review (2026-11-23) | ✅ | End rule: rest only where the arm is still, unknown end = NaN (`1c2e6ce`); label tool `x` = outside the clip + angle shown (`80ae79e`); manifest session null + notes, CRLF→LF hash note (`3ce4ac9`). T_01 end relabelled (9397), T_02/T_05 ends outside the clip. Recording 1–2 s after return added to #9 |
| 7willA's PR #16 (scorer for schema v2) | ✅ reviewed | 2026-11-27: covers all four #13 points (`block_role`, unavailable features dropped, per-fold DTW via `add_dtw_dist`, abstain reasons); tilt wording matches B7. Bug found: a null `block_role` (B's v2 reps from v1 frames) made reps neither calibration nor test. Fix in **PR #17** into `Wilkend` (roles per row, `is_calib` per row, unknown role raises; `is_abstain_reason`, `no_dtw_distance` listed; 259 tests). Proposed in #16: bump scores `SCHEMA_VERSION` to 2, log features per fold, G1 check = plumbing only. Waiting on 7willA |
| 7willA's PR #13 (C5–C8) | ✅ reviewed, merged 2026-11-26 | Comment review 2026-11-25: fine for the mock scope (tests pass, fold-leak check fixed); branch behind main. Real-data blockers posted to 7willA in #14: hardcoded `normal_calib`/`s1` and `normal_test`/`s2`, all-NaN features crash the fit, no per-fold DTW template, censored reps abstain unreported |
| 7willA's PR #8 (C3–C4) | ✅ reviewed | Comment review 2026-11-22: calibration lookup fails on real ids, `dtw_dist` per fold, `fold_id = -1` turns the leak check off for locked test, scikit-learn. Answered calibration question: rest per session, range/duration from session 1 |
| Calibration block per session (Adastra) | 🟡 | Issue #9: stable-rest rule agreed (0.5 s below closing threshold, slope ≤ 3°/s, 1 s post-roll, 3 s cap from return confirmation), `block_role` = `calib` / `practice` / `test`. 2026-11-25: posted slope fit, valid-frame rules, cap origin and 3 timestamp names; waiting on Adastra (recorder side) |
| Calibration marking in `reps.parquet` (with 7willA) | 🟡 | Agreed in #14: `block_role` = `calib` / `test` / `practice` in frames + reps (spelling to confirm with Adastra, #9); G1 practice clips have no session ID (null). 2026-11-25: posted B's v2 proposal (keep `angle_col`; reps = v1 + `block_role`, `max_shoulder_tilt_deg`; NaN never 0). 7willA agreed 2026-11-26 (each member does their own table under `schema_version` 2). Draft on local branch `b-schema-v2-draft` (unpushed): reps `block_role` after `block_id`, `schema_version` 2, null `block_role` in v2 frames rejected, `FEATURE_COLUMNS`. **G1 calibration reps** (`8de0aea`): `calib_reps()` in `calib_manifest.py` gives the manifest's practice clips (complete_07–09) `block_role = "calib"`, sha256-checked, manifest's own rest/rom; null session → `NA` in `rep_id`; 190 tests. Private G1 v2 package (`data/g1/v2/reps.parquet` + `rep_curves.parquet`, 3 calib + 3 test reps, 4 curves) sent to 7willA on Google Drive 2026-11-27; caveats: complete_08 and T_05 ends outside the clip, T_02 end 29 ms before the clip edge (lower bound), no tilt on T clips (G1 frames lack `shoulder_tilt_deg`). Rebase on main and open a PR after #15 merges. 7willA's `make_mock_reps.py` still v1 |
| Adastra's draft PRs #7 (A5) and #11 (A6) | 🟡 reviewed | Comments posted 2026-11-27 (B's interface only). #7: works with B (`theta_v_deg`, tilt sign doesn't matter, `side` per rep); B will record the arm in the calibration manifest and refuse the other arm's calibration; asked about exporting diagnostic clips and old metadata without `arm`. #11: one row per frame agreed; B segments `theta_v_left_deg` / `theta_v_right_deg` separately, `rep_id` gets the side for `exercise_mode = both`, pairing by IoU ≥ 0.5 with `pair_id` (later), #9 stable-rest rule per arm (staggered returns), per-arm calibration from a bilateral `calib` block, count +1 ms timestamp shifts. Waiting on Adastra |
| Long tracking gaps interpolated over by `smooth()` | ✅ | `04450e6`, PR #15 (closes #12): gaps > 0.2 s stay NaN, reps spanning one are dropped, no boundary inside missing frames |
| Reps columns `schema_version`, `block_id`, `side` (from PR #2) | ✅ | Same names and order as 7willA's contract (tested against `make_mock_reps.REP_COLUMNS`); `angle_col` is the only extra |
| Per-fold DTW template (with C, before G2) | 🟡 | `dtw_feature.py` (`6605f7f`): `rep_curves` (B) + `fit_template` / `dtw_dist` / `add_dtw_dist` (C); 7willA pointed to it in #14. `angle_col` column + `save_curves` / `load_curves`, mixed angles refused (`38a98b4`, PR #15). Aliases `build_template` / `dtw_to_template` on `b-schema-v2-draft` (unpushed). 7willA's harness calls `add_dtw_dist` inside each fold (PR #16) |

## Gates involving B

| Gate | Status |
| --- | --- |
| G1 Signal (Adastra's real frames) | 🟡 Adastra's draft PR #5; 3 real clips, `theta_v_deg` only; data shared on Google Drive |
| G2 Reps (B's real reps, reviewed by Adastra) | ✅ Adastra reproduced the comparison and approved PR #3 (2026-11-24, merged 2026-11-25): 1 rep per clip, all 4 known boundaries within ±67 ms. Preliminary (rule-aided, 1 participant); end timing tested on 1 rep only |
| G3 Scorer (7willA's harness on real tables, reviewed by B) | 🟡 PR #16 + fix PR #17. Proposed in #16: G1 (1 participant, 3 calibration reps) can't run LOSO, so its check is plumbing only (roles found, fit on P001 alone fails cleanly, every T rep scored or abstains with a reason, no numbers reported); G3 itself needs ≥ 2 real subjects (realistically G4) |
