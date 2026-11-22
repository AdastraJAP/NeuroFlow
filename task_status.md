# Member B: Task Status

Last updated: 2026-11-22 · Branch: `Melchior-branch` · B1–B6 merged (PR #1); B7, B9 + fixes for both Adastra reviews pushed, waiting on re-review (PR #3)

Legend: ✅ done · 🟡 in progress / blocked on a decision · ⬜ not started

## Setup

| Item | Status | Note |
| --- | --- | --- |
| Read project docs (plan, research brief, research basis) | ✅ | |
| Branch `Melchior-branch` (one branch for all B work) | ✅ | Push over SSH (see tasks.md) |
| Python environment `.venv` | ✅ | `requirements.txt` added (incl. scikit-learn for 7willA's scorer). MediaPipe needs `libgles2` on WSL; B9's window needs `libsm6 libice6`. If `gh` / `git fetch` time out on WSL: `sudo ip link set dev eth0 mtu 1350` (resets on WSL restart) |
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
| B8 | S3 study: automatic vs manual segmentation | L | 🟡 | Harness done (`s3_study.py`): IoU matching, count MAE, exact/off-by-one rate, P/R/F1, boundary error; 7 tests (124 total). On mocks: v1 F1 1.0 everywhere, v0 misses low_rom. Now checks label clock/identity and lists missing label files (`ce764a2`). **Real G1 run** (rule-aided labels): count right on all 3 clips (F1 1.0 for v0 and v1), but boundaries loose: median abs error about 100–120 ms, v1 within ±67 ms on 2/6. Ends are the problem (T_01 +230, T_02 −615 truncated clip, T_05 −170 ms) |
| B9 | Manual labelling helper | S | ✅ | `label_tool.py`: labels on the capture clock, `load_labels()` feeds `segment.validate` for B8; 8 tests. Committed `13df5fe`. Real G1 clips labelled 2026-11-22 (`data/g1/T_0x_labels.csv`, gitignored): **rule-aided** (points placed at the 5 % crossing, then checked), T_02 end = last frame (rep cut off by the clip), peaks loose (flat tops). Strict frame-to-time mapping + finite marks (`6b915ed`); save and quit also with Esc / Enter / closing the window |

## Follow-ups from A and C

| Item | Status | Note |
| --- | --- | --- |
| Configurable angle input (`theta_ht_deg` / `theta_v_deg`), issue #4 | ✅ | `angle_col` on segmenter, calibration, features, templates, label tool; recorded in reps; thresholds hold on A's 6 V4 practice clips (1 rep found in each, rest 11.6°, rom 79.8°). 7 tests (112 total) |
| Shoulder-line tilt feature (`max_shoulder_tilt_deg`, rest-corrected) | 🟡 | Waiting on 7willA's OK in issue #6 (issue #4 closed) |
| Adastra's PR #3 review (3 blocking) | ✅ | Zero-rep angle kept (`126dade`), B9 strict mapping (`6b915ed`), B8 label checks (`ce764a2`); replied 2026-11-22 |
| Adastra's PR #3 re-review (2026-11-22) | ✅ | Label times finite, ordered, inside the clip (`569fe59`); calibration manifest `calib_manifest.py`, B8 takes `--calib` (`c548f59`, 12 tests, 156 total); tracking gaps moved to issue #12. Labels + manifest + README sent privately to Adastra (`data/g1/G2_labels_MelchiorRM.zip`). Replied; asked whether practice clips are P001 and which session. Waiting on re-review / approval |
| 7willA's PR #8 (C3–C4) | ✅ reviewed | Comment review 2026-11-22: calibration lookup fails on real ids, `dtw_dist` per fold, `fold_id = -1` turns the leak check off for locked test, scikit-learn. Answered calibration question: rest per session, range/duration from session 1 |
| Calibration block per session (Adastra) | 🟡 | Issue #9: Adastra agrees (~10 reps per session), wants protocol confirmed by B and 7willA first |
| Calibration marking in `reps.parquet` (with 7willA) | 🟡 | Issue #10. Adastra proposes `block_role = calibration / practice / test` (issue #9) |
| Long tracking gaps interpolated over by `smooth()` | ⬜ | Issue #12 (not affecting G1) |
| Reps columns `schema_version`, `block_id`, `side` (from PR #2) | ✅ | Same names and order as 7willA's contract (tested against `make_mock_reps.REP_COLUMNS`); `angle_col` is the only extra |
| Per-fold DTW template (with C, before G2) | 🟡 | `dtw_feature.py` (`6605f7f`): `rep_curves` (B) + `add_dtw_dist` (C, template from fold's normal train reps only); 5 tests (140 total). 7willA notified in issue #10. To do: save real `rep_curves.parquet` |

## Gates involving B

| Gate | Status |
| --- | --- |
| G1 Signal (Adastra's real frames) | 🟡 Adastra's draft PR #5; 3 real clips, `theta_v_deg` only; data shared on Google Drive |
| G2 Reps (B's real reps, reviewed by Adastra) | 🟡 Dry run on Adastra's 3 G1 clips: 1 rep found in each (correct count). B9 labels done; B8 shows boundary errors above ±67 ms (see B8). Next: look into end boundaries, then Adastra's review |
