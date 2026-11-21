# Member B: Task Status

Last updated: 2026-11-21 · Branch: `Melchior-branch` · B1–B6 merged (PR #1); B7, B9 committed, in review (PR #3)

Legend: ✅ done · 🟡 in progress / blocked on a decision · ⬜ not started

## Setup

| Item | Status | Note |
| --- | --- | --- |
| Read project docs (plan, research brief, research basis) | ✅ | |
| Branch `Melchior-branch` (one branch for all B work) | ✅ | Push over SSH (see tasks.md) |
| Python environment `.venv` | ✅ | MediaPipe needs `libgles2` on WSL; B9's window needs `libsm6 libice6` |
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
| B8 | S3 study: automatic vs manual segmentation | L | 🟡 | Harness done (`s3_study.py`): IoU matching, count MAE, exact/off-by-one rate, P/R/F1, boundary error; 7 tests (124 total). On mocks: v1 F1 1.0 everywhere, v0 misses low_rom. Waiting on B9 labels of the real clips |
| B9 | Manual labelling helper | S | ✅ | `label_tool.py`: labels on the capture clock, `load_labels()` feeds `segment.validate` for B8; 8 tests. GUI not tried yet (WSL needs `libsm6 libice6`). Committed `13df5fe` |

## Follow-ups from A and C

| Item | Status | Note |
| --- | --- | --- |
| Configurable angle input (`theta_ht_deg` / `theta_v_deg`), issue #4 | ✅ | `angle_col` on segmenter, calibration, features, templates, label tool; recorded in reps; thresholds hold on A's 6 V4 practice clips (1 rep found in each, rest 11.6°, rom 79.8°). 7 tests (112 total) |
| Shoulder-line tilt feature (`max_shoulder_tilt_deg`, rest-corrected) | 🟡 | Waiting on 7willA's OK for the new column |
| Reps columns `schema_version`, `block_id`, `side` (from PR #2) | ✅ | Same names and order as 7willA's contract (tested against `make_mock_reps.REP_COLUMNS`); `angle_col` is the only extra |
| Per-fold DTW template (with C, before G2) | ⬜ | |

## Gates involving B

| Gate | Status |
| --- | --- |
| G1 Signal (Adastra's real frames) | 🟡 Adastra's draft PR #5; 3 real clips, `theta_v_deg` only; data shared on Google Drive |
| G2 Reps (B's real reps, reviewed by Adastra) | 🟡 Dry run on Adastra's 3 G1 clips: 1 rep found in each (correct count). Needs B9 labels to check boundaries, then Adastra's review |
