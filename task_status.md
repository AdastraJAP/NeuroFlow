# Member B: Task Status

Last updated: 2026-11-18 · Branch: `Melchior-branch` · B1–B5 committed; B6 not yet committed

Legend: ✅ done · 🟡 in progress / blocked on a decision · ⬜ not started

## Setup

| Item | Status | Note |
| --- | --- | --- |
| Read project docs (plan, research brief, research basis) | ✅ | |
| Branch `b1-mock-frames` created and pushed | ✅ | Push over SSH (see tasks.md) |
| Python environment `.venv` | ✅ | MediaPipe needs `libgles2` on WSL (not needed for B1–B8) |
| Review of Member A's prototype code | ✅ | Kept local, not posted |

## B tasks

| ID | Task | Size | Status | Note |
| --- | --- | --- | --- | --- |
| B1 | Mock `frames.parquet` generator (normal, slow, capped, leaning) | S | ✅ | Truth files use the 5 % boundary definition. B2 validated against all four. Committed `e716e9e` |
| B2 | Segmenter v0: fixed-threshold hysteresis state machine | M | ✅ | 10/10 reps and start/end within ±67 ms on all four mocks; `pytest tests/` passes. Known limit: across 50 random mocks, slow passes 49/50 and capped 46/50 (known limit). Committed `e716e9e` |
| B3 | Segmenter v1: ROM-normalised thresholds (S3 core) | M | ✅ | Finds 10/10 reps on new `low_rom` mock where v0 finds 0; all 15 tests pass. Known limit: low_rom boundary timing misses ±67 ms. Not yet committed |
| B4 | DTW with Sakoe-Chiba band | M | ✅ | Matches `dtaidistance` exactly (200 random pairs × 4 bands); band behaves as expected; 7 tests. Committed `2023cad` |
| B5 | Templates: linear-stretch mean, then DBA | M | ✅ | Mean template recovers the true shape (≈0.4°); DBA beats mean on distance to members everywhere; 13 tests (35 total). Known: DBA fits noise when timing doesn't vary. Committed `14fb690` |
| B6 | Features: duration, time-to-peak ratio, personal-centred ROM, SPARC | M | ✅ | Each feature validated on its own, then integrated in `rep_features()`; 24 tests (59 total). Known: SPARC reads small movements as less smooth when noise is present. Not yet committed |
| B7 | Compensation features: max lean, min elbow, max hike | M | ⬜ | |
| B8 | S3 study: automatic vs manual segmentation | L | ⬜ | |
| B9 | Manual labelling helper | S | ⬜ | Must label start/end with the 5 % definition (see tasks.md) |

## Gates involving B

| Gate | Status |
| --- | --- |
| G1 Signal (A's real frames) | ⬜ waiting on A |
| G2 Reps (B's real reps, reviewed by A) | ⬜ |
