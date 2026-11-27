# A6 seated bilateral physical validation

These are single-participant prototype observations from seated, fixed frontal 1280 x 720 webcam capture. Metadata records a shoulder/chest-height camera, marked position, and visible upper arms. The measured angle was shoulder-to-elbow versus downward image vertical. These tests do not independently validate anatomical joint-angle accuracy or clinical use. D_01-D_04 used bilateral pose schema 4, D_05-D_06 schema 5, and D_07-D_11 schema 6. All were diagnostic clips, separate from calibration and held-out evaluation.

## Verified local recordings

All eleven original AVIs, metadata JSONs, measured timestamp CSVs, and live-pose sidecars were present. For every clip, the independently decoded AVI count matched metadata, sidecar, and CSV rows. CSV and sidecar frame indices were zero-based and contiguous; timestamps matched exactly and increased strictly. Invalid frames remained in the sidecars. Version-aware sidecar replay reproduced every saved detector outcome.

| Clip | Intended scenario | Frames / schema | Observed outcome and evidence |
| --- | --- | --- | --- |
| D_01 | Initial startup diagnostic | 2 / 4 | `unable_to_judge`, `frame_gap`; 270 ms first gap before movement. |
| D_02 | Normal bilateral repetition | 240 / 4 | `complete`, both confirmed returns. |
| D_03 | Initial startup diagnostic | 2 / 4 | `unable_to_judge`, `frame_gap`; 254 ms first gap before movement. |
| D_04 | Left starts first | 308 / 4 | `complete`; left onset 1,595 ms, right 3,109 ms. |
| D_05 | Baseline initialization problem | 224 / 5 | `unable_to_judge`, `rest_baseline_timeout`; left baseline absent, right baseline 5.77 degrees. Clip ends mid-return at 8,013 ms. |
| D_06 | Normal bilateral repetition | 310 / 5 | `complete`; historical left peak 139.58 degrees at 5 ms preceded its 1,434 ms movement onset. |
| D_07 | Post-fix normal bilateral repetition | 370 / 6 | `complete`; left/right movement peaks 92.16 / 92.77 degrees, both within their onset-to-return intervals. |
| D_08 | Right starts first | 452 / 6 | `complete`; right onset 2,371 ms, left 5,043 ms: right first by 2,672 ms. |
| D_09 | Staggered return | 447 / 6 | `complete`; left return 10,997 ms, right 15,555 ms: left first by 4,558 ms. |
| D_10 | Only one arm returns | 768 / 6 | `incomplete_return`, `movement_timeout`; left return at 10,350 ms, right return absent. |
| D_11 | Independent left-arm tracking loss | 318 / 6 | `unable_to_judge`, `left_tracking_lost`; final four frames have left invalid and right valid. |

D_05's saved measurements contain a genuine stable left-rest subsequence from 91-607 ms with at most 5 degrees spread. A read-only schema-6 detector dry run finds both baselines ready by 639 ms; the original clip is not relabeled or extended. D_06's first left elbow presence was about 0.391, and its next angle was about 16.2 degrees. A read-only schema-6 movement-only calculation gives a left peak about 90.09 degrees at 6,783 ms. Its historical schema-5 metadata remains unchanged. Presence is retained for review rather than used as a hard gate: D_05 has steady right-rest frames with presence below 0.5. A geometric or temporal-outlier threshold requires independent validation.

The startup change warms MediaPipe before countdown and starts saving with a fresh frame after AVI setup. One pre-movement gap can restart rest collection; the active-movement 250 ms gap rule remains strict. Schema 6 finds a contiguous 500 ms stable rest window and displays **WAIT: BOTH ARMS DOWN** until **BOTH READY - RAISE**. Its peak statistics use movement samples only. D_07-D_11 physically exercise the post-fix baseline, peak, and detector behavior. No post-fix physical clip triggered the startup-gap recovery fallback itself.

Physically observed capabilities include simultaneous left/right measurements, anatomical arm identity, complete bilateral capture, left-first and right-first starts, asymmetric return, incomplete return, and independent left tracking loss. These tests do not establish generality across participants or camera placements, the full S1 camera-robustness or S2 tracking-reliability studies, bilateral DTW calibration/scoring, Member B/C integration, or a final bilateral Parquet contract.

## Further manual checks

Use menu option 16 for `both`, option 2 for framing, and option 17 for new diagnostic clips. Declare the intended movement before viewing the system result. Hold both arms down through the five-second countdown until **BOTH READY - RAISE** appears. Run option 17 twice after a fresh menu launch to test first-attempt startup. Inspect `startup_gap_restarts` and `startup_gap_reset`; the fallback has not yet been triggered physically after the fix. Do not overwrite D_01-D_11.

For each new clip, decode the AVI and compare its count with metadata, timestamp, and sidecar rows. Check zero-based contiguous indices, strictly increasing identical CSV/sidecar timestamps, independent left/right validity, and that each reported movement peak time falls between that arm's onset and return when a return exists. Use `check_clip.py` to review saved angles without rerunning MediaPipe. Keep diagnostic clips out of calibration and held-out sets. Avoid uncomfortable sustained positions merely to force a timeout; simulated tests cover that path.
