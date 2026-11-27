# Neuro-Flow — automatic repetition detection on Windows

Open **START_HERE.bat** in this folder. The menu shows the active position, anatomical arm, reference and segmented DTW calibration. Option **14** switches between seated and standing; option **16** switches between right and left arm. Each position and arm needs its own reference and practice calibration. Option **2** shows the live arm angle and shoulder-line tilt; pressing **A** there also saves the selected arm for the menu.

The launcher prints the active project path. The baseline folder's **START_HERE.bat** opens this updated project. Use **START_LEGACY_BASELINE.bat** inside the baseline folder to explicitly run the archived 10-second workflow.

## Record and calibrate

For a fresh seated **left** experiment, select left with option 16, then follow steps 1-4 below. A fresh checkout starts without a left reference or DTW threshold. The existing seated right-arm threshold of **2.224** is not applied to it. Use option **17** for diagnostic captures so they stay out of the reference, practice and held-out folders; see [A5_PHYSICAL_VALIDATION.md](A5_PHYSICAL_VALIDATION.md) for the physical check sequence.

1. Use **2** to check framing and arm angle. Use **4** if you need a reference, and **7** to review it. An existing same-position reference can be reused if its measured timestamps and full return are usable.
2. Use **5** three times for complete practice repetitions. Use **6** at least three times for shorter or deliberately different raises that still return to rest. These are negative examples with a segment that DTW can score. A deliberately failed return is useful for checking detection, and produces no DTW score.
3. Run **8** to score fresh practice. Use **9** to freeze the suggested value if three or more scoreable clips in each group separate and the complete examples reach the target range. Investigate practice overlap before freezing. The previous whole-clip threshold of **1.48** belongs to the baseline version.
4. After freezing, use **10** to label and record new held-out attempts, then **11** to score them. Report abstentions along with the other outcomes. Keep the threshold fixed when reviewing test results.
5. Use **12** for live scoring. Press **C**, **I** or **U** to label the attempt before pressing **Space**.

In the webcam window, press **Space** once. After the countdown, hold the arm down and still until the window says **Ready**, then raise and lower it once. Keep the upper arm in view and hold the returned position until the result. The recorder saves completed attempts and automatic failure outcomes for review. Press **Q** to cancel.

## How detection works

- The arm needs 500 ms of stable tracking at rest (at or below 25° with at most 5° of variation). Start requires at least 15° above that resting angle for 180 ms. Return requires staying within 7° of rest for 400 ms.
- Start must be confirmed within 8 seconds of capture. After start, the default timeout is 25 seconds from movement onset. Option **15** changes the countdown and movement timeout. Changing the movement timeout invalidates the current calibration for that setup.
- A started attempt without a confirmed return yields **Incomplete return**. If the arm stays down without a confirmed start, the outcome is **No repetition start detected**. Missing arm tracking after resting calibration, a frame gap over 250 ms after rest, or inability to establish rest yields **Unable to judge**.
- Options 8 and 11 replay the exact capture timestamps, angles, and tracking decisions saved in the new `.live_pose.jsonl` sidecar. The sidecar also stores normalized MediaPipe landmarks at capture precision. Each segment runs from the first sustained start crossing to the first sustained return crossing. DTW uses 101 interpolated angle values spaced across measured movement time.
- Earlier practice and test clips lack that sidecar, so the new scorer reports their exact capture measurements as unavailable. Their existing scores and clips remain available. A saved legacy reference can still be processed from its AVI and measured timestamps, with its source identified as `video_reprocessed`. Option 7 and `check_clip.py` independently reprocess the AVI for visual diagnostics.
- The detector handles one attempt per recording. Its angle and timing parameters are prototype defaults. Shoulder-line tilt remains an exploratory cue, and 2D elevation depends on camera placement and movement plane.

## Data and tests

Right-arm outputs go under `clips/<position>/segmented_v1/`, `traces/<position>/segmented_v1/capture_pose_v1/`, and `logs/<position>/segmented_v1/`. Left-arm outputs add `/left/` after `<position>` in each path. The scorer saves measured frame traces, detected boundaries, DTW scores and outcomes. Review and calibration checks also have separate segmented folders. The adjacent timestamped baseline folder contains the previous code, recordings, settings and evaluation outputs.

New recording JSON contains `detector_status`, `termination_reason`, `termination_elapsed_ms`, `repetition_start_ms`, `repetition_end_ms`, and `repetition_duration_s`. Boundaries are relative to the first captured frame. After a successful return, the clip includes the extra frames needed to confirm the 400 ms dwell. A timeout after movement start reports `incomplete_return` and `movement_timeout`; the confirmed end and duration remain blank.

## Anatomical arm selection and data isolation

The selected arm is anatomical, following MediaPipe's landmark labels. Left shoulder/elbow are landmarks **11/13**, right shoulder/elbow **12/14**. The primary elevation uses only shoulder and elbow pixel coordinates: `degrees(atan2(abs(elbow_x - shoulder_x), elbow_y - shoulder_y))`. It is 0 degrees with the upper arm down and about 90 degrees when horizontal. Left wrist 15 and right wrist 16 may be observed for supplementary analysis, but neither wrist nor a hip is required for this angle or its tracking validity. The metric is `theta_v_deg` (image-vertical-relative), not trunk-relative `theta_ht_deg` or anatomical `trunk_lean_deg`. Shoulder-line tilt is separate and exploratory. This is a 2D camera-plane estimate, not validated anatomical range of motion.

Pose inference and AVI capture use the **unmirrored** webcam frame. Only the preview is flipped horizontally. Skeleton highlighting transforms display coordinates while retaining anatomical landmark labels and the raw-frame angle. Do not infer the selected arm from screen-left or screen-right.

Historical right-arm data and settings remain at `clips/seated/segmented_v1/`, `traces/seated/segmented_v1/` and the matching logs, calibration and review paths. New left-arm outputs use `clips/seated/left/segmented_v1/`, `traces/seated/left/segmented_v1/` and matching arm-specific paths. Within each clip tree, `diagnostic/`, `reference/`, `practice/` and `test/` remain separate. Right-arm settings stay in the existing `conditions` records; left-arm reference and calibration live in separate `arm_conditions` records. This keeps future bilateral work able to reuse shared capture and geometry components without adding bilateral scoring now. New left research output paths are gitignored. The tracked `settings.json` may change when selecting an arm or freezing calibration; review it before any future commit.

The local single-participant prototype has independently frozen seated thresholds of **2.224** for right and **4.056** for left. The left calibration and its reference identity are participant-specific local settings and are not included in the A5 pull request. A fresh checkout must collect and freeze its own left practice calibration, or receive the matching reference, recordings and settings privately; a threshold number alone does not constitute a usable calibration. Do not tune either threshold on held-out recordings.

The initial held-out summaries classified **8/8 definitively labeled recordings for each arm**; uncertain recordings were analyzed separately. For each arm separately, the two-sided exact 95% binomial confidence interval for 8/8 is approximately **[0.63, 1.00]**. These single-participant prototype results do not establish a difference between the arm-specific thresholds or clinical validity. Partial landmark occlusion can cause tracking abstention. Independent anatomical angle accuracy validation is still pending, and the 2D image angle is not a clinically validated range-of-motion measure.

New `.live_pose.jsonl` rows retain the measured timestamp, selected anatomical `arm`, tracking status, exact angle, normalized landmarks and an optional `detector_state`. Offline scoring reuses those saved values. Historical right-arm schema-3 rows without `arm` remain readable **only when their recording metadata explicitly says `arm=right`**; the metadata takes precedence over the active menu selection. An armless recording with no reliable arm metadata is rejected as ambiguous, and conflicting metadata and row arms are rejected. Schema-3 recording metadata saved `arm` even before A5, while its older pose rows did not. AVI reprocessing remains diagnostic. A completed repetition means the selected arm returned to rest, even if elevation is insufficient for exercise success.

Option 17 records `category=diagnostic` in a separate folder for physical checks; it is excluded from reference, calibration, practice scoring and held-out evaluation. The unmerged G1 schema-v1 exporter does not export these A5 diagnostics. A future explicit schema-v2 diagnostic export, if agreed by the team, should preserve `category=diagnostic`, set `block_role=practice` for exploratory analysis, and never relabel a diagnostic clip as calibration or held-out test data.

Draft G1 PR #5 still exports the original three right-arm clips only. Its 78-column schema is unchanged here. Future left-arm exports must use `side=left`, the anatomical left landmarks, and the saved `theta_v_deg` without filling unsupported trunk-relative fields. Extending that unmerged exporter and Member B's configurable-angle integration require separate review; this milestone does not change either implementation.

To run unit tests from this folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```
