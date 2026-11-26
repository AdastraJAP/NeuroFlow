# Neuro-Flow — automatic repetition detection on Windows

Open **START_HERE.bat** in this folder. The menu shows the active position, arm, reference and segmented DTW calibration. Option **14** switches between seated and standing. Each position needs its own reference and practice calibration.

The launcher prints the active project path. The baseline folder's **START_HERE.bat** opens this updated project. Use **START_LEGACY_BASELINE.bat** inside the baseline folder to explicitly run the archived 10-second workflow.

## Record and calibrate

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

New outputs go under `clips/<position>/segmented_v1/`, `traces/<position>/segmented_v1/capture_pose_v1/`, and `logs/<position>/segmented_v1/`. The scorer saves measured frame traces, detected boundaries, DTW scores and outcomes. Review and calibration checks also have separate segmented folders. The adjacent timestamped baseline folder contains the previous code, recordings, settings and evaluation outputs.

New recording JSON contains `detector_status`, `termination_reason`, `termination_elapsed_ms`, `repetition_start_ms`, `repetition_end_ms`, and `repetition_duration_s`. Boundaries are relative to the first captured frame. After a successful return, the clip includes the extra frames needed to confirm the 400 ms dwell. A timeout after movement start reports `incomplete_return` and `movement_timeout`; the confirmed end and duration remain blank.

To run unit tests from this folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```
