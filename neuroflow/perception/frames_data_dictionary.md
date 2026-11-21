# G1 frame data dictionary

Schema version 1 follows `fixtures/make_mock_frames.py`: 14 base columns, 55 landmark columns, and 9 derived columns. Numerical missing values are IEEE NaN in parquet. No unavailable value is replaced by zero. Video and pixel coordinates are from the **raw, unmirrored** AVI; x increases image-right and y increases down. The mirrored webcam preview was not saved. MediaPipe `left`/`right` labels refer to anatomical sides.

| Field | Type | Definition and units |
| --- | --- | --- |
| `schema_version` | int64 | Contract version, 1. |
| `subject_id`, `session_id` | string | Verified pseudonymous participant and session identifiers supplied in `selection.csv`. |
| `block_id` | string | Source clip ID, e.g. `T_01`; unique with `frame_idx`. |
| `condition`, `level` | string | Confirmed task condition and level supplied in `selection.csv`. |
| `side` | string | `right`; source metadata arm must agree. |
| `frame_idx` | int64 | Zero-based captured and decoded video frame ordinal. |
| `t_capture_ms` | int64 | Saved monotonic elapsed capture time in milliseconds; not video FPS time or wall time. |
| `img_w`, `img_h` | int64 | Decoded raw AVI dimensions in pixels. |
| `pose_model` | string | `mediapipe_pose_landmarker_lite`, inferred from V4 source. |
| `pose_model_ver` | string | `unrecorded`; actual model/runtime version was not recorded in clip metadata. |
| `detected` | bool | Exact saved `pose_detected` decision. |
| `lm{N}_x_px`, `lm{N}_y_px` | float64 | MediaPipe normalized x/y multiplied by raw frame width/height. Out-of-frame coordinates remain available; absent landmark is NaN. N = 0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24. |
| `lm{N}_z` | float64 | Raw MediaPipe normalized-image landmark z, unchanged; no physical unit. |
| `lm{N}_vis`, `lm{N}_pres` | float64 | Saved MediaPipe visibility and presence probabilities in [0,1]. |
| `theta_v_deg` | float64 | Exact saved V4 `arm_elevation_deg`: right shoulder-to-elbow vector against downward **image vertical**, unsigned degrees. NaN when source `usable=0`. |
| `theta_ht_deg` | float64 | Derived angle in degrees between right shoulder-to-elbow and downward mid-shoulder-to-mid-hip vector. NaN unless both shoulders, right elbow and both hips are in frame, finite, and visibility/presence >= source `min_visibility`. This is a 2D projected trunk-relative angle. |
| `trunk_lean_deg` | float64 | Derived frontal 2D trunk lean in degrees from hip midpoint H to shoulder midpoint S. Positive means toward anatomical right. Let `r = sign(x_right_shoulder - x_left_shoulder)` in the raw image; `lean = atan2(r*(Sx-Hx), Hy-Sy)*180/pi`. All four joints must pass the same validity check. This sign remains the same under horizontal mirroring if anatomical labels are preserved. A 30 px anatomical-right shoulder shift over 300 px vertical rise gives +5.71°. A 30 px anatomical-left shift gives -5.71°. This is not shoulder-line tilt. |
| `elbow_deg` | float64 | Interior 2D angle at right elbow between vectors to right shoulder and right wrist, degrees; 180° straight. NaN unless all three joints pass validity check. |
| `hike_ratio` | float64 | NaN. V4 has no agreed shoulder-hike baseline or ratio definition. |
| `in_frame` | bool | Right shoulder and elbow have finite x/y inside the image and saved visibility/presence >= `min_visibility`. |
| `jitter`, `arm_len_ratio`, `frame_conf` | float64 | NaN. Their real-data definitions have not been agreed; landmark confidence is retained in `vis`/`pres`. |

`frame_provenance.parquet` is a local supplemental table keyed by `(block_id, frame_idx)` with `t_capture_ms`, exact `source_usable`, `source_status`, and saved `shoulder_tilt_deg`. The tilt is **not** put in `trunk_lean_deg`. Adding it to v1 `frames.parquet` or changing the meaning of `theta_ht_deg` needs team agreement. Member B's B7 uses the absolute value of valid signed `trunk_lean_deg`.

## Proposed signed shoulder-line tilt convention

V4's saved value comes from the raw, unmirrored frame. Let R and L be the right and left shoulder pixel points. The V4 formula is `wrap[-90,90)(atan2(Ly - Ry, Lx - Rx) * 180/pi)`. In these frontal raw frames `Lx > Rx`, so **positive means the subject's right shoulder is higher** (`Ry < Ly`); negative means the left shoulder is higher. For example, 20 px right-higher difference across 200 px horizontal separation is +5.71 degrees. The recorded values are already signed; do not flip or recompute them in this export.

If a downstream system processes horizontally mirrored frames, it must use the raw recorded value or normalize the horizontal direction using the anatomical shoulder labels; applying the raw V4 formula to mirrored coordinates would reverse the sign. For exploratory compensation, agree on whether to use `abs(shoulder_tilt_deg)` or a **rest-baseline-corrected** magnitude. V4's existing `tilt_change` subtracts a median baseline from the first 8 valid tilt frames and wraps the result to [-90,90). The two magnitudes answer different questions, so the team should choose explicitly before using one as a B7 feature.
