# Unilateral V4 frames schema v2

Schema v2 appends **`block_role`** and **`angle_col`** to the original 78-column
[G1 schema v1](frames_data_dictionary.md), in that order. The first 78 names,
units, coordinates and missing-value meanings are unchanged. A v2 frame table
has exactly 80 columns and one row per decoded, captured video frame. This is
for one anatomical arm per block; it does not define a bilateral aggregate.

| Field | Parquet type | Required | Meaning and validation |
| --- | --- | --- | --- |
| `schema_version` | int64 | Yes | `2` on every row. Existing v1 field. |
| `block_role` | string | Yes | Original recording role: exactly `calib`, `practice` or `test`, constant within `block_id`. Explicitly supplied in the verified selection and required to match source recording `category`. It is never inferred from `block_id`. |
| `angle_col` | string | Yes | Selected physical angle column, constant within `block_id`. Current saved unilateral V4 measurements support only `theta_v_deg`. Source metadata must identify `shoulder_to_elbow_vs_downward_image_vertical`. The selected angle may be NaN on invalid frames without changing this provenance tag. |
| `subject_id` | string | Yes | Verified pseudonymous participant identifier from the local selection. Source V4 metadata does not encode it, so the exporter cannot independently establish the real identity. |
| `session_id` | nullable string | No | Verified recording session, or null when not recorded. An all-null column is still written as Arrow/Parquet `string`; no session is guessed from the participant or file path. |
| `block_id` | string | Yes | Explicit unique recording-block identifier. `(block_id, frame_idx)` is unique. It does not encode role. `clip_id` is the source filename stem and may differ from `block_id`. |
| `condition`, `level` | string | Yes | Verified experimental condition and level from the selection. Existing v1 meanings, including the valid level `none`. |
| `side` | string | Yes | Anatomical `left` or `right`, matched against recording metadata and explicit pose-row arms. Pre-A5 armless schema-3 rows are accepted only with explicit right-arm metadata. |
| `frame_idx` | int64 | Yes | Exact zero-based, contiguous saved and decoded video-frame ordinal. |
| `t_capture_ms` | int64 | Yes | Exact saved monotonic capture timestamp in milliseconds, strictly increasing within each block and equal in sidecar and timestamp CSV. It is not derived from AVI FPS. |
| `detected`, `in_frame` | bool | Yes | Existing v1 semantics: saved pose-detection decision and selected shoulder/elbow landmark validity. `in_frame` does not replace the saved tracking decision. |
| `theta_v_deg` | nullable float64 | No | Exact saved selected shoulder-to-elbow elevation relative to downward image vertical. NaN when source `usable=0`; wrist and hips are not needed. |
| `theta_ht_deg`, `trunk_lean_deg`, `elbow_deg` | nullable float64 | No | Existing conditional 2D geometry, adapted to the selected anatomical arm. Hip or wrist measurements are computed only when all supporting landmarks pass the existing visibility/presence and image-bounds check; otherwise NaN. Shoulder tilt is never substituted for trunk lean. |
| Landmark coordinates, visibility, presence | nullable float64 | No | The unchanged 55 v1 fields for IDs 0, 7, 8, 11–16 and 23–24, in raw unmirrored video coordinates. Missing source values remain NaN. |
| `hike_ratio`, `jitter`, `arm_len_ratio`, `frame_conf` | nullable float64 | No | NaN until supported definitions and measurements exist. No synthetic confidence or zero fill. |

`img_w`, `img_h` remain int64 decoded AVI dimensions. `pose_model` and
`pose_model_ver` remain strings with the existing v1 meanings. All numerical
missing values are IEEE NaN in Parquet.

## Supplemental provenance and recording roles

`frame_provenance.parquet` remains separate, keyed by `(block_id, frame_idx)`.
It contains the same `t_capture_ms`, exact saved `source_usable` (bool),
`source_status` (string), and saved signed `shoulder_tilt_deg` (nullable
float64). Positive tilt retains V4's **anatomical right shoulder higher**
convention. This is not `trunk_lean_deg`; no `max_shoulder_tilt_deg` repetition
feature is calculated here.

Historical T_01/T_02/T_05 selections have `block_role=test`. Historical
`complete_07–09` were recorded as `practice` and retain that frame role, with
null `session_id` where no session was recorded. If B later uses these practice
recordings to build a calibration reference, a **separate calibration
manifest** must identify the source block IDs, participant, selected angle,
anatomical side and reference values. That subsequent analysis use does not
rewrite original recording roles. A new dedicated calibration recording may
have `block_role=calib` when its source category agrees.

The source manifest records relative source paths and SHA-256 hashes of each
AVI, metadata JSON, timestamp CSV and pose sidecar. Relative paths resolve
under the `--source-root` and each selection row's `source_rel_dir`; the manifest
does not need a private absolute Windows path. The exporter checks counts,
resolution, indices, timestamps, arm identity and angle definition before
writing any output. Historical AVI write calls were not independently checked
at capture time, so decoded count plus sidecar agreement does not prove every
historical frame/pose association independently.

## Contract boundaries

This frame export does not define B's repetition role conversion, DTW curve
provenance, C's scoring behavior, or a bilateral frame schema. The proposed
calibration-corrected `max_shoulder_tilt_deg` definition is unresolved and is
not implemented here. The frozen G1 v1 export and data dictionary remain
unchanged.
