# Proposed bilateral frame contract for Members B and C

**Proposal only; not implemented.** G1 PR #5 defines a 78-column schema v1 with one anatomical `side` and one `theta_v_deg` per row. It exports three unilateral right-arm clips. A6 does not change that exporter, its values, or Member B/C interfaces. The V4 bilateral sidecar is a separate local capture format (schema 6), not a Parquet contract version.

## Recommendation: one row per captured video frame

Keep a single `(block_id, frame_idx)` row for each decoded raw video frame. This preserves B9's direct frame-index lookup and the exact capture timestamp shared by both arms. The alternative, one row per arm per frame, duplicates `frame_idx` and timestamp, breaks current uniqueness checks, and requires B9 and study joins to add `side` everywhere. That alternative would allow a long-form single-angle segmenter, but the duplicate video-frame mapping is a larger migration.

Propose **frames schema v2** with the existing 78 v1 columns retained and explicit additional fields:

| Proposed field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | int64 | `2` for this contract; v1 remains readable as v1. |
| `exercise_mode` | string | `right`, `left`, or `both`, from saved capture metadata. |
| `angle_definition` | string | `shoulder_to_elbow_vs_downward_image_vertical`; record provenance rather than infer it from a column name. |
| `theta_v_left_deg`, `theta_v_right_deg` | float64, NaN missing | Independent anatomical shoulder-to-elbow elevations from the same captured frame. |
| `left_usable`, `right_usable` | bool | Independent saved tracking decisions; false does not remove the frame. |
| `left_status`, `right_status` | string | Saved reason such as `ok`, `no_person`, or `unreliable_arm_joints`. |
| `shoulder_tilt_deg` | float64, NaN missing | Signed shoulder-line orientation, separate from anatomical `trunk_lean_deg`; include only after team agreement. |

Retain `frame_idx` and `t_capture_ms` as int64, zero-based contiguous and strictly increasing within each clip. The existing landmark columns already include anatomical shoulders 11/12 and elbows 13/14 with x/y, visibility and presence. Use raw unmirrored image coordinates. Do not substitute zero for an unavailable angle or confidence. `theta_ht_deg` and anatomical `trunk_lean_deg` remain NaN when hip-based anatomy is unsupported. `frame_conf` remains unavailable until the team defines and validates it.

For a bilateral row, the v1 single-side `side` and single-angle `theta_v_deg` have no unique anatomical value. Recommend nullable `side` and NaN `theta_v_deg` in schema v2 for `exercise_mode=both`; explicit left/right fields carry the measurements. This requires a versioned validator and consumer update. For unilateral v2 rows, `side` and `theta_v_deg` retain their v1 meaning; the matching explicit arm field may also be populated from the same saved measurement. Do not relabel bilateral data as one side or average the two arms.

Member B can run the existing configurable-angle segmenter separately on `theta_v_left_deg` and `theta_v_right_deg`, retaining arm and angle provenance in each repetition, then evaluate a **separate agreed** pairing policy. Independent arm-level boundaries must remain available. B8/G2 manual labels should reference original `(block_id, frame_idx, t_capture_ms)` values. Member C should distinguish `exercise_mode=both` from unilateral configurations in templates, calibration and reports; no unilateral DTW threshold or score transfers to bilateral data. The team must decide bilateral repetition pairing, confidence aggregation, compensation features, calibration roles, and a scorer before implementation.
