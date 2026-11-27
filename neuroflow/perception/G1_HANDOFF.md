# Member A G1 handoff status

## Source and contract

The V4 source is `NeuroFlow_Shoulder_to_Elbow_and_ShoulderLine_tilt_approach`. Three real clips were validated locally: a normal complete repetition, a slow complete repetition, and an insufficient-elevation attempt with complete return. All have raw AVI, metadata JSON, exact timestamp CSV and `.live_pose.jsonl`. The insufficient-elevation condition is coded `rom_cap` to match Member B's mock vocabulary. Capture completion means the arm returned to rest; task-quality completion is a separate label.

The final local outputs are `data/g1/frames.parquet` (749 rows, 78 columns), `data/g1/frame_provenance.parquet`, `data/g1/manifest.json`, and `data/g1/selection.csv`. The manifest links each clip to its raw video, metadata, timestamp CSV and pose sidecar with SHA-256 hashes. A local validator reproduces the final source-to-parquet checks. All of `data/g1` is gitignored and distributed separately because it contains personal recording data.

The 78-column v1 frame contract comes from `fixtures/make_mock_frames.py`. The exporter preserves that shape and the original saved V4 vertical angle. The original research specification calls for anatomical trunk lean and trunk-relative humerothoracic angle. Hip-free shoulder-line tilt is a different signal and remains in a supplemental provenance table.

## Integration status

The three clips have 749 decoded frames in total. Their metadata frame counts, saved pose rows and timestamp rows match. Indices start at zero and are contiguous within each clip, timestamps increase strictly, and every saved frame is retained. The capture code saves pose and timestamp from a single loop iteration before writing that raw frame, then increments the frame counter. It does not skip frames conditionally. The writer does not check per-frame write success; decoded count and source-code inspection are the strongest historical alignment evidence, short of an embedded visual frame counter or frame hash.

No frame in these three clips has a valid **pair** of hip landmarks under the source confidence and in-frame rules. Consequently `theta_ht_deg` and `trunk_lean_deg` are NaN on every row. The final parquet has all 78 schema columns and 749 exact saved `theta_v_deg` values, with zero valid `theta_ht_deg` values. A diagnostic run of B's unchanged fixed-threshold segmenter returned zero reps. B's segmentation and feature extraction require `theta_ht_deg`; these clips cannot yet unblock B8 or real-data segmentation until the team agrees to use `theta_v_deg` as a separate angle input. The B9 tool can use the frame indices and capture timestamps after filtering one clip, but its current `theta_ht_deg` trace will be blank. B7's trunk lean summary will remain NaN. These are measurement limitations, not exporter failures.

## Proposed interface for team approval

1. Give B's segmentation, calibration, rep-feature and template functions an explicit angle-column parameter, defaulting to `theta_ht_deg` for current mock behavior. Pass `theta_v_deg` for V4 seated clips. Propagate that choice through calls and record it in derived results so two angle definitions are not mixed. Keep `theta_ht_deg` missing in these clips. The fixed and ROM-normalized segmentation thresholds must be checked for the new angle definition using separate calibration and labeled development data; do not tune on these held-out T clips. G2 and B8 can proceed only after this interface and threshold policy are agreed.
2. Keep the signed saved shoulder-line tilt separate from anatomical `trunk_lean_deg`. In the raw unmirrored frontal view, positive `shoulder_tilt_deg` means the subject's right shoulder is higher. The exact formula, mirror rule and baseline option are in `frames_data_dictionary.md`. If B7 needs it, agree on a separately named field or consume the supplemental provenance table; use the absolute magnitude only after deciding whether to subtract rest baseline. Do not silently substitute tilt for trunk lean.
3. Define `hike_ratio`, `jitter`, `arm_len_ratio` and aggregate `frame_conf` before treating those fields as real measures.

The frozen seated segmented DTW threshold 2.224 was not modified. No held-out clips were used for tuning. The local `data/g1` output is ignored by git and must not be pushed without Member A's explicit approval.

## Verification

The final local validator confirms 749 rows, 78 columns, exact saved vertical angles and shoulder tilts, decoded frames matched to metadata, sidecar and timestamp rows, contiguous unique zero-based frame indices, strictly increasing timestamps, and exact B9 PR #3 capture-time lookup for every frame. All three clips have zero invalid tracking frames; the exporter and regression tests also retain deliberately invalid synthetic frames without renumbering. The new exporter tests pass (3/3). The root `tests` suite passes (89 passed, 1 skipped), and the V4 folder's own `tests` suite passes from its directory (32 passed, 2 subtests passed). Running unconstrained `pytest` at the repository root mixes two independent import layouts and collides on duplicate `visibility_test.py` names, producing seven collection errors; the scoped runs above use each project's intended import root. No existing implementation was changed to address that collection layout.
