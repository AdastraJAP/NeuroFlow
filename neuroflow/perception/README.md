# G1 saved V4 frame export

The exporter reads the exact `.live_pose.jsonl` measurements recorded by V4. It does not run MediaPipe again. It checks every zero-based source index and timestamp against the capture timestamp CSV, decodes every AVI frame, and checks the metadata count and resolution. A mismatch aborts before any parquet is written.

Create `data/g1/selection.csv` locally with verified pseudonymous participant and session IDs. It is ignored by git and distributed with the research dataset. The V4 clip metadata does not itself contain participant/session IDs.

```csv
clip_id,subject_id,session_id,condition,level
T_01,VERIFIED_SUBJECT,VERIFIED_SESSION,normal,unspecified
T_02,VERIFIED_SUBJECT,VERIFIED_SESSION,slow,unspecified
T_05,VERIFIED_SUBJECT,VERIFIED_SESSION,rom_cap,insufficient_elevation
```

Member A confirmed T_01 as normal complete, T_02 as slow complete, and T_05 as insufficient elevation with complete return. `rom_cap` follows Member B's mock condition vocabulary; `insufficient_elevation` gives its specific level. The `unspecified` levels avoid claiming a measured 2x speed or target cap. `expected_labels.csv` marks T_05 `incomplete` for task quality while metadata records `confirmed_return` for capture termination.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m neuroflow.perception.export_v4 --selection data\g1\selection.csv
.\.venv\Scripts\python.exe -m pytest tests\test_export_v4.py -q
```

The output directory is `data/g1` by default. It contains `frames.parquet`, `frame_provenance.parquet` and `manifest.json`. The directory is gitignored because it contains personal pose data. The manifest records source file paths and SHA-256 hashes. A separate local validation script may accompany the separately distributed dataset. `block_id` is the clip identifier and the B9 join key, alongside zero-based `frame_idx`. B9 must receive one clip at a time, filtered by `block_id`.

The 78 columns in `frames.parquet` follow `fixtures/make_mock_frames.py` schema version 1. See [frames_data_dictionary.md](frames_data_dictionary.md) and [G1_HANDOFF.md](G1_HANDOFF.md) for semantics and known limitations.

## Separate unilateral schema-v2 export

The v1 command above remains the historical G1 reproduction path. The
independent v2 command appends required `block_role` and `angle_col` fields to
those 78 frame columns. It accepts verified right- or left-arm V4 recordings,
retains a nullable session ID, and never reruns MediaPipe. It does not export
bilateral captures or implement repetition-level calibration or scoring.

Create a **local, gitignored** `data/frames_v2/selection.csv` with exactly these
headers:

```csv
source_rel_dir,clip_id,block_id,subject_id,session_id,condition,level,side,block_role,angle_col
segmented_v1/test,T_01,T_01,VERIFIED_SUBJECT,VERIFIED_SESSION,normal,none,right,test,theta_v_deg
```

`source_rel_dir` is relative to `--source-root`, which defaults to the V4
`clips/seated/` directory. For a left recording, an example subdirectory is
`left/segmented_v1/test`. Each selection row is checked against the source
metadata's clip, anatomical arm, original category and V4 image-vertical angle
definition. All required identities and roles must be verified before export;
the source metadata does not itself verify participant or session identity.
Leave `session_id` empty when it was not recorded. Do not put personal selection
files in Git. Historical practice clips used later for calibration retain
`block_role=practice`; a separate calibration manifest records that later use.

```powershell
.\.venv\Scripts\python.exe -m neuroflow.perception.export_v4_v2 --selection data\frames_v2\selection.csv
.\.venv\Scripts\python.exe -m pytest tests\test_export_v4_v2.py tests\test_export_v4.py -q
```

The default output is the ignored `data/frames_v2/` directory with
`frames.parquet` (80 columns), `frame_provenance.parquet`, and `manifest.json`.
The manifest contains relative source paths and SHA-256 hashes. See
[frames_v2_data_dictionary.md](frames_v2_data_dictionary.md) for types, missing
values and provenance rules. Keep v1 and v2 outputs in separate directories.
