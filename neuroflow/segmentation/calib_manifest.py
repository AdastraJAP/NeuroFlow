"""
Calibration manifest (owner: Member B): where rest_ref / rom_ref came from.

segment.calibrate() returns two bare numbers, so nothing stops references made on
one angle (or person) from being used on another. For real-data runs (G2, B8) the
references travel in a small JSON manifest instead:

    {"manifest_version": 1, "angle_col": "theta_v_deg",
     "rest_ref": ..., "rom_ref": ...,            exact values from calibrate()
     "subject_id": "P001", "session_id": "...",  whose calibration it is
     "source": "...", "source_blocks": [...],    what was calibrated on
     "source_files": [{"name": ..., "sha256": ...}], "method": "segment.calibrate"}

check_manifest(m, frames, angle_col) is the only way B8 gets the numbers. It
raises when:
  - the manifest's angle_col differs from the angle being segmented,
  - the subject differs from the frames' subject,
  - a source block is one of the blocks being evaluated (test clips stay held out),
  - the values are not finite or rom_ref <= 0.
A different session only warns (and is reported): on G1 the calibration comes
from separate practice clips, so a same-session calibration isn't possible yet.
session_id may be null when the source has no session ID (G1 practice clips);
that also warns, and "notes" says why.

Make one from Adastra's V4 practice sidecars (*.jsonl):
    python -m neuroflow.segmentation.calib_manifest data/g1/practice/complete_0{7,8,9}.jsonl \\
        --subject P001 --no-session --note "practice metadata has no session ID" \
        --out data/g1/calibration_manifest.json
"""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from neuroflow.segmentation.segment import ANGLE_COLS, calibrate

MANIFEST_VERSION = 1
GAP_MS = 1000.0       # practice clips are joined back to back with 1 s of nothing between


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def practice_frames(paths, angle_col="theta_v_deg"):
    """Adastra's V4 practice sidecars (one JSON object per frame) joined into one
    calibration block. Frames marked not usable get NaN."""
    parts, off = [], 0.0
    for p in paths:
        rows = [json.loads(line) for line in open(p)]
        t = np.array([float(r["timestamp_ms"]) for r in rows]) + off
        th = [r["arm_elevation_deg"] if r.get("usable") else np.nan for r in rows]
        parts.append(pd.DataFrame({"t_capture_ms": t, angle_col: th}))
        off = t.max() + GAP_MS
    return pd.concat(parts, ignore_index=True)


def make_manifest(frames, angle_col, subject_id, session_id, source, source_blocks, source_files=(),
                  notes=""):
    """Run calibrate() on a calibration block and record where the numbers came from.
    session_id None = the source has no session ID (recorded as null, never guessed)."""
    rest, rom = calibrate(frames, angle_col=angle_col)
    return {
        "manifest_version": MANIFEST_VERSION,
        "angle_col": angle_col,
        "rest_ref": rest,
        "rom_ref": rom,
        "subject_id": str(subject_id),
        "session_id": None if session_id is None else str(session_id),
        "source": source,
        "source_blocks": [str(b) for b in source_blocks],
        "source_files": [{"name": os.path.basename(f), "sha256": _sha256(f)} for f in source_files],
        "method": "segment.calibrate",
        "notes": notes,
    }


def save_manifest(m, path):
    with open(path, "w") as f:
        json.dump(m, f, indent=2)


def load_manifest(path):
    with open(path) as f:
        m = json.load(f)
    need = ["manifest_version", "angle_col", "rest_ref", "rom_ref", "subject_id",
            "session_id", "source", "source_blocks"]
    missing = [k for k in need if k not in m]
    if missing:
        raise ValueError(f"{path}: calibration manifest misses {missing}")
    if m["manifest_version"] != MANIFEST_VERSION:
        raise ValueError(f"{path}: manifest_version {m['manifest_version']}, expected {MANIFEST_VERSION}")
    return m


def check_manifest(m, frames, angle_col):
    """(rest_ref, rom_ref, warnings) if the manifest fits these frames and angle; raises otherwise."""
    if m["angle_col"] not in ANGLE_COLS:
        raise ValueError(f"manifest angle_col {m['angle_col']!r} is not one of {ANGLE_COLS}")
    if m["angle_col"] != angle_col:
        raise ValueError(f"calibration was made on {m['angle_col']}, not {angle_col}")
    rest, rom = float(m["rest_ref"]), float(m["rom_ref"])
    if not (np.isfinite(rest) and np.isfinite(rom) and rom > 0):
        raise ValueError(f"manifest values not usable: rest_ref {rest}, rom_ref {rom}")
    subjects = sorted(frames["subject_id"].astype(str).unique())
    if subjects != [m["subject_id"]]:
        raise ValueError(f"calibration is for subject {m['subject_id']}, frames hold {subjects}")
    overlap = sorted(set(m["source_blocks"]) & set(frames["block_id"].astype(str)))
    if overlap:
        raise ValueError(f"calibration source blocks {overlap} are also being evaluated; "
                         "calibrate on separate clips")
    warnings = []
    sessions = sorted(frames["session_id"].astype(str).unique())
    if m["session_id"] is None:
        warnings.append(f"calibration session not recorded ({m.get('notes') or 'no session ID'}); "
                        f"frames are {sessions}")
    elif sessions != [m["session_id"]]:
        warnings.append(f"calibration session {m['session_id']} differs from frames' {sessions}")
    return rest, rom, warnings


def main():
    p = argparse.ArgumentParser(description="Calibration manifest from V4 practice sidecars")
    p.add_argument("jsonl", nargs="+", help="practice sidecars, e.g. complete_07.jsonl")
    p.add_argument("--subject", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--session")
    g.add_argument("--no-session", action="store_true", help="the source has no session ID")
    p.add_argument("--note", default="", help="free text kept in the manifest (e.g. file provenance)")
    p.add_argument("--angle", default="theta_v_deg")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    blocks = [os.path.splitext(os.path.basename(f))[0] for f in a.jsonl]
    m = make_manifest(practice_frames(a.jsonl, a.angle), a.angle, a.subject,
                      None if a.no_session else a.session,
                      source="V4 practice sidecars, joined back to back", source_blocks=blocks,
                      source_files=a.jsonl, notes=a.note)
    save_manifest(m, a.out)
    print(f"rest_ref {m['rest_ref']:.3f} deg, rom_ref {m['rom_ref']:.3f} deg on {a.angle} "
          f"from {blocks} -> {a.out}")


if __name__ == "__main__":
    main()
