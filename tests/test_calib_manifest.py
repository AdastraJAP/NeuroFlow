"""Calibration manifest: references carry their angle, subject and source (PR #3 re-review)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from neuroflow.segmentation.calib_manifest import (
    check_manifest, load_manifest, make_manifest, practice_frames, save_manifest)
from neuroflow.segmentation.segment import calibrate

FX = os.path.join(os.path.dirname(__file__), "..", "fixtures")


@pytest.fixture(scope="module")
def frames():
    return pd.read_parquet(os.path.join(FX, "mock_frames_normal.parquet"))


@pytest.fixture()
def manifest(frames, tmp_path):
    sub = frames["subject_id"].iloc[0]
    calib = pd.read_parquet(os.path.join(FX, "mock_frames_slow.parquet"))   # a separate block
    src = tmp_path / "calib.parquet"
    calib.to_parquet(src)
    return make_manifest(calib, "theta_v_deg", sub, frames["session_id"].iloc[0],
                         source="mock slow block", source_blocks=["slow"], source_files=[src])


def test_values_are_exactly_calibrate(manifest):
    calib = pd.read_parquet(os.path.join(FX, "mock_frames_slow.parquet"))
    assert (manifest["rest_ref"], manifest["rom_ref"]) == calibrate(calib, angle_col="theta_v_deg")
    assert len(manifest["source_files"][0]["sha256"]) == 64


def test_round_trip_and_check(manifest, frames, tmp_path):
    path = tmp_path / "m.json"
    save_manifest(manifest, path)
    rest, rom, warnings = check_manifest(load_manifest(path), frames, "theta_v_deg")
    assert (rest, rom) == (manifest["rest_ref"], manifest["rom_ref"]) and warnings == []


def test_other_angle_is_rejected(manifest, frames):
    with pytest.raises(ValueError, match="made on theta_v_deg, not theta_ht_deg"):
        check_manifest(manifest, frames, "theta_ht_deg")


def test_other_subject_is_rejected(manifest, frames):
    with pytest.raises(ValueError, match="calibration is for subject"):
        check_manifest({**manifest, "subject_id": "P999"}, frames, "theta_v_deg")


def test_calibrating_on_an_evaluated_block_is_rejected(manifest, frames):
    with pytest.raises(ValueError, match="also being evaluated"):
        check_manifest({**manifest, "source_blocks": ["normal"]}, frames, "theta_v_deg")


@pytest.mark.parametrize("rest, rom", [(np.nan, 80.0), (10.0, np.inf), (10.0, 0.0), (10.0, -5.0)])
def test_unusable_values_are_rejected(manifest, frames, rest, rom):
    with pytest.raises(ValueError, match="not usable"):
        check_manifest({**manifest, "rest_ref": rest, "rom_ref": rom}, frames, "theta_v_deg")


def test_other_session_only_warns(manifest, frames):
    _, _, warnings = check_manifest({**manifest, "session_id": "s9"}, frames, "theta_v_deg")
    assert len(warnings) == 1 and "session" in warnings[0]


def test_incomplete_manifest_is_rejected(manifest, tmp_path):
    path = tmp_path / "m.json"
    json.dump({k: v for k, v in manifest.items() if k != "angle_col"}, open(path, "w"))
    with pytest.raises(ValueError, match="misses"):
        load_manifest(path)


def test_practice_sidecars_are_joined_with_gaps(tmp_path):
    paths = []
    for i in range(2):
        p = tmp_path / f"complete_0{i}.jsonl"
        rows = [{"timestamp_ms": 40 * k, "arm_elevation_deg": 10.0 + k, "usable": int(k != 1)}
                for k in range(3)]
        p.write_text("\n".join(json.dumps(r) for r in rows))
        paths.append(p)
    f = practice_frames(paths)
    assert list(f["t_capture_ms"]) == [0, 40, 80, 1080, 1120, 1160]
    assert np.isnan(f["theta_v_deg"].iloc[1]) and f["theta_v_deg"].iloc[2] == 12.0


def test_unrecorded_session_is_null_and_warns(manifest, frames, tmp_path):
    m = {**manifest, "session_id": None, "notes": "practice metadata has no session ID"}
    path = tmp_path / "m.json"
    save_manifest(m, path)
    back = load_manifest(path)
    assert back["session_id"] is None
    _, _, warnings = check_manifest(back, frames, "theta_v_deg")
    assert len(warnings) == 1 and "not recorded" in warnings[0] and "no session ID" in warnings[0]
