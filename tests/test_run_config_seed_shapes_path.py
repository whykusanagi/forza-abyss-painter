"""RunConfig.seed_shapes_path: fresh-mode resume points to a snapshot.

Validation:
- str → Path on parse
- Missing file → ValueError
- Set in polish_only mode → ValueError (resume is fresh-only)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from forza_abyss_painter.runtime.torch_runner import RunConfig


def _fresh_dict(tmp_path) -> dict:
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    return {
        "image_path": str(image),
        "output_json_path": str(tmp_path / "out.json"),
        "num_shapes": 100,
        "max_resolution": 360,
        "random_samples": 1024,
    }


def _snapshot(tmp_path) -> Path:
    p = tmp_path / "out_50.json"
    p.write_text(json.dumps({"format": "fd6.shapes", "version": 1, "shapes": []}))
    return p


def test_default_seed_shapes_path_is_none(tmp_path):
    cfg = RunConfig.from_dict(_fresh_dict(tmp_path))
    assert cfg.seed_shapes_path is None


def test_seed_shapes_path_parses(tmp_path):
    snap = _snapshot(tmp_path)
    d = _fresh_dict(tmp_path)
    d["seed_shapes_path"] = str(snap)
    cfg = RunConfig.from_dict(d)
    assert cfg.seed_shapes_path == snap


def test_seed_shapes_path_missing_file_raises(tmp_path):
    d = _fresh_dict(tmp_path)
    d["seed_shapes_path"] = str(tmp_path / "does_not_exist_999.json")
    with pytest.raises(ValueError, match="seed_shapes_path"):
        RunConfig.from_dict(d)


def test_polish_only_rejects_seed_shapes_path(tmp_path):
    snap = _snapshot(tmp_path)
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    shapes = tmp_path / "in.json"
    shapes.write_text(json.dumps({"format": "fd6.shapes", "version": 1, "shapes": []}))
    d = {
        "image_path": str(image),
        "output_json_path": str(tmp_path / "out.json"),
        "mode": "polish_only",
        "input_shapes_path": str(shapes),
        "seed_shapes_path": str(snap),
    }
    with pytest.raises(ValueError, match="seed_shapes_path"):
        RunConfig.from_dict(d)
