"""snapshot_path_for builds the <output_stem>_<count>.json path.

Used by both the CPU worker (existing inline construction) and the GPU
runner (new code). Pure path math — no I/O.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from forza_abyss_painter.io.snapshots import snapshot_path_for


def test_basic(tmp_path):
    out = tmp_path / "ziz_dnace.json"
    assert snapshot_path_for(out, 2900) == tmp_path / "ziz_dnace_2900.json"


def test_preserves_parent_dir(tmp_path):
    out = tmp_path / "subdir" / "ziz_dnace.json"
    assert snapshot_path_for(out, 100) == tmp_path / "subdir" / "ziz_dnace_100.json"


def test_stem_with_dots(tmp_path):
    out = tmp_path / "ziz.dance.v2.json"
    assert snapshot_path_for(out, 500) == tmp_path / "ziz.dance.v2_500.json"


def test_no_extension(tmp_path):
    out = tmp_path / "ziz"   # caller didn't add .json
    assert snapshot_path_for(out, 100) == tmp_path / "ziz_100.json"


def test_count_zero(tmp_path):
    out = tmp_path / "x.json"
    assert snapshot_path_for(out, 0) == tmp_path / "x_0.json"


def test_accepts_str_path():
    result = snapshot_path_for("/tmp/x.json", 100)
    assert result == Path("/tmp/x_100.json")
