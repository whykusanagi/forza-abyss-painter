"""Tests for the fap-validate CLI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from forza_abyss_painter.cli import validate as cli


def _good_doc() -> dict:
    return {
        "format": "fd6.shapes",
        "version": 1,
        "image_size": [100, 100],
        "shape_count": 1,
        "shapes": [
            {"type": "rotated_ellipse",
             "x": 50, "y": 50, "rx": 10, "ry": 10, "angle": 0,
             "color": [255, 0, 0, 255]},
        ],
    }


@pytest.fixture
def good_json_file(tmp_path):
    p = tmp_path / "good.json"
    p.write_text(json.dumps(_good_doc()), encoding="utf-8")
    return p


@pytest.fixture
def warning_json_file(tmp_path):
    """A doc with one warning (shape_count mismatch) but no errors."""
    d = _good_doc()
    d["shape_count"] = 99
    p = tmp_path / "warn.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


@pytest.fixture
def error_json_file(tmp_path):
    """A doc with an actual error (missing format)."""
    d = _good_doc()
    del d["format"]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def test_exit_0_on_clean_document(good_json_file, capsys):
    rc = cli.main([str(good_json_file)])
    assert rc == 0
    assert "no findings" in capsys.readouterr().out


def test_exit_2_on_error_document(error_json_file, capsys):
    """Errors are non-recoverable — exit 2 distinguishes from warnings."""
    rc = cli.main([str(error_json_file)])
    assert rc == 2
    assert "missing_format" in capsys.readouterr().out


def test_warnings_exit_0_by_default(warning_json_file):
    """Warnings let the document load — default behavior matches the
    GUI, which accepts warning-level JSONs."""
    rc = cli.main([str(warning_json_file)])
    assert rc == 0


def test_warnings_exit_1_in_strict_mode(warning_json_file):
    """--strict is the CI-friendly mode."""
    rc = cli.main([str(warning_json_file), "--strict"])
    assert rc == 1


def test_missing_file_returns_1(tmp_path, capsys):
    """File-not-found is a load failure, distinct from validation
    errors (exit 1 vs. 2)."""
    rc = cli.main([str(tmp_path / "nope.json")])
    assert rc == 1
    assert "not found" in capsys.readouterr().err


def test_malformed_json_returns_1(tmp_path, capsys):
    p = tmp_path / "garbage.json"
    p.write_text("this is not json", encoding="utf-8")
    rc = cli.main([str(p)])
    assert rc == 1
    assert "JSON parse error" in capsys.readouterr().err


def test_quiet_suppresses_output(error_json_file, capsys):
    """--quiet is for scripts that only want the exit code."""
    rc = cli.main([str(error_json_file), "--quiet"])
    assert rc == 2
    assert capsys.readouterr().out == ""


def test_json_output_is_parseable(error_json_file, capsys):
    """--json mode must emit parseable JSON to stdout. Used by CI."""
    rc = cli.main([str(error_json_file), "--json"])
    assert rc == 2
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["path"] == str(error_json_file)
    assert parsed["summary"]["errors"] >= 1
    assert any(i["code"] == "missing_format" for i in parsed["issues"])


def test_json_output_includes_paths_for_shape_issues(tmp_path, capsys):
    """The `path` field on each issue is what the GUI / CI uses to
    point at the bad shape. Must round-trip through the JSON output."""
    d = _good_doc()
    d["shapes"][0]["rx"] = -5     # error: non-positive extent
    d["shape_count"] = 1
    p = tmp_path / "bad_shape.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    cli.main([str(p), "--json"])
    out = capsys.readouterr().out
    parsed = json.loads(out)
    paths = {i["path"] for i in parsed["issues"]}
    assert "shapes[0].rx" in paths
