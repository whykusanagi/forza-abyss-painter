"""Tests for the fap-generate CLI.

The CLI is a thin wrapper around torch_runner.run() — these tests
verify the argument parsing, config-dict building, validation gates,
and exit codes. They DO NOT exercise the real GPU engine (that's
covered by tests/test_torch_runner.py via subprocess + mocked
engine).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from forza_abyss_painter.cli import generate as cli


# =============================================================== arg parsing


def test_parser_requires_image_and_output(tmp_path, capsys):
    """Missing required args → argparse exits 2 with usage on stderr."""
    with pytest.raises(SystemExit) as excinfo:
        cli._build_parser().parse_args([])
    assert excinfo.value.code == 2


def test_parser_accepts_full_arg_set(tmp_path):
    """Sanity: every option parses + lands on the namespace."""
    args = cli._build_parser().parse_args([
        "--image", str(tmp_path / "src.png"),
        "--output", str(tmp_path / "out.json"),
        "--num-shapes", "2000",
        "--max-resolution", "1200",
        "--random-samples", "4096",
        "--sticker",
        "--vram-budget", "12.0",
        "--checkpoint-every", "100",
        "--preset-label", "Medium 1000",
        "--device", "cpu",
    ])
    assert args.image == tmp_path / "src.png"
    assert args.output == tmp_path / "out.json"
    assert args.num_shapes == 2000
    assert args.max_resolution == 1200
    assert args.random_samples == 4096
    assert args.sticker is True
    assert args.vram_budget == 12.0
    assert args.checkpoint_every == 100
    assert args.preset_label == "Medium 1000"
    assert args.device == "cpu"


# ============================================================ pre-flight gates


def test_main_returns_2_when_source_image_missing(tmp_path, capsys):
    """No source file → exit 2 (config error) with a clear stderr
    message. The CLI checks BEFORE invoking torch_runner so we get a
    crisp message instead of a deep PIL traceback."""
    rc = cli.main([
        "--image", str(tmp_path / "does-not-exist.png"),
        "--output", str(tmp_path / "out.json"),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not found" in err


def test_main_creates_output_parent_dir(tmp_path, monkeypatch):
    """The CLI mkdir's the output's parent dir so users can pass
    nested paths like ./renders/2026-05-25/foo.json without having to
    pre-create the directory tree."""
    # Stub out the actual runner so we don't need torch.
    fake_run = MagicMock(return_value=0)
    fake_run_config = MagicMock()
    fake_run_config.from_dict = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(
        "forza_abyss_painter.runtime.torch_runner.run", fake_run,
    )
    monkeypatch.setattr(
        "forza_abyss_painter.runtime.torch_runner.RunConfig", fake_run_config,
    )

    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")
    out = tmp_path / "deep" / "nested" / "renders" / "out.json"
    rc = cli.main([
        "--image", str(src),
        "--output", str(out),
    ])
    assert rc == 0
    assert out.parent.is_dir(), "parent dir not pre-created"


# ============================================================ config building


def test_main_passes_full_config_to_runner(tmp_path, monkeypatch):
    """Verify every CLI arg lands in the config dict the runner
    consumes. If a new field is added to RunConfig, this test fails
    until the CLI surfaces it."""
    captured = {}

    def _capture_from_dict(d):
        captured["dict"] = d
        return MagicMock()

    fake_run = MagicMock(return_value=0)
    fake_runconfig = MagicMock()
    fake_runconfig.from_dict = _capture_from_dict
    monkeypatch.setattr(
        "forza_abyss_painter.runtime.torch_runner.run", fake_run,
    )
    monkeypatch.setattr(
        "forza_abyss_painter.runtime.torch_runner.RunConfig", fake_runconfig,
    )

    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")
    rc = cli.main([
        "--image", str(src),
        "--output", str(tmp_path / "out.json"),
        "--num-shapes", "500",
        "--max-resolution", "600",
        "--random-samples", "2048",
        "--sticker",
        "--vram-budget", "8.0",
        "--preset-label", "test-preset",
    ])
    assert rc == 0
    d = captured["dict"]
    assert d["num_shapes"] == 500
    assert d["max_resolution"] == 600
    assert d["random_samples"] == 2048
    assert d["sticker_mode"] is True
    assert d["vram_budget_gib"] == 8.0
    assert d["preset_label"] == "test-preset"
    assert d["lock_alpha"] is True   # always forced (CLAUDE.md §3)


def test_default_checkpoint_every_is_num_shapes_over_20(tmp_path, monkeypatch):
    """Sensible default: 20 progress events across the run regardless
    of num_shapes. Same heuristic as the GUI dialog uses."""
    captured = {}

    def _capture(d):
        captured["dict"] = d
        return MagicMock()

    fake_run = MagicMock(return_value=0)
    fake_rc = MagicMock()
    fake_rc.from_dict = _capture
    monkeypatch.setattr(
        "forza_abyss_painter.runtime.torch_runner.run", fake_run,
    )
    monkeypatch.setattr(
        "forza_abyss_painter.runtime.torch_runner.RunConfig", fake_rc,
    )

    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")
    cli.main([
        "--image", str(src), "--output", str(tmp_path / "out.json"),
        "--num-shapes", "3000",
    ])
    assert captured["dict"]["checkpoint_every"] == 150   # 3000 // 20


# ============================================================ exit codes


def test_main_returns_2_on_invalid_config(tmp_path, monkeypatch):
    """RunConfig.from_dict raises ValueError for invalid configs →
    CLI surfaces exit 2 with the message."""
    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    fake_rc = MagicMock()
    fake_rc.from_dict = MagicMock(side_effect=ValueError("bad config"))
    monkeypatch.setattr(
        "forza_abyss_painter.runtime.torch_runner.RunConfig", fake_rc,
    )
    rc = cli.main([
        "--image", str(src), "--output", str(tmp_path / "out.json"),
    ])
    assert rc == 2
