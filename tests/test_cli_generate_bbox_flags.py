"""--bbox-local / --full-canvas CLI flags route to RunConfig.bbox_local."""
from __future__ import annotations

from pathlib import Path

import pytest

from forza_abyss_painter.cli.generate import _build_parser


def test_default_is_bbox_local_true():
    # No flag → CLI must yield bbox_local=True (matches runner default
    # and CLAUDE.md §1/§8a production stance).
    args = _build_parser().parse_args([
        "--image", "x.png", "--output", "out.json",
    ])
    assert args.bbox_local is True


def test_full_canvas_flag_sets_false():
    args = _build_parser().parse_args([
        "--image", "x.png", "--output", "out.json", "--full-canvas",
    ])
    assert args.bbox_local is False


def test_bbox_local_flag_sets_true():
    args = _build_parser().parse_args([
        "--image", "x.png", "--output", "out.json", "--bbox-local",
    ])
    assert args.bbox_local is True


def test_both_flags_mutually_exclusive():
    # argparse's mutually_exclusive_group raises SystemExit on conflict.
    with pytest.raises(SystemExit):
        _build_parser().parse_args([
            "--image", "x.png", "--output", "out.json",
            "--bbox-local", "--full-canvas",
        ])


def test_cli_config_dict_includes_bbox_local(tmp_path):
    """End-to-end check: main() builds a config dict with the bbox_local
    field set correctly for diagnostic runs."""
    from forza_abyss_painter.cli import generate as gen
    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    out = tmp_path / "out.json"

    # Patch the runner so we capture the config dict without spawning torch.
    captured = {}

    class _StopRun(Exception):
        pass

    class _FakeRunConfig:
        @classmethod
        def from_dict(cls, d):
            captured["dict"] = d
            raise _StopRun()   # short-circuit before run()

    orig = gen.__dict__.get  # we'll monkeypatch instead
    import sys as _sys
    import types as _types
    fake_module = _types.ModuleType("forza_abyss_painter.runtime.torch_runner")
    fake_module.RunConfig = _FakeRunConfig
    fake_module.run = lambda cfg: 0
    _sys.modules["forza_abyss_painter.runtime.torch_runner"] = fake_module
    try:
        with pytest.raises(_StopRun):
            gen.main([
                "--image", str(img), "--output", str(out),
                "--full-canvas",
            ])
        assert captured["dict"]["bbox_local"] is False
    finally:
        del _sys.modules["forza_abyss_painter.runtime.torch_runner"]
