"""Tests for the fap-refresh CLI — re-copies bundled subpackages into
the embedded site-packages without redoing the torch install.

Wraps torch_installer._copy_runner_package; these tests verify the CLI
pre-flight gates + verify-only mode + dry-run mode + the actual copy
flow (via fake dirs).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from forza_abyss_painter.cli import refresh as cli
from forza_abyss_painter.runtime import torch_installer as ti


@pytest.fixture
def _isolated_runtime(tmp_path, monkeypatch):
    """Redirect runtime_root() so we don't touch real LOCALAPPDATA."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(ti.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    return ti.runtime_root()


@pytest.fixture
def _stub_source(tmp_path, monkeypatch):
    """Fake on-disk source package layout that _copy_runner_package
    can read from. Mirrors the real package's RUNNER_REQUIRED_SUBPACKAGES."""
    src = tmp_path / "src_pkg"
    src.mkdir()
    (src / "__init__.py").write_text("", encoding="utf-8")
    for sub in ti.RUNNER_REQUIRED_SUBPACKAGES:
        (src / sub).mkdir()
        (src / sub / "__init__.py").write_text(f"# {sub}\n", encoding="utf-8")
    monkeypatch.setattr(ti, "_source_package_dir", lambda: src)
    return src


def _install_minimal_runtime(rt: Path) -> None:
    """Mock a 'runtime is installed' state — python.exe + marker
    exist but no subpackages yet (the state fap-refresh fixes)."""
    embed = ti.embedded_python_dir()
    embed.mkdir(parents=True, exist_ok=True)
    ti.embedded_python_exe().write_text("#!/bin/sh\necho fake",
                                        encoding="utf-8")
    site = embed / "Lib" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    marker = {
        "python_version": "3.11.9",
        "torch_version": "2.7.0+cu128",
        "cuda_available": True,
        "cuda_device_name": "FAKE",
        "installed_at_utc": "2026-05-26T00:00:00Z",
    }
    ti.runtime_marker().write_text(json.dumps(marker), encoding="utf-8")


# =================================================================== pre-flight


def test_main_returns_2_when_runtime_missing(_isolated_runtime, capsys):
    """No runtime install at all → exit 2 with a clear 'install
    first' message. Don't silently no-op (user expects a refresh)."""
    rc = cli.main([])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Install GPU runtime" in err


def test_main_returns_2_when_site_packages_missing(_isolated_runtime, capsys):
    """python.exe exists but site-packages dir doesn't → install was
    aborted mid-extract. Surface that as a re-install need, not a
    refresh."""
    ti.embedded_python_dir().mkdir(parents=True, exist_ok=True)
    ti.embedded_python_exe().write_text("fake", encoding="utf-8")
    # Note: NOT creating site-packages.
    rc = cli.main([])
    assert rc == 2
    err = capsys.readouterr().err
    assert "site-packages missing" in err


# =================================================================== verify-only


def test_verify_only_reports_present_subpackages(
    _isolated_runtime, _stub_source, capsys,
):
    """--verify-only lists each required subpackage's presence + mtime.
    Used for 'did my last refresh actually land?' triage."""
    _install_minimal_runtime(_isolated_runtime)
    # Pre-populate one subpackage so the report shows mixed state.
    site = ti.embedded_python_dir() / "Lib" / "site-packages"
    fap = site / "forza_abyss_painter"
    fap.mkdir(parents=True, exist_ok=True)
    (fap / "shapegen").mkdir()
    (fap / "shapegen" / "__init__.py").write_text("# old", encoding="utf-8")
    # Don't create io / runtime / cli — verify should flag them MISSING.
    rc = cli.main(["--verify-only"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "shapegen: present" in out
    # io / runtime / cli should be marked missing.
    assert any(f"{sub}: MISSING" in out
               for sub in ("io", "runtime", "cli"))
    # Marker info also reports.
    assert "torch=2.7.0+cu128" in out


# =================================================================== dry-run


def test_dry_run_lists_targets_without_copying(
    _isolated_runtime, _stub_source, capsys,
):
    """--dry-run prints what would be copied but doesn't touch the
    embedded site-packages. Lets users sanity-check before the real
    copy."""
    _install_minimal_runtime(_isolated_runtime)
    rc = cli.main(["--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[DRY RUN]" in out
    # Nothing actually copied — embedded forza_abyss_painter dir is absent.
    site = ti.embedded_python_dir() / "Lib" / "site-packages"
    assert not (site / "forza_abyss_painter" / "shapegen").exists(), (
        "dry-run wrote files anyway"
    )


# =================================================================== real copy


def test_real_run_copies_all_required_subpackages(
    _isolated_runtime, _stub_source, capsys,
):
    """Real invocation copies every entry in RUNNER_REQUIRED_SUBPACKAGES
    to the embedded site-packages. Verifies the fix actually lands +
    is idempotent (running twice doesn't error)."""
    _install_minimal_runtime(_isolated_runtime)
    rc = cli.main([])
    assert rc == 0
    site = ti.embedded_python_dir() / "Lib" / "site-packages"
    for sub in ti.RUNNER_REQUIRED_SUBPACKAGES:
        assert (site / "forza_abyss_painter" / sub).is_dir(), (
            f"required subpackage {sub} not copied"
        )
    # Second run = idempotent.
    rc2 = cli.main([])
    assert rc2 == 0
