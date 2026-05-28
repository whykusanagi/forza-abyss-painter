"""fap-refresh must refuse to run when source == destination
(invocation from embedded Python = self-destruction)."""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest

from forza_abyss_painter.cli import refresh as refresh_mod


def test_refuses_when_source_equals_destination(tmp_path, capsys):
    """When _source_package_dir() resolves to the embedded
    site-packages's forza_abyss_painter dir, the copy would delete
    shapegen. Guard rejects with exit 2."""
    embed_dir = tmp_path / "runtime" / "python311"
    site_pkgs = embed_dir / "Lib" / "site-packages"
    src_pkg = site_pkgs / "forza_abyss_painter"   # SAME as dest
    src_pkg.mkdir(parents=True)
    (src_pkg / "shapegen").mkdir()
    (embed_dir / "python.exe").touch()

    with patch.object(refresh_mod, "_build_parser",
                       wraps=refresh_mod._build_parser):
        with patch("forza_abyss_painter.runtime.torch_installer.embedded_python_dir",
                    return_value=embed_dir), \
             patch("forza_abyss_painter.runtime.torch_installer.embedded_python_exe",
                    return_value=embed_dir / "python.exe"), \
             patch("forza_abyss_painter.runtime.torch_installer._source_package_dir",
                    return_value=src_pkg):
            rc = refresh_mod.main([])
    assert rc == 2, "fap-refresh must reject same-dir source/dest with exit 2"
    captured = capsys.readouterr()
    assert "same directory" in captured.err.lower() or \
           "embedded python" in captured.err.lower(), (
        f"error message should explain the embedded-py invocation cause; "
        f"got stderr={captured.err!r}"
    )


def test_verify_only_unaffected_by_same_dir(tmp_path, capsys):
    """--verify-only just reports state; doesn't copy; should not trip
    the guard."""
    embed_dir = tmp_path / "runtime" / "python311"
    site_pkgs = embed_dir / "Lib" / "site-packages"
    src_pkg = site_pkgs / "forza_abyss_painter"   # same as dest
    src_pkg.mkdir(parents=True)
    for sub in ("shapegen", "io", "runtime", "cli"):
        (src_pkg / sub).mkdir()
    (embed_dir / "python.exe").touch()

    with patch("forza_abyss_painter.runtime.torch_installer.embedded_python_dir",
                return_value=embed_dir), \
         patch("forza_abyss_painter.runtime.torch_installer.embedded_python_exe",
                return_value=embed_dir / "python.exe"), \
         patch("forza_abyss_painter.runtime.torch_installer._source_package_dir",
                return_value=src_pkg), \
         patch("forza_abyss_painter.runtime.torch_installer.installed_runtime_info",
                return_value=None):
        rc = refresh_mod.main(["--verify-only"])
    assert rc == 0, "--verify-only must NOT be blocked by the same-dir guard"


def test_dry_run_unaffected_by_same_dir(tmp_path):
    """--dry-run lists what WOULD be copied; doesn't actually copy.
    But it also doesn't help the user; the guard's job is to PREVENT
    the destructive copy. Spec choice: keep --dry-run unblocked (it's
    informational and harmless)."""
    embed_dir = tmp_path / "runtime" / "python311"
    site_pkgs = embed_dir / "Lib" / "site-packages"
    src_pkg = site_pkgs / "forza_abyss_painter"
    src_pkg.mkdir(parents=True)
    for sub in ("shapegen", "io", "runtime", "cli"):
        (src_pkg / sub).mkdir()
    (embed_dir / "python.exe").touch()

    with patch("forza_abyss_painter.runtime.torch_installer.embedded_python_dir",
                return_value=embed_dir), \
         patch("forza_abyss_painter.runtime.torch_installer.embedded_python_exe",
                return_value=embed_dir / "python.exe"), \
         patch("forza_abyss_painter.runtime.torch_installer._source_package_dir",
                return_value=src_pkg):
        rc = refresh_mod.main(["--dry-run"])
    assert rc == 0


def test_different_dirs_pass(tmp_path):
    """Normal happy path: source (PyInstaller _MEIPASS) differs from
    destination (embedded site-packages). Guard does not fire."""
    embed_dir = tmp_path / "runtime" / "python311"
    site_pkgs = embed_dir / "Lib" / "site-packages"
    dest_pkg = site_pkgs / "forza_abyss_painter"
    dest_pkg.mkdir(parents=True)
    src_pkg = tmp_path / "mei" / "forza_abyss_painter"
    src_pkg.mkdir(parents=True)
    for sub in ("shapegen", "io", "runtime", "cli"):
        (src_pkg / sub).mkdir()
        (src_pkg / sub / "__init__.py").write_text("")
    # source has __init__.py at the top too
    (src_pkg / "__init__.py").write_text("")
    (embed_dir / "python.exe").touch()

    with patch("forza_abyss_painter.runtime.torch_installer.embedded_python_dir",
                return_value=embed_dir), \
         patch("forza_abyss_painter.runtime.torch_installer.embedded_python_exe",
                return_value=embed_dir / "python.exe"), \
         patch("forza_abyss_painter.runtime.torch_installer._source_package_dir",
                return_value=src_pkg):
        rc = refresh_mod.main(["--dry-run"])
    assert rc == 0
