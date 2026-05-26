"""`fap-refresh` — re-copy the bundled forza_abyss_painter subpackages
into the embedded Python runtime's site-packages WITHOUT re-downloading
torch.

## Why this exists

Cursor's QUASAR Run 4 surfaced a real-world workflow problem: rebuilding
the EXE doesn't update what's inside the runtime install. The runtime
holds its own snapshot of `forza_abyss_painter/{shapegen,io,runtime,cli}`
in `%LOCALAPPDATA%/ForzaAbyssPainter/runtime/python311/Lib/site-packages/`,
copied there ONCE at install time. A fresh EXE with bug fixes ships
NEW code, but the embedded snapshot keeps the OLD code — so users see
"why didn't my upgrade fix anything?".

Recovery options before this CLI:
  1. Delete the runtime dir + re-run Install GPU runtime (full ~3 GiB
     re-download of torch).
  2. Manually copy files from the EXE's `_MEI*` extracted folder to
     the embedded site-packages (Cursor did this on QUASAR to test
     run 4).

This CLI is option 2 automated. Re-copy just the runner packages
(shapegen + io + runtime + cli) — torch + numpy stay in place. Total
runtime: ~5 seconds.

## Usage

    fap-refresh

Or as a python module if entry point isn't on PATH:

    "<EXE bundle>\\python.exe" -m forza_abyss_painter.cli.refresh

The CLI uses the same `_copy_runner_package` helper `install_runtime`
calls — guaranteed identical layout to a fresh install, so the
embedded Python doesn't see drift.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fap-refresh",
        description=(
            "Re-copy bundled forza_abyss_painter subpackages "
            "(shapegen/io/runtime/cli) into the GPU runtime's embedded "
            "site-packages so a freshly-rebuilt EXE's code reaches the "
            "subprocess runner. Skips the multi-GiB torch reinstall — "
            "use 'Tools → Install GPU runtime' for that."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print what would be copied without actually copying",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="report which subpackages are present in the embedded "
             "site-packages + their on-disk timestamp, then exit. "
             "Useful for diagnosing 'did my last refresh actually land?'",
    )
    return parser


def main(argv: "list[str] | None" = None) -> int:
    args = _build_parser().parse_args(argv)

    # Lazy imports: torch_installer pulls a few constants but no torch.
    try:
        from forza_abyss_painter.runtime.torch_installer import (
            _copy_runner_package, _source_package_dir,
            RUNNER_REQUIRED_SUBPACKAGES,
            embedded_python_dir, embedded_python_exe,
            installed_runtime_info,
        )
    except ImportError as exc:
        print(f"fap-refresh: failed to import torch_installer: {exc}",
              file=sys.stderr)
        return 3

    embed_dir = embedded_python_dir()
    py_exe = embedded_python_exe()
    site_pkgs = embed_dir / "Lib" / "site-packages"

    # Pre-flight: runtime must actually exist.
    if not py_exe.exists():
        print(
            f"fap-refresh: embedded Python not found at {py_exe}.\n"
            f"  Run 'Tools → Install GPU runtime' from the EXE first.",
            file=sys.stderr,
        )
        return 2
    if not site_pkgs.exists():
        print(
            f"fap-refresh: site-packages missing at {site_pkgs}. "
            f"Runtime install is incomplete — re-install required.",
            file=sys.stderr,
        )
        return 2

    src_pkg = _source_package_dir()
    print(f"Source package: {src_pkg}")
    print(f"Embedded site-packages: {site_pkgs}")

    # Verify-only: report state + exit. Useful when triaging 'did my
    # last refresh actually land?'.
    if args.verify_only:
        for sub in RUNNER_REQUIRED_SUBPACKAGES:
            dst = site_pkgs / "forza_abyss_painter" / sub
            if dst.exists():
                try:
                    mtime = dst.stat().st_mtime
                    from datetime import datetime, timezone
                    ts = datetime.fromtimestamp(mtime, tz=timezone.utc) \
                        .isoformat(timespec="seconds")
                    print(f"  ✓ {sub}: present, mtime={ts}")
                except OSError as exc:
                    print(f"  ⚠ {sub}: stat failed: {exc}")
            else:
                print(f"  ✗ {sub}: MISSING")
        info = installed_runtime_info()
        if info:
            print(f"\nMarker: torch={info.torch_version}, "
                  f"cuda_available={info.cuda_available}, "
                  f"device={info.cuda_device_name}")
        else:
            print("\nMarker: missing")
        return 0

    # Dry-run: list what would be copied.
    if args.dry_run:
        print("\n[DRY RUN] would copy:")
        for sub in RUNNER_REQUIRED_SUBPACKAGES:
            src = src_pkg / sub
            if src.is_dir():
                size_mb = sum(
                    f.stat().st_size for f in src.rglob("*") if f.is_file()
                ) / (1 << 20)
                print(f"  {src} → {site_pkgs}/forza_abyss_painter/{sub} "
                      f"(~{size_mb:.1f} MiB)")
        return 0

    # Real copy.
    print(f"\nRefreshing {len(RUNNER_REQUIRED_SUBPACKAGES)} subpackages…")
    try:
        _copy_runner_package(src_pkg, site_pkgs)
    except Exception as exc:
        print(f"fap-refresh: copy failed: {exc}", file=sys.stderr)
        return 1
    for sub in RUNNER_REQUIRED_SUBPACKAGES:
        dst = site_pkgs / "forza_abyss_painter" / sub
        marker = "✓" if dst.is_dir() else "✗"
        print(f"  {marker} {sub}")
    print(
        f"\nDone. The embedded runtime now uses the bundled code. "
        f"Re-launch the EXE / re-run fap-generate to pick up the new "
        f"behavior."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
