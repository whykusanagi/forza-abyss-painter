"""Diagnostics-bundle export: zip the GPU logs + runtime marker + system
info into a single ForzaAbyssPainter-diag-{ts}.zip the tester can email
or upload for post-mortem.

Surfaced as:
  - `fap-diag` CLI subcommand (set in pyproject.toml [project.scripts])
  - Help → Save diagnostics zip… menu item (wired in main_window.py)

The zip contents:
  - logs/                       — every file in %LOCALAPPDATA%/.../logs/
  - runtime/installed.json      — the marker if present (capture state)
  - system_info.json            — OS, Python, available CUDA driver
  - README.md                   — recipient-facing explanation

Why: when a Windows tester says "the install dialog showed an error" we
need the on-disk artifacts to know WHICH error, WHERE, WHEN. Without
this bundle the tester has to paste-screenshot the modal + manually
zip the logs dir + guess what other context matters. With it: one
click + one upload.

## CLI

  fap-diag [-o OUTPUT] [--include-runtime-dir]

Defaults to writing the zip next to the user's home directory with a
timestamped name. --include-runtime-dir adds the embedded Python tree
itself (~4 GiB) for cases where we need to inspect the install
artifacts — normally excluded because the bundle is meant to be
emailable.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


_BUNDLE_README = """\
# Forza Abyss Painter — Diagnostics Bundle

Generated: {ts}

This zip captures the state of the GPU-pipeline runtime + recent
session logs so the maintainer can post-mortem a failure without
needing access to your machine.

## What's inside

- `logs/` — Structured JSON-lines logs from recent GPU sessions.
  Each file is one session (install attempt, generation run, etc).
  Format: one JSON object per line. Field `kind` identifies the
  event type; `stage` names the phase that fired it; `elapsed_s`
  is seconds since session start.

- `runtime/installed.json` — The runtime install marker (if a runtime
  exists). Records the torch version, CUDA availability, GPU device
  name, and install timestamp.

- `system_info.json` — Captured at bundle creation time: OS version,
  Python version, available CUDA driver (best-effort via nvidia-smi).

## What's NOT inside

By design, the bundle does NOT include:
  - The embedded Python interpreter / installed torch wheels (~4 GiB)
  - Any source images or generated JSON files (user content)
  - System-wide environment variables (may contain secrets)

Pass `--include-runtime-dir` to fap-diag if the maintainer specifically
asks for the runtime tree (rare; only relevant when the install
itself is suspect).

## How to share

Email the zip, drop it into the project's GitHub issue tracker as an
attachment, or upload to the Discord linked in the README. Don't paste
the contents inline — the logs include absolute paths that may identify
your username.
"""


def _system_info() -> dict:
    """Capture machine state relevant to GPU runtime diagnosis. Best-
    effort: each probe wraps in try/except so a single failure (e.g.,
    nvidia-smi missing) doesn't break the whole bundle."""
    info = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": sys.platform,
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
    }
    # Best-effort nvidia-smi probe. On Windows this is the canonical
    # way to check CUDA driver version; on macOS/Linux it may not exist.
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if proc.returncode == 0:
            info["nvidia_smi"] = proc.stdout.strip()
        else:
            info["nvidia_smi_unavailable"] = (
                f"rc={proc.returncode}, stderr={proc.stderr.strip()[:200]}"
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        info["nvidia_smi_unavailable"] = f"{type(exc).__name__}: {exc}"
    return info


def build_bundle(
    output_zip: Path,
    *,
    include_runtime_dir: bool = False,
) -> Path:
    """Build the diagnostics zip at `output_zip`. Returns the path on
    success; raises OSError on filesystem/zip failures.

    Idempotent: writes a fresh zip on every call (overwrites if the
    target already exists). The output filename is timestamped by the
    caller so duplicate runs land at different paths by default.
    """
    from forza_abyss_painter.runtime.gpu_logger import logs_root
    from forza_abyss_painter.runtime.torch_installer import (
        runtime_marker, runtime_root,
    )

    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with zipfile.ZipFile(output_zip, mode="w",
                         compression=zipfile.ZIP_DEFLATED) as zf:
        # README first — opens to context immediately when the recipient
        # extracts. Markdown is plain-text-friendly even without a viewer.
        zf.writestr("README.md", _BUNDLE_README.format(ts=ts))

        # Logs directory — every session file.
        log_dir = logs_root()
        if log_dir.exists():
            for log_file in sorted(log_dir.glob("*.log")):
                zf.write(log_file, arcname=f"logs/{log_file.name}")

        # Install marker if present.
        marker = runtime_marker()
        if marker.exists():
            zf.write(marker, arcname="runtime/installed.json")

        # System info captured at bundle time.
        zf.writestr(
            "system_info.json",
            json.dumps(_system_info(), indent=2),
        )

        # Optional: the whole embedded runtime dir. Excluded by default
        # (4 GiB email blockers help nobody) — only included when the
        # maintainer specifically needs to inspect install artifacts.
        if include_runtime_dir:
            rt = runtime_root()
            if rt.exists():
                for f in rt.rglob("*"):
                    if f.is_file():
                        # Skip log files (already added above) +
                        # marker (also added).
                        if f.parent == log_dir or f == marker:
                            continue
                        rel = f.relative_to(rt)
                        zf.write(f, arcname=f"runtime/{rel}")
    return output_zip


def _default_output_path() -> Path:
    """Default zip location: user's home directory with a timestamped
    name. Easy for testers to find + email."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    return Path.home() / f"ForzaAbyssPainter-diag-{ts}.zip"


def main(argv: list[str] | None = None) -> int:
    """fap-diag CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="fap-diag",
        description="Bundle Forza Abyss Painter GPU logs + runtime state "
                    "+ system info into a single zip for support / "
                    "post-mortem analysis.",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        help="output zip path (default: ~/ForzaAbyssPainter-diag-<ts>.zip)",
    )
    parser.add_argument(
        "--include-runtime-dir", action="store_true",
        help="include the embedded Python + torch wheels (~4 GiB; rarely "
             "needed — only when install artifacts themselves are suspect)",
    )
    args = parser.parse_args(argv)

    output = args.output or _default_output_path()
    try:
        result = build_bundle(output,
                              include_runtime_dir=args.include_runtime_dir)
    except OSError as exc:
        print(f"fap-diag: failed to build bundle: {exc}", file=sys.stderr)
        return 1
    size_mb = result.stat().st_size / (1 << 20)
    print(f"Wrote diagnostics bundle: {result} ({size_mb:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
