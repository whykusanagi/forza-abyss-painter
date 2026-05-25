"""Background worker that runs FH6 injection on a QThread and emits progress + colored status signals.

Severity codes for the `status` signal:
  "info"    — neutral (use default text color)
  "success" — green (operation completed OK)
  "warning" — yellow (completed but with caveats)
  "error"   — red (operation failed)

Every status line + scan/write milestone is ALSO appended to a per-run log
file under the per-user log directory (see forza_abyss_painter.io.log_paths).
Without this, when fast-mode missed in the field, the diagnostic message
(eg `[fast-locate] miss: …`) lived only in the QLabel and vanished when the
dialog closed. The log path is emitted as the first status line so the user
sees where to find it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from forza_abyss_painter.io.log_paths import open_inject_log


class InjectionWorker(QObject):
    scan_progress = Signal(int, int, int)   # scanned_regions, total_regions, hits_so_far
    write_progress = Signal(int, int)       # written_shapes, total_shapes
    status = Signal(str, str)               # message, severity ("info"|"success"|"warning"|"error")
    log_path = Signal(str)                  # absolute path to this run's log file (emitted once at start)
    done = Signal()

    def __init__(self, json_path: Path, profile_key: str = "fh6",
                 template_size: int | None = None) -> None:
        super().__init__()
        self.json_path = Path(json_path)
        self.profile_key = profile_key
        # User-picked FH6 template size from the pre-inject dialog. None = Auto
        # (worker walks the standard sizes list, current default behavior).
        # Specific int = constrain heap scan to that exact size — much faster
        # when the user knows what they loaded.
        self.template_size = template_size
        # File handle for the per-run log, opened in `run()`. None until the
        # worker starts so the dialog can construct us cheaply at click time.
        self._log_fh = None
        self._log_path: Path | None = None
        # Last scan-progress sample we logged. Logging every region update
        # would balloon the log to thousands of lines on a large scan; we
        # downsample to a printed line every LOG_PROGRESS_EVERY samples so
        # the file stays scannable. Set to 0 forces the first sample.
        self._last_logged_scan = -1

    LOG_PROGRESS_EVERY = 50    # 1 line per 50 region samples (~50 lines per scan)
    LOG_WRITE_EVERY = 250      # 1 line per 250 shapes written (~12 lines on 3k inject)

    def _log_line(self, severity: str, message: str) -> None:
        """Append a single timestamped line to the persistent log. No-op if
        the log handle is closed (eg early failure before run() opened it)."""
        if self._log_fh is None:
            return
        try:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            self._log_fh.write(f"{ts} [{severity}] {message}\n")
        except (OSError, ValueError):
            # Disk full, handle already closed, etc — don't let logging take
            # down an in-flight injection.
            pass

    def _emit_status(self, message: str, severity: str = "info") -> None:
        """Tee a status update to both the Qt signal (GUI dialog) and the log
        file. ALL status updates should go through this — calling
        `self.status.emit(...)` directly bypasses the log."""
        self._log_line(severity, message)
        self.status.emit(message, severity)

    def run(self) -> None:
        from forza_abyss_painter.inject import FH6Injector, patterns_are_populated
        from forza_abyss_painter.inject.game_profiles import get_profile, default_profile
        from forza_abyss_painter.io.exporter import load_json

        # Open the per-run log FIRST so every subsequent status line gets
        # captured. Failure to open the log file should not abort the inject —
        # we still try Qt-signal-only as a fallback.
        try:
            self._log_path, self._log_fh = open_inject_log()
            self.log_path.emit(str(self._log_path))
            self._log_line("info", f"=== Forza Abyss Painter inject log ===")
            self._log_line("info", f"json={self.json_path}")
            self._log_line("info", f"profile_key={self.profile_key}")
        except Exception as exc:
            # Logging unavailable — continue without it. The Qt signals still
            # surface everything to the GUI, just not persisted.
            self._log_fh = None
            self._log_path = None
            self.status.emit(
                f"Could not open log file ({type(exc).__name__}: {exc}). "
                f"Continuing without persistent logging.",
                "warning",
            )

        if not patterns_are_populated():
            self._emit_status("Patterns file not populated. Re-derive via discovery workflow.", "error")
            self._close_log()
            self.done.emit()
            return

        try:
            profile = get_profile(self.profile_key)
        except ValueError:
            profile = default_profile()

        try:
            doc = load_json(str(self.json_path))
            shapes = doc.materialize_shapes()
        except Exception as exc:
            self._emit_status(f"Could not load JSON: {type(exc).__name__}: {exc}", "error")
            self._close_log()
            self.done.emit()
            return

        n_shapes = len(shapes)
        self._emit_status(f"Loaded {n_shapes} shapes from {self.json_path.name}.", "info")

        if profile.beta:
            self._emit_status(
                f"⚠ BETA target: {profile.label}. {profile.beta_note}",
                "warning",
            )

        inj = FH6Injector(profile=profile)
        try:
            # Upfront expectation-setting so users understand both the workflow
            # they should follow AND why a re-injection scan may take longer.
            self._emit_status(
                "Starting injection. For the fastest scan time, load a fresh, "
                "unmodified sphere-template vinyl group of the matching layer count "
                "before injecting. Re-injecting onto an already-painted template "
                "still works but the locator falls back to a slower memory scan "
                "(typically an extra 2–5 minutes on a large game).",
                "info",
            )

            # ---- Pre-locator process search + attach logging.
            # Painter-parity: surface the PID/process-name resolution explicitly
            # so a failed attach tells us WHICH name we tried, what came back,
            # and what the OS handle is. Without this, "Attaching to FH6..."
            # followed by silence is indistinguishable from "no process found"
            # vs "OpenProcess returned NULL" vs "we got the handle but something
            # else broke immediately after".
            from forza_abyss_painter.inject.win_process import find_process_id
            self._log_line("trace", f"process search: candidates={list(profile.process_names)}")
            found_pid: int | None = None
            found_name: str | None = None
            for name in profile.process_names:
                p = find_process_id(name)
                self._log_line("trace",
                    f"  find_process_id({name!r}) -> {p if p is not None else 'None'}")
                if p is not None:
                    found_pid, found_name = p, name
                    break
            if found_pid is None:
                self._emit_status(
                    f"{profile.label} is not running (looked for: "
                    f"{', '.join(profile.process_names)}). Start the game and try again.",
                    "error",
                )
                self._close_log()
                self.done.emit()
                return
            self._emit_status(
                f"Found {profile.label} as {found_name!r} (PID {found_pid}). Attaching…",
                "info",
            )

            inj.pid = found_pid    # pre-resolve so inj.attach() doesn't re-search
            inj.attach()
            self._log_line("trace",
                f"attach OK: pid={found_pid} ProcessHandle opened (PROCESS_VM_READ | VM_WRITE | QUERY_INFORMATION)")

            if profile.beta:
                self._emit_status(
                    f"Attached. Scanning memory for the {n_shapes}-layer sphere-template "
                    f"LiveryGroup (BETA fallback to RTTI will only run if sphere scan finds nothing)…",
                    "info",
                )
            else:
                self._emit_status(
                    f"Attached. Scanning memory for the {n_shapes}-layer LiveryGroup template…",
                    "info",
                )
            # Kick the dialog out of "Preparing" immediately so user sees activity
            # even before the worker emits real region progress.
            self.scan_progress.emit(0, 1, 0)
            # Callback the injector uses to tell us about phase transitions.
            # Routes the two channels emitted by check_inject_readiness:
            #   [trace] xxx   → log file only (verbose per-operation detail
            #                   would overwrite the dialog QLabel every
            #                   ~30ms). User reads it after the fact.
            #   [fast-locate] / [readiness] / etc → dialog AND log (summary
            #                   per gate, what the user actually needs to
            #                   see live).
            def _on_phase_status(msg: str) -> None:
                if msg.startswith("[trace] "):
                    # Strip the prefix in the log file too — readability.
                    self._log_line("trace", msg[len("[trace] "):])
                else:
                    self._emit_status(msg, "warning")
            # Pass n_shapes as preferred layer_count so we try the matching
            # template first. template_size (from the pre-inject picker) lets
            # the user pin a specific size — when set, the heap-scan tries
            # ONLY that size instead of walking common-sizes-≥-n_shapes.
            if self.template_size is not None:
                self._emit_status(
                    f"Using user-selected template size: {self.template_size} "
                    f"spheres (heap scan will skip other sizes).",
                    "info",
                )
            handle = inj.find_active_vinyl_group(
                progress_cb=self._on_scan_progress,
                layer_count=n_shapes,
                template_size=self.template_size,
                status_cb=_on_phase_status,
            )
            slots = handle.layer_count
            if n_shapes > slots:
                self._emit_status(
                    f"Template has {slots} shape slots but JSON has {n_shapes}. "
                    f"Load a larger template (e.g., {n_shapes}-sphere vinyl group) and re-inject.",
                    "warning",
                )
                self._close_log()
                self.done.emit()
                return
            self._emit_status(f"Found {slots} shape slots. Writing {n_shapes} shapes...", "info")
            # Pass image_size so the injector can center coords + invert Y
            img_w, img_h = doc.image_size if doc.image_size else (0, 0)
            image_size = (img_w, img_h) if img_w > 0 and img_h > 0 else None
            result = inj.inject(
                shapes, handle, progress_cb=self._on_write_progress,
                image_size=image_size, coord_scale=1.0,
            )
            if result.success:
                self._emit_status(
                    f"Injected {result.shapes_written} shapes successfully. {result.message}",
                    "success",
                )
            else:
                self._emit_status(f"Injection failed: {result.message}", "error")
        except Exception as exc:
            self._emit_status(f"Injection error: {type(exc).__name__}: {exc}", "error")
        finally:
            try:
                inj.detach()
            except Exception:
                pass
            self._close_log()
            self.done.emit()

    def _close_log(self) -> None:
        if self._log_fh is not None:
            try:
                self._log_line("info", "=== log end ===")
                self._log_fh.close()
            except OSError:
                pass
            self._log_fh = None

    def _on_scan_progress(self, scanned: int, total: int, hits: int) -> None:
        # Downsample logging — the GUI gets every signal, the log gets one
        # line per LOG_PROGRESS_EVERY samples so we can see the scan finished
        # (or where it was when killed) without flooding the file.
        if scanned - self._last_logged_scan >= self.LOG_PROGRESS_EVERY or scanned == total:
            self._log_line("info", f"scan progress: {scanned}/{total} regions, {hits} candidates")
            self._last_logged_scan = scanned
        self.scan_progress.emit(scanned, total, hits)

    def _on_write_progress(self, written: int, total: int) -> None:
        if written == 0 or written % self.LOG_WRITE_EVERY == 0 or written == total:
            self._log_line("info", f"write progress: {written}/{total} shapes")
        self.write_progress.emit(written, total)
