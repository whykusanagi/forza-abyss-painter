"""Install-prompt + install-progress dialog for the on-demand GPU runtime.

Phase 2 of EXE GPU bundle (task #62) — the UI scaffolding. Phase 3 wires
the actual HTTP download + embedded-Python bootstrap + torch install
(`forza_abyss_painter.runtime.torch_installer.install_runtime`, not yet
implemented). For now the "Install" button reports "Phase 3 not yet
shipped — close this dialog and watch the EXE Releases for a build that
includes the runtime installer."

Two phases inside this single dialog:

  CONFIRM phase (initial state):
    - Explains what gets downloaded (~4 GiB), where (LOCALAPPDATA), why
      (one-time setup so consumer-GPU users can shape-gen in-app without
      needing Colab access)
    - Install / Cancel buttons

  INSTALL phase (after Install clicked):
    - Replaces the confirm text with a progress bar + status line
    - Cancel button still works (would terminate the download mid-stream
      in Phase 3; for now just closes the dialog)
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QMessageBox,
    QProgressBar, QPushButton, QVBoxLayout,
)

from forza_abyss_painter.gui.gpu_install_worker import GpuInstallWorker
from forza_abyss_painter.runtime import torch_installer


class RuntimeInstallDialog(QDialog):
    """Modal dialog: prompt to install the GPU runtime, then show install
    progress. Caller checks `was_installed` after exec() — True if the
    runtime is ready to use, False if the user cancelled or install
    failed.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Install GPU runtime")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.was_installed: bool = False

        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(20, 16, 20, 16)
        self._root.setSpacing(12)

        # Header
        hdr = QLabel("Install local GPU shape-generation runtime")
        hf = QFont(); hf.setBold(True); hf.setPointSize(13)
        hdr.setFont(hf)
        self._root.addWidget(hdr)

        # Confirm-phase content — what the user is agreeing to. The
        # 'DO NOT close ANY windows' callout is the #1 lesson from
        # QUASAR's first install attempt: leaked console + 30%-progress-
        # frozen UI made the tester force-quit mid-pip and we lost the
        # run. Don't bury the warning in subtle text.
        gib = torch_installer.estimated_download_bytes() / (1 << 30)
        self.body = QLabel(
            f"To generate shapes locally on your GPU, Forza Abyss Painter "
            f"needs an isolated PyTorch + CUDA runtime. This is a "
            f"<b>one-time download of ~{gib:.0f} GiB</b> stored in your "
            f"local app data folder (<code>{torch_installer.runtime_root()}</code>) "
            f"— it doesn't affect your system Python or other applications."
            f"<br><br>"
            f"<b style='color: #d94f90;'>Important — first install takes "
            f"5–15 minutes:</b>"
            f"<ul style='margin-top: 4px;'>"
            f"<li><b>Do NOT close any windows</b> while installing — "
            f"closing a window cancels the install mid-download.</li>"
            f"<li>The <b>progress bar may sit at 30%</b> for several "
            f"minutes while torch downloads. That's normal — the bar "
            f"animates so you know it's still working.</li>"
            f"<li>If anything goes wrong, run <b>Tools → Save "
            f"diagnostics zip…</b> and share the file.</li>"
            f"</ul>"
            f"Subsequent runs reuse the cached runtime instantly. You "
            f"can skip this entirely and use the Colab notebooks "
            f"instead — those run on cloud GPUs."
        )
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.RichText)
        self._root.addWidget(self.body)

        # Install-phase content — progress bar + status line, hidden until needed.
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self._root.addWidget(self.progress)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        self.status_label.setVisible(False)
        self._root.addWidget(self.status_label)

        # Buttons — Install (primary) + Cancel.
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)
        self.install_btn = QPushButton("Install", self)
        self.install_btn.setDefault(True)
        self.install_btn.clicked.connect(self._on_install_clicked)
        btn_row.addWidget(self.install_btn)
        self._root.addLayout(btn_row)

    # --------------------------------------------------- install-phase machinery

    def _on_install_clicked(self) -> None:
        """User clicked Install — switch to install-phase UI + spawn the
        GpuInstallWorker on a QThread to actually run install_runtime().
        Progress signals update the progress bar + status label;
        done/error signals route to _on_install_done / _on_install_error
        which set was_installed + close the dialog or surface a modal."""
        self.body.setText(
            "<b>Installing GPU runtime — DO NOT close any windows.</b>"
            "<br><br>"
            "This takes 5–15 minutes on first install (downloading ~3 GB "
            "of PyTorch + CUDA wheels). The dialog will close automatically "
            "when done."
            "<br><br>"
            "<i>If the progress bar sits at 30% for several minutes, that's "
            "the torch download running — the bar animates so you know "
            "it's still going. Wait for it to finish.</i>"
        )
        self.progress.setVisible(True)
        self.status_label.setVisible(True)
        self.install_btn.setEnabled(False)
        # Cancel-during-install isn't safe (torch_installer's HTTP
        # downloads aren't interruptible mid-stream without orphaning
        # temp files). Disable until done; the dialog auto-closes on
        # success or surfaces an error modal on failure.
        self.cancel_btn.setEnabled(False)

        # Elapsed-time ticker: refreshes the status label every 2 sec
        # during the long pip phase so the user has a visible "yes,
        # progress is happening" signal even while the percent is
        # pegged. Started after install-clicked, stopped on done/error.
        from PySide6.QtCore import QTimer
        import time as _time
        self._install_start_t = _time.monotonic()
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(2000)   # every 2 seconds
        self._elapsed_timer.timeout.connect(self._on_elapsed_tick)
        self._elapsed_timer.start()

        # Spawn the worker on a dedicated thread so the GUI stays
        # responsive while torch wheels download. Hold both as instance
        # attrs so the GC doesn't reap them mid-run.
        self._thread = QThread(self)
        self._worker = GpuInstallWorker()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_install_done)
        self._worker.error.connect(self._on_install_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_progress(self, percent: int, status: str) -> None:
        """Called by the install worker on each install_runtime phase
        boundary. percent in [0, 100], status is a human-readable
        label like 'Installing torch 2.4.1 + deps…'.

        Special behavior during the long pip phase (percent stays 30
        until pip returns 50+ seconds later): switch the progress bar
        to indeterminate (animated 'busy') mode so the user sees
        ongoing motion + knows the install isn't frozen. As soon as
        percent moves past 30, restore determinate mode."""
        self.status_label.setText(status)
        self._latest_phase_text = status
        if percent == 30 and "torch" in status.lower():
            # pip download phase — indeterminate busy bar. setRange(0,0)
            # is Qt's idiom for indeterminate progress.
            self.progress.setRange(0, 0)
        else:
            # Normal determinate progress.
            if self.progress.maximum() == 0:
                self.progress.setRange(0, 100)
            self.progress.setValue(max(0, min(100, percent)))

    def _on_elapsed_tick(self) -> None:
        """Tick from the install-phase QTimer (2-second cadence). Appends
        elapsed minutes/seconds to the latest phase text so even while
        the percent is pegged, the user sees the clock moving."""
        import time as _time
        elapsed = int(_time.monotonic() - self._install_start_t)
        mins, secs = divmod(elapsed, 60)
        base = getattr(self, "_latest_phase_text", "Installing…")
        self.status_label.setText(f"{base}  •  elapsed {mins}m {secs:02d}s")

    def _stop_elapsed_timer(self) -> None:
        """Stop the install-phase QTimer if it's still running. Called
        from both terminal slots (done + error) so the status label
        stops ticking when the install resolves."""
        t = getattr(self, "_elapsed_timer", None)
        if t is not None and t.isActive():
            t.stop()

    def _on_install_done(self, runtime_info_dict: dict) -> None:
        """Worker emitted done(RuntimeInfo). Surface a brief success
        state then auto-accept so the caller can immediately verify
        via is_runtime_installed() + proceed to the Generate dialog."""
        self._stop_elapsed_timer()
        # Restore determinate mode for the final '100%' visual.
        self.progress.setRange(0, 100)
        self.was_installed = True
        cuda = runtime_info_dict.get("cuda_available", False)
        device = runtime_info_dict.get("cuda_device_name", "")
        torch_v = runtime_info_dict.get("torch_version", "")
        if cuda:
            self.status_label.setText(
                f"Done — torch {torch_v} installed, CUDA ready on {device}"
            )
        else:
            # Partial install — torch landed but CUDA isn't reachable.
            # The marker records this and is_runtime_installed() will
            # return False, so the EXE doesn't try to GPU-generate.
            self.status_label.setText(
                f"Installed torch {torch_v} but CUDA isn't available — "
                f"check your Nvidia driver version and try again."
            )
            self.was_installed = False
        self.progress.setValue(100)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(1500, self.accept)

    def _on_install_error(self, stage: str, message: str) -> None:
        """Worker emitted error(stage, message). Surface a modal with
        the stage tag so the user knows which phase to investigate.
        Leave the dialog open so they can dismiss after reading.

        'cancelled' stage gets a friendlier (QMessageBox.warning, not
        critical) treatment — that's the QUASAR-incident case where a
        leaked console got closed and the install died via NTSTATUS
        0xC000013A. The user knows what they did; we just want to tell
        them how to recover."""
        self._stop_elapsed_timer()
        self.progress.setRange(0, 100)
        self.was_installed = False
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setText("Close")

        if stage == "cancelled":
            self.status_label.setText("Install cancelled.")
            QMessageBox.warning(
                self, "Install cancelled",
                f"{message}\n\n"
                f"To retry cleanly: re-run Tools → Install GPU runtime "
                f"and leave all windows alone until it completes.",
            )
            return

        # Real failure path — stage-specific guidance instead of the
        # old one-size-fits-all 'use Colab' message.
        hint = self._error_hint_for_stage(stage)
        self.status_label.setText(f"Install failed at {stage}: {message}")
        QMessageBox.critical(
            self, f"GPU runtime install failed — {stage}",
            f"Stage: {stage}\n\n{message}\n\n{hint}\n\n"
            f"For post-mortem: Tools → Save diagnostics zip… and share "
            f"the resulting file.",
        )

    @staticmethod
    def _error_hint_for_stage(stage: str) -> str:
        """Stage-specific recovery suggestion. Each stage names what
        the user can do next, so they don't get a generic 'use Colab'
        for every failure."""
        return {
            "download_python": (
                "This is a network issue downloading the embedded "
                "Python zip. Check your internet connection and retry."
            ),
            "extract_python": (
                "The downloaded zip didn't extract cleanly. Delete "
                "the runtime folder and retry."
            ),
            "download_pip": (
                "Network issue downloading get-pip.py. Check your "
                "connection and retry."
            ),
            "bootstrap_pip": (
                "pip bootstrap failed. Most often this is a corrupted "
                "embedded Python — delete the runtime folder and retry."
            ),
            "pip_install": (
                "torch / numpy / Pillow couldn't be installed. Most "
                "common cause: network interruption during the multi-"
                "GB download. Retry; partial files have been cleaned "
                "up so the next attempt starts fresh."
            ),
            "copy_package": (
                "Couldn't copy the shape-gen package into the embedded "
                "Python. This is a file-permission or disk-space issue."
            ),
            "verify_cuda": (
                "torch installed but CUDA initialization failed. Check "
                "your Nvidia driver version (update from nvidia.com) "
                "and that no other process is hogging the GPU."
            ),
        }.get(stage, (
            "Unrecognized stage. Save the diagnostics zip and share for triage."
        ))


def prompt_install_or_use_existing(parent=None) -> bool:
    """Convenience entry point used by the Tools menu. Returns True if the
    runtime is ready to use (either was already installed, or user just
    installed it). False if the user declined the install or it failed."""
    if torch_installer.is_runtime_installed():
        return True
    dlg = RuntimeInstallDialog(parent)
    dlg.exec()
    return dlg.was_installed
