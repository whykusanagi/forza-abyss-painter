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

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QVBoxLayout,
)

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

        # Confirm-phase content — what the user is agreeing to.
        gib = torch_installer.estimated_download_bytes() / (1 << 30)
        self.body = QLabel(
            f"To generate shapes locally on your GPU, Forza Abyss Painter "
            f"needs an isolated PyTorch + CUDA runtime. This is a "
            f"<b>one-time download of ~{gib:.0f} GiB</b> stored in your "
            f"local app data folder (<code>{torch_installer.runtime_root()}</code>) "
            f"— it doesn't affect your system Python or other applications.\n\n"
            f"Subsequent runs reuse the cached runtime instantly.\n\n"
            f"You can skip this entirely and use the Colab notebooks "
            f"(see the README) — those run on a dedicated cloud GPU and "
            f"don't touch your machine. The local runtime is for users who "
            f"prefer to keep everything offline."
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
        """User clicked Install — switch to install-phase UI, then trigger
        the install. Phase 3 will replace the stub with a real worker
        thread that streams download progress through `_on_progress`."""
        self.body.setText(
            "Installing GPU runtime — this can take 5–15 minutes depending on "
            "your network speed. The dialog will close automatically when done."
        )
        self.progress.setVisible(True)
        self.status_label.setVisible(True)
        self.install_btn.setEnabled(False)
        self.cancel_btn.setText("Abort")

        # ---- PHASE 3 STUB ----
        # Real implementation: spawn a QThread worker that calls
        # torch_installer.install_runtime(progress_cb=self._on_progress).
        # When done, emits a signal that calls self._on_install_done(success: bool).
        # For now: surface a placeholder so the GUI flow is exercisable.
        self.status_label.setText(
            "(Phase 2 scaffolding — Phase 3 will plug in the real downloader.)"
        )
        # Simulate a finished-but-stubbed install so the caller can test the
        # downstream "runtime is installed" path. NOT a real install — caller
        # MUST check `is_runtime_installed()` separately before trusting.
        self.was_installed = False
        # Don't auto-close — let the user dismiss after reading the stub note.
        self.install_btn.setVisible(False)
        self.cancel_btn.setText("Close")

    def _on_progress(self, percent: int, status: str) -> None:
        """Called by the install worker (Phase 3) on each download chunk /
        install step. percent in [0, 100], status is a human-readable label
        like 'downloading torch-2.4.1+cu121-...whl (1.2 GiB / 2.4 GiB)'."""
        self.progress.setValue(max(0, min(100, percent)))
        self.status_label.setText(status)

    def _on_install_done(self, success: bool, message: str = "") -> None:
        """Called by the install worker (Phase 3) on completion. Closes the
        dialog with the right result code."""
        self.was_installed = success
        if success:
            self.accept()
        else:
            self.status_label.setText(f"Install failed: {message}")
            self.cancel_btn.setText("Close")
            self.install_btn.setVisible(False)


def prompt_install_or_use_existing(parent=None) -> bool:
    """Convenience entry point used by the Tools menu. Returns True if the
    runtime is ready to use (either was already installed, or user just
    installed it). False if the user declined the install or it failed."""
    if torch_installer.is_runtime_installed():
        return True
    dlg = RuntimeInstallDialog(parent)
    dlg.exec()
    return dlg.was_installed
