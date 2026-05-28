"""First-launch GPU detection + install offer (#99).

Right now, a new user has to know to click `Tools → Install GPU
runtime…` to get GPU-accelerated shape-gen. That's a discoverability
gap: most users won't think to look for it, and the CPU path is
unusably slow at the resolutions / shape counts the GPU path makes
practical. This module fixes that with a one-time prompt on first
launch when we detect a CUDA-capable GPU.

## Logic

On startup (`showEvent`):

  1. If `GPU_PHASE_3_AVAILABLE` is False → skip (no GPU UX at all).
  2. If the GPU runtime is already installed → skip (already done).
  3. If the user previously chose "Don't ask again" → skip.
  4. Probe nvidia-smi:
     - No driver / no NVIDIA GPU → skip silently. Don't nag users
       on AMD/Intel/Mac who can't use the path anyway.
     - GPU detected → show the install offer modal.
  5. User picks Install Now / Maybe Later / Don't Ask Again.
     - "Install Now" → triggers the existing RuntimeInstallDialog.
     - "Maybe Later" → no persisted state, re-asks next session.
     - "Don't Ask Again" → persisted, never re-asks (user can still
       install manually via Tools menu).

## Persistence

Single QSettings key under `gpu_first_launch.never_ask`. Stored once
when the user picks "Don't Ask Again" — never read back to anything
else. If we ever ship a "reset prompts" feature, that's the key to
clear.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMessageBox, QWidget

from forza_abyss_painter.runtime.nvidia_smi import probe_free_vram, ProbeResult


_SETTINGS_GROUP = "gpu_first_launch"
_KEY_NEVER_ASK = "never_ask"


class GpuPromptDecision(Enum):
    """Outcomes from the first-launch prompt. Used by the caller to
    decide whether to chain into the install dialog."""
    INSTALL_NOW = "install_now"
    MAYBE_LATER = "maybe_later"
    NEVER_ASK_AGAIN = "never_ask"
    SKIPPED_NO_GPU = "skipped_no_gpu"
    SKIPPED_ALREADY_INSTALLED = "skipped_already_installed"
    SKIPPED_USER_OPTED_OUT = "skipped_user_opted_out"
    SKIPPED_FLAG_DISABLED = "skipped_flag_disabled"


@dataclass(frozen=True)
class GpuPromptResult:
    """Decision plus the probe result that drove it. Returning the
    probe data lets the caller log it (the GPU name on the user's
    box is useful triage info)."""
    decision: GpuPromptDecision
    probe: ProbeResult | None = None


def user_opted_out() -> bool:
    """Has the user previously clicked 'Don't Ask Again'?"""
    s = QSettings("ForzaAbyssPainter", "Forza Abyss Painter")
    s.beginGroup(_SETTINGS_GROUP)
    try:
        raw = s.value(_KEY_NEVER_ASK, False)
    finally:
        s.endGroup()
    # QSettings returns the value type-shaped per platform — on
    # Windows it's a string "true"/"false", on macOS it's a bool.
    # Normalize via str(...).lower() to handle both without dragging
    # in QVariant juggling.
    return str(raw).lower() in ("1", "true", "yes")


def set_user_opted_out() -> None:
    """Persist 'Don't Ask Again'. Caller is responsible for invoking
    this only when the user explicitly picked that option."""
    s = QSettings("ForzaAbyssPainter", "Forza Abyss Painter")
    s.beginGroup(_SETTINGS_GROUP)
    try:
        s.setValue(_KEY_NEVER_ASK, True)
    finally:
        s.endGroup()


def should_prompt() -> tuple[bool, GpuPromptDecision, ProbeResult | None]:
    """Return (should_prompt, skip_reason_if_not, probe_result_or_None).

    Pure function — no UI side effects. Called by maybe_prompt before
    constructing any QMessageBox so callers can also use this in
    headless contexts (e.g. CLI startup banner)."""
    from forza_abyss_painter.gui.feature_flags import GPU_PHASE_3_AVAILABLE
    if not GPU_PHASE_3_AVAILABLE:
        return False, GpuPromptDecision.SKIPPED_FLAG_DISABLED, None
    from forza_abyss_painter.runtime.torch_installer import is_runtime_installed
    if is_runtime_installed():
        return False, GpuPromptDecision.SKIPPED_ALREADY_INSTALLED, None
    if user_opted_out():
        return False, GpuPromptDecision.SKIPPED_USER_OPTED_OUT, None
    probe = probe_free_vram()
    if not probe.available:
        # No NVIDIA driver → don't nag users who can't use the path.
        # Includes AMD/Intel GPUs, macOS, and WSL without GPU
        # passthrough. The probe.reason captures the WHY for logs.
        return False, GpuPromptDecision.SKIPPED_NO_GPU, probe
    return True, GpuPromptDecision.INSTALL_NOW, probe   # placeholder; user picks


def maybe_prompt(parent: QWidget | None) -> GpuPromptResult:
    """Show the first-launch GPU install offer if appropriate.

    Returns a GpuPromptResult describing what happened. Caller chains
    into RuntimeInstallDialog only when decision is INSTALL_NOW."""
    should, skip_reason, probe = should_prompt()
    if not should:
        return GpuPromptResult(decision=skip_reason, probe=probe)
    # Build the modal. Three buttons + the GPU name in the body so the
    # user sees we actually detected something specific (not a generic
    # 'do you want torch?' nag).
    gpu_name = probe.name if probe else "your CUDA GPU"
    vram_line = ""
    if probe and probe.total_gib is not None:
        vram_line = f"<br><i>{probe.total_gib:.1f} GiB total VRAM</i>"
    msg = QMessageBox(parent)
    msg.setWindowTitle("CUDA GPU detected — install accelerator?")
    msg.setIcon(QMessageBox.Question)
    msg.setText(
        f"<b>Detected a CUDA-capable GPU: {gpu_name}</b>{vram_line}<br><br>"
        f"Forza Abyss Painter can use the GPU for shape generation, "
        f"which is roughly <b>10–100× faster</b> than the CPU path "
        f"at production shape counts.<br><br>"
        f"The first install downloads ~1.5 GiB of embedded Python + "
        f"PyTorch (one-time, ~3–5 minutes). After that, every shape-"
        f"gen run uses the GPU automatically.<br><br>"
        f"You can always install later via "
        f"<b>Tools → Install GPU runtime…</b>"
    )
    install_btn = msg.addButton("Install now", QMessageBox.AcceptRole)
    later_btn = msg.addButton("Maybe later", QMessageBox.RejectRole)
    never_btn = msg.addButton("Don't ask again", QMessageBox.DestructiveRole)
    msg.setDefaultButton(install_btn)
    msg.exec()
    clicked = msg.clickedButton()
    if clicked is install_btn:
        return GpuPromptResult(GpuPromptDecision.INSTALL_NOW, probe)
    if clicked is never_btn:
        set_user_opted_out()
        return GpuPromptResult(GpuPromptDecision.NEVER_ASK_AGAIN, probe)
    # Default: "Maybe later" — also the path for window-close (X).
    return GpuPromptResult(GpuPromptDecision.MAYBE_LATER, probe)
