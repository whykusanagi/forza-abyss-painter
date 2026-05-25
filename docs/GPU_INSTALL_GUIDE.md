# GPU runtime install — what to expect, what NOT to do

A guide for testers and end users on first installing the local GPU
shape-gen runtime via **Tools → Install GPU runtime…** (or by picking
"GPU" from the *Generate using* dropdown in Settings).

## TL;DR

- First install takes **5–15 minutes** and downloads **~3 GB**
- **Do NOT close any windows** while installing
- The progress bar **may sit at 30%** for several minutes while torch
  downloads — that's normal; the bar animates so you can see it's
  still working, and the elapsed time counter ticks up in the status
  line
- If anything goes wrong: **Tools → Save diagnostics zip…** captures
  everything we need to triage from outside

## Walkthrough — successful first install

1. **Settings panel → "Generate using" dropdown → pick GPU**
   - If no install yet, the label reads **"GPU (Install required…)"**
   - Picking it auto-opens the install dialog
2. **Install dialog → "Install" button**
   - Body text reminds you not to close anything
3. **Progress bar moves to 30% almost immediately** (download Python
   embed zip + bootstrap pip — about 15 seconds total)
4. **Progress bar stays at 30% for 5–15 minutes**
   - This is the torch + CUDA + numpy + Pillow download (~3 GB)
   - The bar **animates** (sweeping pattern) so you know it's busy
   - The status line shows elapsed time: `Installing torch 2.4.1
     + deps … • elapsed 4m 23s`
5. **Progress jumps to 80%** when pip finishes
6. **Phases 80% → 100%** — copying our package + verifying CUDA
   (5–10 seconds total)
7. **Dialog auto-closes**, status bar shows
   `GPU: ready — NVIDIA RTX 4090` (or your card), and the Settings
   dropdown updates to **"GPU — NVIDIA RTX 4090"**

You can now click **Start** with GPU selected — runs go through the
local CUDA card.

## What CAN go wrong + what each error means

### "Install cancelled (a window was closed…)"

You closed something mid-install. Likely candidates:
- The black/grey cmd window if one leaked through (newer builds hide
  this — if you see one, that's a bug; report it)
- The install dialog itself
- The whole EXE

The installer detected NTSTATUS `0xC000013A` (STATUS_CONTROL_C_EXIT).
**Recovery:** click **Install GPU runtime** again and let it run. The
installer wipes the partial torch tree on real failures so retry
starts clean.

### "Install failed at pip_install"

Real pip / network issue. Most common: your connection dropped during
the multi-GB download. **Recovery:** retry. The partial install is
auto-cleaned so pip won't get confused on the second attempt.

### "Install failed at download_python" / "download_pip"

Network issue downloading the embedded Python zip or get-pip.py.
**Recovery:** check internet, retry.

### "Install failed at verify_cuda" — generic case

torch installed cleanly but CUDA initialization failed. The most
common cause: Nvidia driver too old. **Recovery:** update your Nvidia
driver from nvidia.com (you want ≥ 525.x for cu128 wheels), then
re-run install.

### "Install failed at verify_cuda" — "no CUDA kernel runs on this GPU"

You're on an RTX 50-series card (Blackwell, sm_120) and the installer
shipped a torch wheel that doesn't have sm_120 kernels. The installer
detects this by running a real kernel during verify, so this fails
NOW instead of silently slipping past install + crashing on first
Generate.

**As of this build:** the installer pins **`TORCH_CUDA_INDEX=cu128`**
and **`TORCH_VERSION=2.7.0`** — cu128 wheels include sm_50 through
sm_120 so all RTX 20/30/40/50 cards work out of the box.

**If you still see this error** after the cu128 bump landed: it
means the EXE you're running was built BEFORE the bump. Rebuild from
the latest source on SMB or pull a newer Release.

**RTX 5090 / 5080 / 5070 specific:** if you're on the older cu121
EXE, run the workaround script
`diagnostics/upgrade_embedded_torch_cu128.ps1` (on the SMB build
share) to re-pip the embedded runtime to cu128 nightly — that
unblocks you until the new EXE lands.

### "Install incomplete (CPU-only torch)"

The CUDA verify subprocess returned `torch.cuda.is_available() ==
False`. Same fix as above — driver update.

### Status bar says "GPU: install incomplete (CPU-only torch)" after install

Edge case: pip installed the CPU-only torch wheel instead of cu121.
Usually means cu121 index wasn't reachable at install time but the
fallback PyPI was. **Recovery:** delete
`%LOCALAPPDATA%\ForzaAbyssPainter\runtime\` and retry — the next
attempt will hit the cu121 index again.

## Where the logs live

Every install attempt + every GPU shape-gen run writes a structured
JSON-lines log to:

```
%LOCALAPPDATA%\ForzaAbyssPainter\logs\gpu-<timestamp>-main.log
%LOCALAPPDATA%\ForzaAbyssPainter\logs\gpu-<timestamp>-runner.log
```

- `main` = the EXE's own session (install events, dialog state,
  worker thread events)
- `runner` = the subprocess'd shape-gen process (only present when
  you run **Generate locally**)

Each log line is a JSON object with `kind`, `stage`, `elapsed_s`,
`thread`, and freeform fields. On failure, the `phase_error` event
includes the Python traceback.

## Save diagnostics zip

When something fails, click **Tools → Save diagnostics zip…**. It
bundles:

- All log files from the logs folder
- The install marker (`installed.json`) if present
- System info (OS, Python version, nvidia-smi output if installed)
- A README explaining what each file is

Default save location: `~/ForzaAbyssPainter-diag-<timestamp>.zip`.
Email or upload that single file — it's the complete diagnostic
context for triage.

## Manual recovery — when in doubt, nuke and retry

If the install gets into a confusing state and the in-app retry
isn't fixing it:

```powershell
# Close ForzaAbyssPainter first
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\ForzaAbyssPainter\runtime"
```

This wipes the entire embedded Python + torch tree. Logs are
preserved separately under `\logs\` so you don't lose diagnostic
history. Re-open the EXE and re-run **Install GPU runtime…** — it'll
start from scratch.
