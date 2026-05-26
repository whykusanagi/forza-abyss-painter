# Cursor GPU smoke test brief — end-to-end runtime validation

> **Audience:** Cursor agent on QUASAR after a fresh EXE rebuild.
>
> **Goal:** validate the full GPU pipeline (install → generate → cancel
> → diagnose) end-to-end with one fixed seed image, capture concrete
> metrics, and report back so we can see "is this working as expected
> on real hardware" without me asking the user to manually click
> through everything.

## What this proves

The EXE-side smoke tests (run on macOS via offscreen Qt) cover the
plumbing. **This brief covers what only QUASAR can prove:**

1. The cu128 torch wheels actually install + the real-kernel verify
   doesn't crash on RTX 5090 (sm_120)
2. `fap-generate` end-to-end runs without a leaked cmd window and
   produces a JSON that's structurally valid + non-empty
3. The chunked-K mode (engaging on a 12 GiB budget vs the run's
   peak) doesn't break the output — same per-shape quality, just
   slower wall time
4. The kill-ladder actually reaps an in-flight CUDA run within the
   expected 14s when cancelled mid-stream
5. `fap-diag` produces a zip that round-trips through the parser

If any of those fail on QUASAR, that's a real Windows-side or
hardware-side issue we couldn't have caught locally.

## Seed image — fixed for reproducibility

Use the bundled app icon: `assets/forza_abyss_painter_logo.png` —
800×800 RGBA, ships with every build. Don't substitute another image
unless you're explicitly comparing runs.

Why this seed: the icon has hard edges + flat color regions + alpha
transparency — covers the major engine code paths (sticker mode,
edge weighting, alpha-mask scoring) in one go.

## Run procedure

Steps are sequenced — if step N fails, fix it before moving to N+1.
Each step's metrics get filled into the results template below.

### Pre-flight (must pass before testing)

```powershell
# 1. EXE rebuilt from latest source/ on the SMB?
Get-Item "$env:USERPROFILE\Desktop\ForzaAbyssPainter_build\dist\ForzaAbyssPainter.exe" | Select LastWriteTime

# 2. Runtime installed + cuda_available=true?
Get-Content "$env:LOCALAPPDATA\ForzaAbyssPainter\runtime\installed.json"
```

Required: `cuda_available: true` in the marker. If not, run
**Tools → Install GPU runtime…** from the EXE first.

### Step 1 — `fap-generate` end-to-end

Run a moderate-size GPU shape-gen and capture metrics:

```powershell
$py = "$env:LOCALAPPDATA\ForzaAbyssPainter\runtime\python311\python.exe"
$seed = "$env:USERPROFILE\Desktop\ForzaAbyssPainter_build\source\assets\forza_abyss_painter_logo.png"
$out = "$env:USERPROFILE\Desktop\smoke_logo_1000.json"

$t0 = Get-Date
& $py -m forza_abyss_painter.cli.generate `
    --image $seed `
    --output $out `
    --num-shapes 1000 `
    --max-resolution 720 `
    --random-samples 8192 `
    --vram-budget 8.0 `
    --preset-label "smoke-1000" `
    2> "$env:USERPROFILE\Desktop\smoke_logo_1000.stderr.log"
$rc = $LASTEXITCODE
$elapsed = (Get-Date) - $t0
Write-Host "fap-generate exit=$rc elapsed=$([math]::Round($elapsed.TotalSeconds, 1))s"
```

**Capture into the results template:**
- Exit code (must be 0)
- Wall-clock elapsed seconds
- Output file size + first 3 lines of stderr (verifies the IPC
  JSON line stream is well-formed)
- From stderr: any `error` events; the final `done` event's
  `shape_count`
- Whether the engine logged `[chunked-K]` (it should — 8 GiB
  budget + 8192 samples + 720 max_res = likely 1–2 chunks)
- From `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`
  taken DURING the run (run it 30s in, before completion): peak VRAM
  used in MiB

### Step 2 — output JSON validation

Parse the output JSON + sanity-check:

```powershell
$json = Get-Content $out -Raw | ConvertFrom-Json
Write-Host "format = $($json.format)  version = $($json.version)"
Write-Host "image_size = $($json.image_size -join 'x')"
Write-Host "shape_count = $($json.shape_count)"
Write-Host "sticker_mode = $($json.sticker_mode)"
Write-Host "first_shape = $($json.shapes[0] | ConvertTo-Json -Compress)"
```

**Must hold:**
- `format == "fd6.shapes"` (canonical)
- `version == 1`
- `shape_count == 1000`
- `len($json.shapes) == 1000` (no shape budget reclaimed below target)
- First shape dict has `type, x, y, rx, ry, angle, color` (per shape
  type `rotated_ellipse`)

### Step 3 — cancel mid-run (kill-ladder validation)

This requires interactive timing. Start a longer run, wait until
status bar reads `~30%`, then click Stop. The status bar should
flip to `Stopping…` then `Cancelled` within 14 seconds.

```powershell
# Long-running variant to give cancel time to fire
& $py -m forza_abyss_painter.cli.generate `
    --image $seed `
    --output "$env:USERPROFILE\Desktop\smoke_cancel_target.json" `
    --num-shapes 3000 `
    --max-resolution 1200 `
    --random-samples 16384 `
    --vram-budget 12.0 `
    --preset-label "smoke-cancel"
# While running, press Ctrl-C ONCE (sends SIGINT / CTRL_BREAK_EVENT).
# Runner's handler should:
#   1. emit {kind:error, stage:cancelled, ...} on stderr
#   2. call torch.cuda.empty_cache()
#   3. sys.exit(1) within ~10 seconds
```

**Capture:**
- Time between Ctrl-C and process exit (Get-Date before + after)
- Last 5 stderr lines (must include `stage: cancelled` event)
- `nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader`
  immediately after exit — used VRAM should drop close to its
  pre-run baseline within ~5 seconds (driver reclaim). If VRAM stays
  pinned high, the SIGINT handler didn't free the cache.

### Step 4 — probe + diagnostics bundle

After the above runs (success + cancelled), capture the full state:

```powershell
cd "$env:USERPROFILE\Desktop\ForzaAbyssPainter_build\diagnostics"
py -3 probe_gpu_and_runtime.py --output cursor_gpu_probe.json

# fap-diag from the embedded python — bundles logs + marker + sysinfo
& $py -m forza_abyss_painter.runtime.diagnostics `
    --output "$env:USERPROFILE\Desktop\smoke_diag.zip"
```

**Capture:**
- `cursor_gpu_probe.json` — full file contents (small, paste inline)
- `smoke_diag.zip` size + top-level file listing
- The kill-ladder self-test result from the probe
  (`kill_ladder_self_test.stage_that_reaped`)

## Results template — fill in + share

Copy this template, fill in the metrics, paste into chat or save to
`diagnostics/cursor_smoke_results.md`:

```markdown
# QUASAR GPU smoke run — YYYY-MM-DD

## Pre-flight
- EXE build timestamp: ...
- Marker cuda_available: ...
- Driver version (nvidia-smi): ...
- Device + compute_capability: ...

## Step 1: fap-generate 1000 shapes
- Exit code: ...
- Wall time: ... seconds
- ms/shape (wall_time × 1000 / shape_count): ...
- Output file size: ... KiB
- chunked-K activated? (Y/N + chunk count): ...
- Peak VRAM observed (mid-run nvidia-smi): ... MiB
- Free VRAM remaining: ... MiB
- Any error events in stderr: ...

## Step 2: output JSON validation
- format: ...
- version: ...
- shape_count: ... (expected 1000)
- sticker_mode: ...
- First shape sample: ...
- All shapes have rotated_ellipse type? Y/N

## Step 3: kill-ladder mid-run cancel
- Cancel-to-exit elapsed: ... seconds
- stage:cancelled event present in stderr? Y/N
- VRAM reclaimed after exit? Y/N
- VRAM after exit (MiB) vs pre-run baseline (MiB): ...

## Step 4: probe + diagnostics
- probe kill_ladder_self_test.stage_that_reaped: ...
- probe total_seconds: ...
- orphan_processes count: ...
- smoke_diag.zip size: ... KiB
- smoke_diag.zip contents: README.md, system_info.json, runtime/installed.json, logs/...

## Verdict
- Overall PASS / FAIL: ...
- Surprises or anomalies: ...
- Reference timing match? (1000 shapes on a {GPU_CLASS}; expected ... min, observed ... min): ...
```

## What we expect to see

For an RTX 4090 / 4080 — the kind of card you'd test on:
- Step 1 wall time: 60–120 seconds for 1000 shapes
- ms/shape: 60–120 ms
- Peak VRAM mid-run: 6–8 GiB (chunked-K should keep it under 8 GiB budget)
- chunked-K activated: yes, 1–2 chunks
- Step 3 cancel: 2–8 seconds (graceful path); VRAM drops within 5s

For RTX 5090:
- Step 1: 30–60 seconds
- ms/shape: 30–60 ms
- Peak VRAM: similar (chunked-K respects budget regardless of card)
- Same cancel behavior

If your numbers are 3× slower than these targets, capture
`cursor_gpu_probe.json` AND the runner log file from
`%LOCALAPPDATA%\ForzaAbyssPainter\logs\` — most likely cause is
chunked-K firing more times than expected because some other app is
holding VRAM.

## Fallback if `fap-generate` isn't available yet

If the embedded Python doesn't have the `forza_abyss_painter.cli`
module (build didn't bundle it), drop back to using the GUI:

1. Open EXE → **Tools → Generate shapes locally (GPU)…**
2. Pick the logo file as source
3. Pick a preset matching 1000 shapes
4. Click Generate
5. Time it manually + report the same metrics from above

The CLI is convenience; the GUI flow produces identical output.
