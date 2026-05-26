# VRAM math vs what actually allocates (QUASAR Run 4)

## What `vram_planner` models (only one peak)

**Per-candidate scorer intermediate** (bbox-local crop path):

```
crop_e   = min(bbox_crop_max, max_resolution // 8)     # e.g. min(256, 90) = 90
footprint = min((2*crop_e+1)², res²)                   # e.g. 181² ≈ 32,761 px
bytes_per_candidate = footprint × 12 × BBOX_LOCAL_SAFETY  # 12 = float32×3 ch; safety=15
```

**Full-canvas path** (`bbox_local=False`, what smoke used):

```
footprint = res²                                       # 720² = 518,400 px
bytes_per_candidate = footprint × 12 × FULL_CANVAS_SAFETY  # safety=8
```

**Chunk size from budget:**

```
k_max = floor(vram_budget_gib × 1e9 / bytes_per_candidate)
if k_max >= K: no chunking
else chunk_size = k_max   # e.g. 8 GiB / ~50 MB ≈ 160  → matches Run 4 log
```

**UI estimate** (`settings_panel.estimate_peak_vram_gib`) uses **bbox_local=True**.
**Smoke / CLI** default `bbox_local=False` → planner uses **full-canvas** math at runtime.
That alone can make “estimated 5 GiB” vs “OOM at 8 GiB budget” feel contradictory.

## What the budget does NOT include (baseline VRAM)

These stay on GPU for the **whole run** (not divided by chunk):

| Tensor | Rough size @ 720×720 |
|--------|----------------------|
| `target` float (H,W,3) | ~7.4 MiB |
| `canvas` uint8 (H,W,3) | ~1.9 MiB |
| `alpha` / `edge_weight` | ~0.5–2 MiB |
| `params` (K, P) random ellipses | K × ~7 floats × 4 B |
| Rasterizer temps per shape | varies |
| Gradient refiner state | varies |
| **Allocator fragmentation** | can inflate PyTorch “allocated” |

Chunking only caps **one K-batch’s** `(chunk_size, footprint, 3)` scorer peak — not the sum of everything else.

## Why PyTorch says “47.51 GiB allocated” on a 32 GiB card

That number is **`torch.cuda.max_memory_allocated()`** — PyTorch’s caching allocator **high-water mark**, not “unique VRAM in use right now.” It can:

- Count **reserved** pools that weren’t returned to the driver
- **Accumulate** across many large alloc/free cycles (52 chunks × big temps)
- Exceed physical VRAM in the error string while the driver still reports ~4.5 GiB used in `nvidia-smi` (Run 4: ~4662 MiB used)

**Trust for debugging:** `torch.cuda.memory_allocated()` + `memory_reserved()` **at each phase**, and `nvidia-smi` mid-run.

## Today’s instrumentation

| Where | What |
|-------|------|
| `[chunked-K] …` stderr | K, chunk_size, n_chunks, budget |
| OOM handler (`engine.run_gpu`) | `max_memory_allocated` once, then empty_cache |
| `probe_gpu_and_runtime.py` | nvidia-smi snapshot, one kernel probe |
| **Missing** | Per-phase `allocated` / `reserved` inside the shape loop |

## Probe script

`diagnostics/vram_profile_smoke.py` — runs the same logo config, prints memory after each major step (and planner prediction for both bbox paths).
