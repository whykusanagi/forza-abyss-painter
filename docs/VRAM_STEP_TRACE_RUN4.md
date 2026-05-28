# VRAM step trace (upstream smoke, Anthony machine)

**Run:** `vram_profile_smoke.py --trace --both-paths --max-chunks 2`  
**GPU:** ~32 GiB total (OOM message: 31.84 GiB)  
**Config:** K=8192, canvas 720×720, VRAM budget 8.0 GiB (planner), smoke logo defaults  
**Artifacts:** `diagnostics/smoke_results_upstream/vram_step_trace_*.jsonl` / `.md`, `vram_trace_console.log`

## Key steps (both paths)

| Path | Step | alloc GiB | reserved GiB | max alloc GiB | Notes |
|------|------|-----------|--------------|---------------|-------|
| **full_canvas** (`bbox_local=False`) | baseline (`04_canvas_gpu`) | 0.0029 | 0.0195 | 0.0029 | target + canvas on GPU |
| **full_canvas** | after `init_params` | 0.003 | 0.0215 | 0.003 | params `(8192, 5)` |
| **full_canvas** | **after_rasterize_hard** | — | — | — | **Not reached** — OOM inside `rasterize_hard` |
| **full_canvas** | `rasterize_failed` (peak) | **47.5081** | 47.5293 | **47.5081** | CUDA OOM; extra alloc **15.82 GiB** requested |
| **full_canvas** | chunk scoring | — | — | — | Never ran (no masks tensor) |
| **bbox_local** (`bbox_local=True`) | baseline (`04_canvas_gpu`) | 0.0029 | 47.5293* | 0.0029 | *elevated reserved after prior full_canvas OOM until cache cleared |
| **bbox_local** | `bbox_score_chunk0_after` | 0.003 | 47.5352 | **3.657** | k=0..1356 (chunk 1356) |
| **bbox_local** | `bbox_score_chunk1_after` | 0.003 | 3.9062 | **3.6633** | k=1356..2712; stopped at 2 chunks |

### full_canvas rows (requested detail)

| step | alloc GiB | max GiB | delta MiB | masks / error |
|------|-----------|---------|-----------|----------------|
| `04_canvas_gpu` | 0.0029 | 0.0029 | +1.48 | — |
| `after_init_params` | 0.003 | 0.003 | +0.16 | params only |
| `after_rasterize_hard` | *n/a* | *n/a* | *n/a* | **not reached** |
| `rasterize_failed` | 47.5081 | 47.5081 | +48645 | `Tried to allocate 15.82 GiB` (single tensor ~K×H×W float32) |

If `after_rasterize_hard` had succeeded, expected `masks_shape` would be `(8192, 720, 720)` → **~15.9 GiB** dense float32 (`8192×720×720×4` bytes ≈ 15.87 GiB), before any `score_batch` chunking.

## Why full_canvas OOMs

The engine path with `bbox_local=False` calls `kind.rasterize_hard(params, h, w)` for **all K=8192** kinds in one shot. That materializes a dense mask tensor shaped **(K, H, W)** on GPU:

- **K = 8192**, **H = W = 720**
- **Elements** = 8192 × 720 × 720 = 4,243,353,600
- **float32** → 4 bytes/elem → **~17.0×10⁹ bytes (~15.9 GiB)** for masks alone

On a **~32 GiB** card, one allocation of ~16 GiB plus canvas, params, allocator reserved pool, and **rasterize internals** (the trace peaked at **~47.5 GiB** `max_memory_allocated` before failure) exceeds capacity. Chunking in `vram_planner` (`k_chunk=160` for scoring) never applies because chunking is only wired for **scoring**, not for **up-front full rasterize**.

## Compare to `vram_planner` theory

Printed at trace start (same formulas as `shapegen/gpu/vram_planner.py`):

| Mode | bytes/candidate (model) | est. all K=8192 | chunk @ 8 GiB budget | theory one chunk |
|------|-------------------------|-----------------|----------------------|------------------|
| bbox_local | 5.90 MiB | **48.3 GiB** | 1356 × 7 chunks | **8.00 GiB** |
| full_canvas | 49.77 MiB | **407.7 GiB** | 160 × 52 chunks | **7.96 GiB** |

**Observed vs theory**

1. **full_canvas:** Planner correctly predicts that scoring **all K** at full canvas is hundreds of GiB; it also suggests **160** candidates per chunk for an 8 GiB budget. The trace shows failure **earlier**: monolithic **rasterize** is not in the planner’s `_peak_bytes_per_candidate` model (which assumes `(K, footprint, 3)` scoring intermediates). Live failure is dominated by **(K,H,W)** masks, not the 49.77 MiB/cand scoring estimate alone.
2. **bbox_local:** Two chunks peaked at **~3.66 GiB** `max_memory_allocated` per chunk vs **~8 GiB** theoretical cap — planner is **conservative** here (safety multipliers 15× bbox / 8× full canvas). That matches prior calibration notes (smoke once saw ~47.5 GiB on a path; bbox trace after OOM still shows allocator reserved noise on step 0).
3. **Budget 8 GiB** is meaningful for **chunked bbox scoring** but does **not** protect the **full_canvas rasterize** path on this K and resolution.

## DL frameworks vs what we should do

- **Gradient checkpointing (recompute activations):** Trade compute for memory by not storing every intermediate; **analog:** rasterize/score **one chunk of K** at a time and drop masks before the next chunk instead of holding `(K,H,W)`.
- **Microbatching / gradient accumulation:** Split batch dimension across forward passes; **analog:** enforce `vram_planner.resolve_k_chunk_size` on **rasterize** as well as `score_batch` / `crop_score_ellipse_batch` (same K-chunk loop end-to-end).
- **Activation/attention tiling (FlashAttention-style):** Never form the full N×N block; **analog:** **bbox_local** — score in **O(crop²)** tiles per candidate, not full **H×W** planes for all K.
- **Mixed precision (fp16/bf16):** Halve tensor footprint where accuracy allows; **analog:** optional **fp16/bool** masks for rasterize buffer if scoring tolerates it (measure quality).
- **CPU/off-GPU staging & pinned memory:** Keep oversized batches off GPU until needed; **analog:** rasterize chunk on GPU → score → **`.cpu()`/delete** before next chunk; never allocate **8192×720×720** in one kernel graph.

**Product/engineering takeaway:** Treat **monolithic `rasterize_hard` for full K** as a bug for large K; align implementation with planner chunking, default smoke/EXE to **bbox_local** for gradient refine, and surface **“rasterize peak = K×H×W”** in the settings panel when `bbox_local=False`.

---
Generated from upstream trace on share build synced to embedded `site-packages` (shapegen/io/runtime/cli).
