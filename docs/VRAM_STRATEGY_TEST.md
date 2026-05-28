# VRAM strategy harness — consumer vs enterprise

Confirms how to run smoke-logo GPU search on **32 GiB consumer** vs **80–100 GiB enterprise** cards using patterns standard in DL training.

## Run

```powershell
# Recommended — syncs shapegen/cli from share + runs harness
powershell -NoProfile -ExecutionPolicy Bypass -File `
  "\\QUASAR\ContentCreation\ForzaAbyssPainter_build\diagnostics\run_vram_strategy.ps1"

# Quick (K=2048, fewer chunks)
... run_vram_strategy.ps1 -Quick

# Or call Python directly:
& "$env:LOCALAPPDATA\ForzaAbyssPainter\runtime\python311\python.exe" `
  "\\QUASAR\ContentCreation\ForzaAbyssPainter_build\diagnostics\vram_strategy_harness.py"
```

Sync `source/forza_abyss_painter/shapegen` into embedded site-packages after upstream pulls (same as smoke).

## Strategies

| ID | Name | DL analogue | Consumer 32G |
|----|------|-------------|--------------|
| **A** | Monolithic `rasterize_hard(K,H,W)` | Full-batch forward, all activations live | **Avoid** — ~16 GiB masks alone @ K=8192, 720p |
| **B** | Microbatch rasterize+score chunks | `DataLoader` microbatch + `del` + `empty_cache` | **Use when** full-canvas scoring required |
| **C** | `bbox_local` chunked crops | Activation tiling / FlashAttention-style | **Default** for EXE |

## Outputs

- `smoke_results_upstream/vram_strategy_report.json`
- `smoke_results_upstream/vram_strategy_report.md`

## Upstream engine change (after harness proves B)

In `engine.py` full_canvas branch, replace:

```python
masks = kind.rasterize_hard(params, h, w)
scores = score_batch_chunked(masks, ..., chunk_size=_k_chunk)
```

with the same loop as **B** in `vram_strategy_harness.py` (rasterize slice → score → free).

Default `bbox_local=True` in EXE until that lands.
