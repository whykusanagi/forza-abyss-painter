"""Compare VRAM strategies on the smoke logo — consumer vs enterprise GPUs.

Implements a **standard DL microbatch loop** (chunk dim K, free activations between
steps) alongside the engine's current paths so QUASAR can prove what works on
~32 GiB cards vs what needs ~100 GiB.

Requires embedded runtime python (torch + forza_abyss_painter in site-packages).

Usage:
    & "$env:LOCALAPPDATA\ForzaAbyssPainter\runtime\python311\python.exe" `
      "\\QUASAR\\...\\diagnostics\\vram_strategy_harness.py"

    # Quick (fewer K, 2 chunks):
    ... vram_strategy_harness.py --quick

Outputs:
    diagnostics/smoke_results_upstream/vram_strategy_report.json
    diagnostics/smoke_results_upstream/vram_strategy_report.md
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _gib(n: int | float) -> float:
    return float(n) / (1024**3)


def _device_total_gib() -> float:
    import torch
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return round(props.total_memory / (1024**3), 2)


def _nvidia_used_mib() -> tuple[float, float]:
    try:
        line = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip().split(",")
        return float(line[0].strip()), float(line[1].strip())
    except Exception:
        return -1.0, -1.0


@dataclass
class MemSnap:
    allocated_gib: float = 0.0
    reserved_gib: float = 0.0
    max_allocated_gib: float = 0.0

    @classmethod
    def capture(cls) -> "MemSnap":
        import torch
        torch.cuda.synchronize()
        return cls(
            allocated_gib=round(_gib(torch.cuda.memory_allocated()), 4),
            reserved_gib=round(_gib(torch.cuda.memory_reserved()), 4),
            max_allocated_gib=round(_gib(torch.cuda.max_memory_allocated()), 4),
        )


@dataclass
class StrategyResult:
    name: str
    ok: bool
    wall_s: float
    peak_gib: float
    consumer_safe: bool
    enterprise_only: bool
    notes: str = ""
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _load_logo(res: int, root: Path) -> tuple[Any, Any, Any, int, int]:
    from PIL import Image
    import numpy as np
    import torch
    from forza_abyss_painter.shapegen.gpu.device import get_device

    seed = root / "source" / "assets" / "forza_abyss_painter_logo.png"
    img = Image.open(seed).convert("RGBA")
    w0, h0 = img.size
    scale = res / max(w0, h0)
    img = img.resize((int(w0 * scale), int(h0 * scale)), Image.Resampling.LANCZOS)
    rgb = np.array(img.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    dev = get_device()
    target = torch.from_numpy(rgb).to(dev)
    canvas = torch.full((h, w, 3), 40, dtype=torch.uint8, device=dev)
    return target, canvas, dev, h, w


def _reset_cuda() -> None:
    import torch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _theoretical_mask_gib(K: int, h: int, w: int, bytes_per: int = 4) -> float:
    return round(K * h * w * bytes_per / (1024**3), 3)


def strategy_monolithic_rasterize(
    kind, params, h: int, w: int,
) -> StrategyResult:
    """Current full_canvas anti-pattern: (K,H,W) masks in one kernel graph."""
    import torch
    t0 = time.perf_counter()
    _reset_cuda()
    try:
        masks = kind.rasterize_hard(params, h, w)
        peak = MemSnap.capture().max_allocated_gib
        del masks
        return StrategyResult(
            name="A_monolithic_rasterize",
            ok=True,
            wall_s=round(time.perf_counter() - t0, 2),
            peak_gib=peak,
            consumer_safe=peak < 24.0,
            enterprise_only=peak > 28.0,
            notes=f"Single (K,H,W) tensor ~{_theoretical_mask_gib(params.shape[0], h, w)} GiB theoretical",
        )
    except torch.cuda.OutOfMemoryError as exc:
        return StrategyResult(
            name="A_monolithic_rasterize",
            ok=False,
            wall_s=round(time.perf_counter() - t0, 2),
            peak_gib=MemSnap.capture().max_allocated_gib,
            consumer_safe=False,
            enterprise_only=True,
            error=str(exc)[:300],
            notes="Fails before scoring — chunking in planner never runs",
        )


def strategy_microbatch_rasterize_score(
    kind,
    params,
    canvas,
    target,
    h: int,
    w: int,
    k_chunk: int,
    max_chunks: int | None,
) -> StrategyResult:
    """DL-style microbatch: rasterize[k:k+C] → score → free → repeat."""
    import torch
    from forza_abyss_painter.shapegen.gpu.scoring import score_batch

    K = params.shape[0]
    if k_chunk <= 0 or k_chunk >= K:
        k_chunk = K
    t0 = time.perf_counter()
    _reset_cuda()
    best = float("inf")
    chunks_run = 0
    chunk_peaks: list[float] = []

    try:
        for k_start in range(0, K, k_chunk):
            if max_chunks is not None and chunks_run >= max_chunks:
                break
            k_end = min(k_start + k_chunk, K)
            masks = kind.rasterize_hard(params[k_start:k_end], h, w)
            scores, _c, _a = score_batch(masks, canvas, target)
            val = float(scores.min().cpu().item())
            best = min(best, val)
            chunk_peaks.append(MemSnap.capture().max_allocated_gib)
            del masks, scores
            torch.cuda.empty_cache()
            chunks_run += 1

        peak = max(chunk_peaks) if chunk_peaks else 0.0
        return StrategyResult(
            name="B_microbatch_rasterize_score",
            ok=True,
            wall_s=round(time.perf_counter() - t0, 2),
            peak_gib=round(peak, 4),
            consumer_safe=peak < 24.0,
            enterprise_only=False,
            notes=(
                f"Standard DL microbatch on K; chunk={k_chunk} "
                f"ran {chunks_run} chunks; best_score={best:.4f}"
            ),
            extra={
                "chunks_run": chunks_run,
                "chunk_peaks_gib": chunk_peaks,
                "per_chunk_theoretical_mask_gib": round(
                    _theoretical_mask_gib(k_chunk, h, w), 3
                ),
            },
        )
    except torch.cuda.OutOfMemoryError as exc:
        return StrategyResult(
            name="B_microbatch_rasterize_score",
            ok=False,
            wall_s=round(time.perf_counter() - t0, 2),
            peak_gib=MemSnap.capture().max_allocated_gib,
            consumer_safe=False,
            enterprise_only=True,
            error=str(exc)[:300],
            extra={"chunks_run": chunks_run, "chunk_peaks_gib": chunk_peaks},
        )


def strategy_bbox_local_chunked(
    kind,
    params,
    canvas,
    target,
    k_chunk: int,
    max_chunks: int | None,
) -> StrategyResult:
    """Engine path when bbox_local=True — no full (K,H,W) plane."""
    import torch
    from forza_abyss_painter.shapegen.gpu.bbox_score import (
        crop_score_ellipse_batch_chunked,
    )

    t0 = time.perf_counter()
    _reset_cuda()
    K = params.shape[0]
    chunks_run = 0
    chunk_peaks: list[float] = []

    try:
        from forza_abyss_painter.shapegen.gpu.bbox_score import (
            crop_score_ellipse_batch,
        )
        eff_chunk = k_chunk
        if max_chunks is not None and k_chunk > 0:
            # Run only first N chunks for a faster harness pass.
            eff_chunk = k_chunk
            for k_start in range(0, K, k_chunk):
                if chunks_run >= max_chunks:
                    break
                k_end = min(k_start + k_chunk, K)
                crop_score_ellipse_batch(
                    params[k_start:k_end], canvas, target,
                )
                chunk_peaks.append(MemSnap.capture().max_allocated_gib)
                torch.cuda.empty_cache()
                chunks_run += 1
        else:
            crop_score_ellipse_batch_chunked(
                params, canvas, target, chunk_size=k_chunk,
            )
            chunks_run = (
                (K + k_chunk - 1) // k_chunk if k_chunk > 0 else 1
            )
            chunk_peaks.append(MemSnap.capture().max_allocated_gib)

        peak = max(chunk_peaks) if chunk_peaks else MemSnap.capture().max_allocated_gib
        return StrategyResult(
            name="C_bbox_local_chunked",
            ok=True,
            wall_s=round(time.perf_counter() - t0, 2),
            peak_gib=round(peak, 4),
            consumer_safe=peak < 24.0,
            enterprise_only=False,
            notes="Activation tiling per candidate crop — production path for consumer GPUs",
            extra={"chunks_run": chunks_run, "chunk_peaks_gib": chunk_peaks},
        )
    except torch.cuda.OutOfMemoryError as exc:
        return StrategyResult(
            name="C_bbox_local_chunked",
            ok=False,
            wall_s=round(time.perf_counter() - t0, 2),
            peak_gib=MemSnap.capture().max_allocated_gib,
            consumer_safe=False,
            enterprise_only=True,
            error=str(exc)[:300],
        )


def strategy_planner_budget_check(
    K: int, h: int, w: int, budget_gib: float,
) -> dict[str, Any]:
    from forza_abyss_painter.shapegen.gpu import vram_planner as vp

    out: dict[str, Any] = {}
    for bbox_local, tag in ((False, "full_canvas"), (True, "bbox_local")):
        chunk = vp.resolve_k_chunk_size(
            K, bbox_local, max(h, w), budget_gib,
        )
        out[tag] = {
            "bytes_per_candidate_mb": round(
                vp._peak_bytes_per_candidate(bbox_local, max(h, w)) / 1e6, 2
            ),
            "estimate_all_K_gib": round(
                vp.estimate_peak_vram_gib(K, bbox_local, max(h, w)), 1
            ),
            "chunk_size": chunk or K,
            "n_chunks": (K + chunk - 1) // chunk if chunk else 1,
            "monolithic_mask_gib": _theoretical_mask_gib(K, h, w),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="VRAM strategy harness (logo sample)")
    parser.add_argument("--K", type=int, default=8192)
    parser.add_argument("--res", type=int, default=720)
    parser.add_argument("--budget", type=float, default=8.0,
                        help="Consumer VRAM budget GiB for chunk sizing")
    parser.add_argument("--quick", action="store_true",
                        help="K=2048, max 2 chunks per strategy")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Cap chunks (default all, or 2 if --quick)")
    args = parser.parse_args()

    if args.quick:
        args.K = 2048
        if args.max_chunks is None:
            args.max_chunks = 2

    root = Path(__file__).resolve().parents[1]
    out_dir = Path(__file__).resolve().parent / "smoke_results_upstream"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ImportError:
        print("Use embedded python.exe (has torch)", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 1

    device_gib = _device_total_gib()
    used_mib, free_mib = _nvidia_used_mib()
    consumer_card = device_gib > 0 and device_gib < 48
    enterprise_card = device_gib >= 80

    target, canvas, dev, h, w = _load_logo(args.res, root)
    from forza_abyss_painter.shapegen.gpu.engine import (
        KINDS, _resolve_k_chunk_size,
    )
    from forza_abyss_painter.shapegen.gpu import vram_planner as vp

    kind = KINDS["rotated_ellipse"]
    gen = torch.Generator(device="cpu").manual_seed(42)
    params = kind.init(args.K, w, h, gen).to(dev)

    k_chunk_full = _resolve_k_chunk_size(
        args.K, False, max(h, w), args.budget,
    )
    k_chunk_bbox = _resolve_k_chunk_size(
        args.K, True, max(h, w), args.budget,
    )
    # Microbatch B uses full_canvas chunk size (planner budget for scoring plane).
    k_chunk_micro = k_chunk_full if k_chunk_full > 0 else min(256, args.K)

    planner = strategy_planner_budget_check(args.K, h, w, args.budget)

    results: list[StrategyResult] = []
    results.append(strategy_monolithic_rasterize(kind, params, h, w))
    _reset_cuda()
    results.append(strategy_microbatch_rasterize_score(
        kind, params, canvas, target, h, w,
        k_chunk_micro, args.max_chunks,
    ))
    _reset_cuda()
    results.append(strategy_bbox_local_chunked(
        kind, params, canvas, target, k_chunk_bbox, args.max_chunks,
    ))

    report = {
        "sample_image": str(root / "source" / "assets" / "forza_abyss_painter_logo.png"),
        "canvas": f"{w}x{h}",
        "K": args.K,
        "vram_budget_gib": args.budget,
        "device_total_gib": device_gib,
        "nvidia_used_mib": used_mib,
        "nvidia_free_mib": free_mib,
        "consumer_class_gpu": consumer_card,
        "enterprise_class_gpu": enterprise_card,
        "k_chunk_microbatch": k_chunk_micro,
        "k_chunk_bbox_local": k_chunk_bbox,
        "planner_theory": planner,
        "strategies": [asdict(r) for r in results],
        "recommendation": {
            "consumer_32g": (
                "Use strategy C (bbox_local) OR B (microbatch rasterize+score). "
                "Never A for large K at 720p+. Wire B into engine when bbox_local=False."
            ),
            "enterprise_100g": (
                "Strategy A may succeed when monolithic_mask_gib + overhead < ~80 GiB; "
                "B/C still preferred for predictable peaks and FH6 headroom."
            ),
        },
    }

    json_path = out_dir / "vram_strategy_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# VRAM strategy harness — logo sample\n",
        f"- **Image:** `forza_abyss_painter_logo.png` @ {w}×{h}\n",
        f"- **K:** {args.K}  **Budget:** {args.budget} GiB  **GPU:** {device_gib} GiB total\n",
        f"- **Class:** {'consumer (~32G)' if consumer_card else 'other'}"
        f"{' / enterprise (80G+)' if enterprise_card else ''}\n\n",
        "## Planner theory vs monolithic mask\n\n",
        f"| | full_canvas | bbox_local |\n",
        f"|---|---|---|\n",
        f"| Monolithic (K,H,W) mask | **{planner['full_canvas']['monolithic_mask_gib']} GiB** | N/A (crop path) |\n",
        f"| Planner chunk @ budget | {planner['full_canvas']['chunk_size']} "
        f"({planner['full_canvas']['n_chunks']} chunks) | "
        f"{planner['bbox_local']['chunk_size']} "
        f"({planner['bbox_local']['n_chunks']} chunks) |\n",
        f"| Planner est. all K | {planner['full_canvas']['estimate_all_K_gib']} GiB | "
        f"{planner['bbox_local']['estimate_all_K_gib']} GiB |\n\n",
        "## Live strategies (measured)\n\n",
        "| Strategy | OK | Peak GiB | Consumer safe | Notes |\n",
        "|----------|----|---------:|---------------|-------|\n",
    ]
    for r in results:
        md_lines.append(
            f"| {r.name} | {'yes' if r.ok else '**no**'} | {r.peak_gib} | "
            f"{'yes' if r.consumer_safe else 'no'} | {r.notes[:80]} |\n"
        )
    md_lines.append("\n## Standard DL mapping\n\n")
    md_lines.append(
        "- **A** = no microbatching (full batch activation) — OK on 100G, breaks on 32G\n"
        "- **B** = microbatch + free activations — PyTorch/DataLoader pattern\n"
        "- **C** = tiled activations (bbox crops) — best default for consumer\n"
    )
    md_lines.append(f"\nFull JSON: `{json_path.name}`\n")

    md_path = out_dir / "vram_strategy_report.md"
    md_path.write_text("".join(md_lines), encoding="utf-8")

    print(f"Device: {device_gib} GiB  K={args.K}  canvas={w}x{h}")
    print(f"Monolithic mask theory: {_theoretical_mask_gib(args.K, h, w)} GiB")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(f"  [{status}] {r.name}: peak={r.peak_gib} GiB  {r.wall_s}s  {r.notes[:60]}")
    return 0 if all(r.ok for r in results[1:]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
