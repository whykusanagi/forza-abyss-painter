"""Per-step CUDA VRAM trace for smoke logo config (embedded runtime).

Prints planner theory + live torch.cuda bytes after each allocation phase.
Use the embedded python.exe (has torch), not host py -3.

Usage:
    & "$env:LOCALAPPDATA\ForzaAbyssPainter\runtime\python311\python.exe" `
      "\\QUASAR\\...\\diagnostics\\vram_profile_smoke.py" --trace

Outputs JSON lines to stderr and optional --out file for SMB handoff.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _gib(nbytes: int) -> float:
    return nbytes / (1024**3)


def _nvidia_used_mib() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        return out.replace("\n", " | ")
    except Exception as exc:
        return f"unavailable ({exc})"


class _Tracer:
    def __init__(self, out_path: Path | None) -> None:
        self._steps: list[dict] = []
        self._out_path = out_path
        self._prev_alloc = 0

    def snap(self, label: str, **extra) -> dict:
        import torch
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated()
        row = {
            "step": label,
            "allocated_gib": round(_gib(alloc), 4),
            "reserved_gib": round(_gib(torch.cuda.memory_reserved()), 4),
            "max_allocated_gib": round(
                _gib(torch.cuda.max_memory_allocated()), 4
            ),
            "delta_alloc_mib": round((alloc - self._prev_alloc) / (1024**2), 2),
            "nvidia_smi_mib": _nvidia_used_mib(),
            **extra,
        }
        self._prev_alloc = alloc
        self._steps.append(row)
        line = json.dumps(row)
        print(line, file=sys.stderr, flush=True)
        if self._out_path is not None:
            with self._out_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return row

    def write_summary_md(self, path: Path) -> None:
        lines = [
            "# VRAM step trace\n",
            "| step | alloc GiB | reserved GiB | delta MiB | max GiB | notes |\n",
            "|------|-----------|--------------|-----------|---------|-------|\n",
        ]
        for r in self._steps:
            notes = " ".join(
                f"{k}={v}" for k, v in r.items()
                if k not in {
                    "step", "allocated_gib", "reserved_gib",
                    "max_allocated_gib", "delta_alloc_mib", "nvidia_smi_mib",
                }
            )
            lines.append(
                f"| {r['step']} | {r['allocated_gib']} | {r['reserved_gib']} | "
                f"{r['delta_alloc_mib']} | {r['max_allocated_gib']} | {notes} |\n"
            )
        path.write_text("".join(lines), encoding="utf-8")


def _planner_table(K: int, res: int, budget: float) -> None:
    from forza_abyss_painter.shapegen.gpu import vram_planner as vp

    print("=== vram_planner theory ===", file=sys.stderr)
    print(f"K={K} res={res} budget={budget} GiB", file=sys.stderr)
    for bbox_local, name in ((True, "bbox_local"), (False, "full_canvas")):
        bpc = vp._peak_bytes_per_candidate(bbox_local, res)
        est = vp.estimate_peak_vram_gib(K, bbox_local, res)
        chunk = vp.resolve_k_chunk_size(K, bbox_local, res, budget)
        n_chunks = (K + chunk - 1) // chunk if chunk > 0 else 1
        theory_chunk_gib = (chunk * bpc / 1e9) if chunk else est
        print(
            f"  [{name}] bpc={bpc/1e6:.2f}MB est_all_K={est:.1f}GiB "
            f"chunk={chunk or 'all'} n_chunks={n_chunks} "
            f"theory_one_chunk={theory_chunk_gib:.2f}GiB",
            file=sys.stderr,
        )


def _trace_chunks_generic(
    tr: _Tracer,
    *,
    K: int,
    k_chunk: int,
    max_chunks: int,
    run_one_chunk,
    label: str,
) -> None:
    """Call run_one_chunk(k_start, k_end) with snaps; run_one_chunk does the work."""
    if k_chunk <= 0 or k_chunk >= K:
        tr.snap(f"{label}_single_start", K=K)
        run_one_chunk(0, K)
        tr.snap(f"{label}_single_end", K=K)
        return

    n_chunks = (K + k_chunk - 1) // k_chunk
    tr.snap(f"{label}_begin", K=K, chunk_size=k_chunk, n_chunks=n_chunks)
    for i, k_start in enumerate(range(0, K, k_chunk)):
        if i >= max_chunks:
            tr.snap(f"{label}_stopped", ran=i)
            break
        k_end = min(k_start + k_chunk, K)
        tr.snap(f"{label}_chunk{i}_before", k_start=k_start, k_end=k_end)
        run_one_chunk(k_start, k_end)
        tr.snap(f"{label}_chunk{i}_after", k_start=k_start, k_end=k_end)
        import torch
        torch.cuda.empty_cache()
        tr.snap(f"{label}_chunk{i}_cached", k_start=k_start)


def _trace_bbox_path(
    tr: _Tracer,
    *,
    params,
    canvas,
    target,
    k_chunk: int,
    max_chunks: int,
) -> None:
    from forza_abyss_painter.shapegen.gpu.bbox_score import crop_score_ellipse_batch

    K = params.shape[0]

    def one(k_start: int, k_end: int) -> None:
        crop_score_ellipse_batch(params[k_start:k_end], canvas, target)

    _trace_chunks_generic(
        tr, K=K, k_chunk=k_chunk, max_chunks=max_chunks,
        run_one_chunk=one, label="bbox_score",
    )


def _trace_full_canvas_path(
    tr: _Tracer,
    *,
    kind,
    canvas,
    target,
    w: int,
    h: int,
    K: int,
    k_chunk: int,
    max_chunks: int,
    gen,
) -> None:
    """Engine path when bbox_local=False: rasterize (K,H,W) then score_batch."""
    from forza_abyss_painter.shapegen.gpu.scoring import score_batch

    per_kind = max(1, K)
    tr.snap("rasterize_start", per_kind=per_kind, canvas=f"{w}x{h}")
    params = kind.init(per_kind, w, h, gen).to(device_holder["dev"])
    tr.snap("after_init_params", shape=tuple(params.shape))

    try:
        masks = kind.rasterize_hard(params, h, w)
        tr.snap(
            "after_rasterize_hard",
            masks_shape=tuple(masks.shape),
            masks_gib=round(masks.numel() * 4 / 1e9, 3),
        )
    except Exception as exc:
        tr.snap("rasterize_failed", error=str(exc)[:120])
        return

    def one(k_start: int, k_end: int) -> None:
        score_batch(
            masks[k_start:k_end], canvas, target,
        )

    _trace_chunks_generic(
        tr, K=masks.shape[0], k_chunk=k_chunk, max_chunks=max_chunks,
        run_one_chunk=one, label="score_batch",
    )
    del masks, params


device_holder: dict = {"dev": None}


def _trace_scoring(
    tr: _Tracer,
    *,
    use_bbox: bool,
    kind,
    canvas,
    target,
    w: int,
    h: int,
    K: int,
    k_chunk: int,
    max_chunks: int,
    gen,
) -> None:
    if use_bbox:
        params = kind.init(K, w, h, gen).to(device_holder["dev"])
        tr.snap("params_bbox", shape=tuple(params.shape))
        _trace_bbox_path(
            tr, params=params, canvas=canvas, target=target,
            k_chunk=k_chunk, max_chunks=max_chunks,
        )
    else:
        _trace_full_canvas_path(
            tr, kind=kind, canvas=canvas, target=target,
            w=w, h=h, K=K, k_chunk=k_chunk, max_chunks=max_chunks, gen=gen,
        )


def _run_trace(
    *,
    K: int,
    res: int,
    budget: float,
    bbox_local: bool,
    max_chunks: int,
    out_path: Path | None,
) -> int:
    import torch
    from PIL import Image
    import numpy as np
    from forza_abyss_painter.shapegen.gpu.device import get_device
    from forza_abyss_painter.shapegen.gpu.engine import (
        GPUConfig, _resolve_k_chunk_size, KINDS,
    )

    root = Path(__file__).resolve().parents[1]
    seed_img = root / "source" / "assets" / "forza_abyss_painter_logo.png"
    if not seed_img.is_file():
        print(f"seed missing: {seed_img}", file=sys.stderr)
        return 1

    if out_path and out_path.exists():
        out_path.unlink()

    tr = _Tracer(out_path)
    _planner_table(K, res, budget)

    torch.cuda.reset_peak_memory_stats()
    tr.snap("00_import_torch")

    img = Image.open(seed_img).convert("RGBA")
    w, h = img.size
    scale = res / max(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    rgb = np.array(img.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    device = get_device()
    device_holder["dev"] = device
    tr.snap("01_setup_cpu_image", canvas=f"{w}x{h}")

    cfg = GPUConfig(
        num_shapes=1,
        random_samples=K,
        vram_budget_gib=budget,
        bbox_local=bbox_local,
        refine_mode="gradient",
        shape_types=["rotated_ellipse"],
    )
    use_bbox = cfg.bbox_local and cfg.refine_mode == "gradient"
    k_chunk = _resolve_k_chunk_size(
        K=cfg.random_samples,
        bbox_local=use_bbox,
        max_resolution=max(int(h), int(w)),
        vram_budget_gib=cfg.vram_budget_gib,
    )
    tr.snap(
        "02_config",
        bbox_local=cfg.bbox_local,
        use_bbox_path=use_bbox,
        k_chunk=k_chunk,
    )

    target = torch.from_numpy(rgb).to(device)
    tr.snap("03_target_gpu", shape=[h, w, 3])

    canvas = torch.full((h, w, 3), 40, dtype=torch.uint8, device=device)
    tr.snap("04_canvas_gpu")

    gen = torch.Generator(device="cpu").manual_seed(0)
    kind = KINDS["rotated_ellipse"]

    try:
        _trace_scoring(
            tr,
            use_bbox=use_bbox,
            kind=kind,
            canvas=canvas,
            target=target,
            w=w,
            h=h,
            K=K,
            k_chunk=k_chunk,
            max_chunks=max_chunks,
            gen=gen,
        )
    except torch.cuda.OutOfMemoryError as exc:
        tr.snap("OOM", error=str(exc)[:200])
        print(f"OOM during trace: {exc}", file=sys.stderr)

    tr.snap("99_end")
    if out_path:
        md = out_path.with_suffix(".md")
        tr.write_summary_md(md)
        print(f"Wrote {out_path} and {md}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true", help="Per-step VRAM trace")
    parser.add_argument("--run-one-shape", action="store_true", help="Legacy short trace")
    parser.add_argument("--K", type=int, default=8192)
    parser.add_argument("--res", type=int, default=720)
    parser.add_argument("--budget", type=float, default=8.0)
    parser.add_argument(
        "--bbox-local", action="store_true",
        help="Use bbox_local=True (EXE panel assumption)",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=3,
        help="Stop after N chunks to avoid OOM during profiling",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSONL output path (default: smoke_results_upstream/vram_step_trace.jsonl)",
    )
    parser.add_argument(
        "--both-paths", action="store_true",
        help="Run trace twice: smoke default (full_canvas) then bbox_local",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("torch not importable — use embedded python.exe", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 1

    if args.trace:
        base = Path(__file__).resolve().parent / "smoke_results_upstream"
        base.mkdir(parents=True, exist_ok=True)
        if args.both_paths:
            rc = 0
            for tag, bbox in (("full_canvas_smoke_default", False), ("bbox_local_exe_panel", True)):
                out = base / f"vram_step_trace_{tag}.jsonl"
                print(f"\n=== TRACE {tag} bbox_local={bbox} ===", file=sys.stderr)
                rc |= _run_trace(
                    K=args.K, res=args.res, budget=args.budget,
                    bbox_local=bbox, max_chunks=args.max_chunks, out_path=out,
                )
            return rc
        out = args.out or (base / "vram_step_trace.jsonl")
        return _run_trace(
            K=args.K, res=args.res, budget=args.budget,
            bbox_local=args.bbox_local,
            max_chunks=args.max_chunks,
            out_path=out,
        )

    # Legacy --run-one-shape: planner + short path
    _planner_table(args.K, args.res, args.budget)
    if not args.run_one_shape:
        print("Use --trace for full per-step profile", file=sys.stderr)
        return 0
    return _run_trace(
        K=args.K, res=args.res, budget=args.budget,
        bbox_local=False, max_chunks=1, out_path=None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
