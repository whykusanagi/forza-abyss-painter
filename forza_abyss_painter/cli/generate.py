"""`fap-generate` — headless GPU shape-gen CLI.

Lets users / scripts trigger a GPU run without going through the EXE
GUI. Built for batch / overnight automation: feed it an image + the
same knobs the GUI exposes, get a JSON file out.

## Why this exists

The EXE's `Tools → Generate shapes locally (GPU)…` menu is the
interactive flow. For overnight queues, scripted A/B tests, and
"render 50 source images while I sleep" use cases, the GUI is the
wrong tool. Painter-fh6's bundled OpenCL tool exposes `-settings` +
`-profile`; this is our equivalent.

## Usage

    fap-generate --image SOURCE.png --output OUT.json \\
                 --num-shapes 1000 --max-resolution 720 \\
                 --random-samples 8192 \\
                 [--vram-budget 8] [--sticker] [--bbox-local | --full-canvas]

Prints progress to stderr (same JSON-line schema torch_runner emits
over IPC). Final exit code 0 on success, 1 on engine error, 2 on
config error, 3 on missing torch.

## Implementation note

Wraps `forza_abyss_painter.runtime.torch_runner.run()` IN-PROCESS —
not via subprocess. The CLI's whole point is to be the entry point,
so spawning a subprocess to do the work would be a layer of
redundant indirection. The runner's `run()` function is reentrant
enough to call directly; it installs the same signal handlers either
way.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fap-generate",
        description=(
            "Headless GPU shape-generation. Runs the same engine the EXE "
            "uses (forza_abyss_painter.runtime.torch_runner), without the "
            "GUI. Requires the local GPU runtime to be installed — install "
            "it from the EXE first (Tools → Install GPU runtime), or "
            "ensure your system Python has torch + CUDA available."
        ),
    )
    parser.add_argument(
        "--image", required=True, type=Path,
        help="path to source image (PNG/JPG/WEBP)",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="path to write the generated JSON",
    )
    parser.add_argument(
        "--num-shapes", type=int, default=1000,
        help="how many shapes to commit (default: 1000)",
    )
    parser.add_argument(
        "--max-resolution", type=int, default=720,
        help="long-side resolution the engine works at (default: 720)",
    )
    parser.add_argument(
        "--random-samples", type=int, default=8192,
        help="per-shape random candidate count K (default: 8192). "
             "Higher = better quality + more VRAM. Use --vram-budget "
             "to cap.",
    )
    parser.add_argument(
        "--sticker", action="store_true",
        help="treat the source image as a sticker (preserve transparent "
             "pixels via alpha mask). Default: composite onto white.",
    )
    parser.add_argument(
        "--vram-budget", type=float, default=0.0,
        help="GPU VRAM budget in GiB. 0 = unlimited (single full-K pass). "
             "When set, engine auto-chunks the K-batch to fit. Higher = "
             "faster (single pass) but uses more VRAM; lower = slower "
             "(multiple chunks) but stays under budget.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=0,
        help="emit a 'checkpoint' progress event every N shapes "
             "(default: num_shapes // 20). 0 = no checkpoints.",
    )
    parser.add_argument(
        "--preset-label", default="custom",
        help="human-readable preset label (echoed in 'started' event). "
             "Cosmetic; doesn't affect generation.",
    )
    parser.add_argument(
        "--device", default="cuda", choices=("cuda", "cpu"),
        help="torch device. 'cpu' is for testing only — generation is "
             "absurdly slow without GPU.",
    )
    bbox_mode = parser.add_mutually_exclusive_group()
    bbox_mode.add_argument(
        "--bbox-local", dest="bbox_local", action="store_true",
        default=True,
        help="(default) Use bbox-local scoring — small per-candidate "
             "crops, low VRAM peak. Required for ellipse-only runs on "
             "consumer GPUs. See vram_planner.py for the math.",
    )
    bbox_mode.add_argument(
        "--full-canvas", dest="bbox_local", action="store_false",
        help="Use full-canvas scoring — per-candidate masks cover the "
             "whole canvas. Required for multi-shape (rectangle, "
             "triangle) but OOMs on 32 GiB GPUs at K > 1024 without "
             "chunked rasterize (#129). Use for diagnostic A/B against "
             "the bbox-local baseline (Cursor's T1.2).",
    )
    return parser


def main(argv: "list[str] | None" = None) -> int:
    """fap-generate entry point. Returns exit code: 0 success, 1 engine
    error, 2 config error, 3 import error (torch missing)."""
    args = _build_parser().parse_args(argv)

    # Validate image exists BEFORE invoking torch_runner so we get a
    # clean CLI-level error message instead of a deep traceback.
    if not args.image.exists():
        print(f"fap-generate: source image not found: {args.image}",
              file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Default checkpoint_every: 20 progress events across the run.
    checkpoint_every = args.checkpoint_every
    if checkpoint_every <= 0:
        checkpoint_every = max(1, args.num_shapes // 20)

    # Build the RunConfig dict the runner consumes. Matches the schema
    # in torch_runner.RunConfig.from_dict — any new field added there
    # needs surfacing here too.
    config_dict = {
        "image_path": str(args.image),
        "output_json_path": str(args.output),
        "num_shapes": int(args.num_shapes),
        "max_resolution": int(args.max_resolution),
        "random_samples": int(args.random_samples),
        "sticker_mode": bool(args.sticker),
        "checkpoint_every": int(checkpoint_every),
        "vram_budget_gib": float(args.vram_budget),
        "lock_alpha": True,   # hard system constraint per CLAUDE.md §3
        "preset_label": str(args.preset_label),
        "device": str(args.device),
        "bbox_local": bool(args.bbox_local),
    }

    try:
        from forza_abyss_painter.runtime.torch_runner import RunConfig, run
    except ImportError as exc:
        print(
            f"fap-generate: couldn't import torch_runner — typically means "
            f"the GPU runtime isn't installed in this Python.\n  {exc}\n"
            f"From the EXE: Tools → Install GPU runtime.\n"
            f"From a dev environment: pip install torch (+ matching CUDA).",
            file=sys.stderr,
        )
        return 3

    try:
        cfg = RunConfig.from_dict(config_dict)
    except ValueError as exc:
        print(f"fap-generate: invalid config: {exc}", file=sys.stderr)
        return 2

    # run() emits its JSON event stream on stderr by default, which is
    # exactly what the user wants for a CLI (progress visible, doesn't
    # interfere with stdout if they're piping output elsewhere).
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
