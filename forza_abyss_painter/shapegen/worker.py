from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QThread, Signal

from forza_abyss_painter.shapegen.engine import Engine, EngineConfig
from forza_abyss_painter.shapegen.profile import Profile
from forza_abyss_painter.io.exporter import save_json
from forza_abyss_painter.io.json_schema import FD6Document


def _pad_with_source_mean(
    img: Image.Image,
    *,
    sticker_mode: bool,
    alpha_mask: np.ndarray | None,
    buffer_frac: float = 0.08,
) -> tuple[Image.Image, np.ndarray | None]:
    """Pad img to a square + edge-buffer using the source image's mean RGB
    as the fill color. Replaces the previous hardcoded white fill that was
    causing the shape-generator to waste budget covering a phantom white
    border around real content.

    Ports the v0.1.2 'opaque padding fill = source mean' fix that was
    shipped to the colab GPU pipeline (sister repo's
    fd6/shapegen/gpu/engine.py:407-408) into the EXE's CPU shape-gen path.

    Both padding steps (non-square → square, then edge buffer) use the
    SAME pre-computed src_mean — measuring the mean after the non-square
    pad would dilute the original colors toward the previous pad fill on
    every subsequent step.

    Sticker mode keeps black (0, 0, 0) padding because alpha_mask drives
    the real content boundary; the pad pixels get masked out of scoring
    anyway, so the fill color doesn't matter as long as it stays the
    upstream-compatible black.
    """
    if sticker_mode:
        fill: tuple[int, int, int] = (0, 0, 0)
    else:
        fill = tuple(
            int(c) for c in
            np.asarray(img, dtype=np.uint8).reshape(-1, 3)
            .mean(axis=0).round().clip(0, 255)
        )

    # Non-square pad (skipped in sticker mode — original behavior preserved
    # so alpha-driven content stays unpadded on this step).
    if not sticker_mode and img.size[0] != img.size[1]:
        side = max(img.size)
        square = Image.new("RGB", (side, side), fill)
        offset = ((side - img.size[0]) // 2, (side - img.size[1]) // 2)
        square.paste(img, offset)
        img = square

    # Edge-buffer pad (every generation, both modes).
    pad_px = max(8, int(round(max(img.size) * buffer_frac)))
    new_w = img.size[0] + 2 * pad_px
    new_h = img.size[1] + 2 * pad_px
    buffered = Image.new("RGB", (new_w, new_h), fill)
    buffered.paste(img, (pad_px, pad_px))
    img = buffered
    if sticker_mode and alpha_mask is not None:
        padded_alpha = np.zeros((new_h, new_w), dtype=np.uint8)
        src_h, src_w = alpha_mask.shape[:2]
        padded_alpha[pad_px:pad_px + src_h, pad_px:pad_px + src_w] = alpha_mask
        alpha_mask = padded_alpha
    return img, alpha_mask


class GenerationWorker(QObject):
    """Wraps Engine.run() in a QThread-friendly object. Emits Qt signals for the GUI."""

    progress = Signal(int, int, float)  # shape_count, total, rms
    preview = Signal(object)            # np.ndarray (H,W,3) uint8
    finished = Signal(str)              # final json output path
    error = Signal(str)
    checkpoint_written = Signal(str)    # checkpoint json path

    def __init__(self, image_path: Path, profile: Profile, output_dir: Path | None = None, sticker_mode: bool = False) -> None:
        super().__init__()
        self.image_path = Path(image_path)
        self.profile = profile
        self.output_dir = Path(output_dir) if output_dir else self.image_path.parent / self.image_path.stem
        self.sticker_mode = sticker_mode  # When True, keep source alpha and skip transparent areas
        self._engine: Engine | None = None
        self._paused = False

    def stop(self) -> None:
        if self._engine:
            self._engine.request_stop()

    def set_pause(self, paused: bool) -> None:
        self._paused = paused
        if self._engine:
            self._engine.set_pause(paused)

    def run(self) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            img = Image.open(self.image_path)
            alpha_mask: np.ndarray | None = None  # None = full opacity (treat all pixels equally)
            has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
            if has_alpha:
                rgba = img.convert("RGBA")
                if self.sticker_mode:
                    # Keep transparency: extract alpha mask, use RGB channels as target
                    # (transparent areas keep whatever RGB they had — we ignore them via the mask)
                    arr_rgba = np.asarray(rgba, dtype=np.uint8)
                    img = Image.fromarray(arr_rgba[:, :, :3], "RGB")
                    alpha_mask = arr_rgba[:, :, 3].copy()  # H x W, 0 = transparent, 255 = opaque
                else:
                    # Default: composite onto white to avoid leaking under-transparent RGB junk
                    bg = Image.new("RGB", rgba.size, (255, 255, 255))
                    bg.paste(rgba, mask=rgba.split()[3])
                    img = bg
            else:
                img = img.convert("RGB")
            # Source-mean padding: both the non-square square-up step AND the
            # 8%-per-side edge buffer fill with the source image's mean RGB
            # instead of hardcoded white. Stops the shape-generator from
            # wasting budget covering a phantom white border around real
            # content. Sticker mode keeps the (0,0,0) fill — alpha_mask
            # gates content so the pad color doesn't get scored.
            # See _pad_with_source_mean above for full reasoning + the
            # colab pipeline port reference.
            img, alpha_mask = _pad_with_source_mean(
                img,
                sticker_mode=self.sticker_mode,
                alpha_mask=alpha_mask,
                buffer_frac=0.08,
            )
            # Downscale to profile.max_resolution along the longer side.
            mr = self.profile.max_resolution
            if max(img.size) > mr:
                scale = mr / max(img.size)
                new_size = (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale)))
                img = img.resize(new_size, Image.LANCZOS)
                if alpha_mask is not None:
                    am_img = Image.fromarray(alpha_mask, "L").resize(new_size, Image.LANCZOS)
                    alpha_mask = np.asarray(am_img, dtype=np.uint8)
            target = np.asarray(img, dtype=np.uint8)

            self._engine = Engine(target, EngineConfig(profile=self.profile), alpha_mask=alpha_mask)
            stem = self.image_path.stem
            final_path = self.output_dir / f"{stem}.json"

            for event in self._engine.run():
                if event.kind == "shape_committed":
                    self.progress.emit(event.shape_count, self.profile.stop_at, event.rms)
                elif event.kind == "preview" and event.canvas is not None:
                    self.preview.emit(event.canvas)
                elif event.kind == "checkpoint":
                    cp_path = self.output_dir / f"{stem}_{event.shape_count}.json"
                    doc = FD6Document.from_engine(
                        source_image=self.image_path.name,
                        image_size=(target.shape[1], target.shape[0]),
                        shapes=self._engine.shapes,
                        profile_name=self.profile.name,
                        sticker_mode=self.sticker_mode,
                    )
                    save_json(doc, cp_path)
                    self.checkpoint_written.emit(str(cp_path))
                elif event.kind == "error":
                    self.error.emit(event.message)
                    return
                elif event.kind == "done":
                    doc = FD6Document.from_engine(
                        source_image=self.image_path.name,
                        image_size=(target.shape[1], target.shape[0]),
                        shapes=self._engine.shapes,
                        profile_name=self.profile.name,
                        sticker_mode=self.sticker_mode,
                    )
                    save_json(doc, final_path)
                    self.finished.emit(str(final_path))
                    return
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
