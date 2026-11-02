#!/usr/bin/env python3
"""Build a weakly augmented real-background bank for hybrid Mura data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _image_to_luma_float(img: Image.Image) -> np.ndarray:
    """Return grayscale luminance in [0, 1], preserving 16-bit PNGs when present."""
    if img.mode in {"I;16", "I;16L", "I;16B"}:
        arr = np.asarray(img, dtype=np.uint16).astype(np.float32)
        return arr / 65535.0
    if img.mode in {"I", "F"}:
        arr = np.asarray(img, dtype=np.float32)
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.float32)
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _resize_float_luma(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    img16 = Image.fromarray(np.clip(arr * 65535.0 + 0.5, 0, 65535).astype(np.uint16), mode="I;16")
    img16 = img16.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(img16, dtype=np.float32) / 65535.0


def crop_luma_edges(arr: np.ndarray, crop: dict[str, int] | None = None) -> np.ndarray:
    if not crop:
        return arr
    top = max(0, int(crop.get("top", 0)))
    bottom = max(0, int(crop.get("bottom", 0)))
    left = max(0, int(crop.get("left", 0)))
    right = max(0, int(crop.get("right", 0)))
    h, w = arr.shape[:2]
    y0 = min(top, h)
    y1 = max(y0, h - bottom)
    x0 = min(left, w)
    x1 = max(x0, w - right)
    out = arr[y0:y1, x0:x1]
    if out.size == 0:
        raise ValueError(f"crop removes the full image: shape={arr.shape}, crop={crop}")
    return out


def load_luma(path: Path, target_size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as img:
        arr = _image_to_luma_float(img)
    if target_size is not None and (arr.shape[1], arr.shape[0]) != target_size:
        arr = _resize_float_luma(arr, target_size)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def save_luma_png(arr: np.ndarray, path: Path) -> None:
    out = np.clip(arr * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    Image.fromarray(out, mode="I;16").save(path)


def lowfreq_1d(length: int, knots: int, rng: np.random.Generator) -> np.ndarray:
    knots = max(3, int(knots))
    xp = np.linspace(0, length - 1, knots)
    fp = rng.normal(0.0, 1.0, knots)
    vec = np.interp(np.arange(length), xp, fp).astype(np.float32)
    vec -= float(vec.mean())
    peak = float(np.max(np.abs(vec)))
    return vec / peak if peak > 1e-8 else vec


def blur_float(arr: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 1e-4:
        return arr
    sigma = float(radius)
    kr = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-kr, kr + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    kernel /= float(kernel.sum())

    tmp = np.zeros_like(arr, dtype=np.float32)
    padded = np.pad(arr.astype(np.float32), ((kr, kr), (0, 0)), mode="edge")
    for idx, weight in enumerate(kernel):
        tmp += float(weight) * padded[idx : idx + arr.shape[0], :]

    out = np.zeros_like(arr, dtype=np.float32)
    padded = np.pad(tmp, ((0, 0), (kr, kr)), mode="edge")
    for idx, weight in enumerate(kernel):
        out += float(weight) * padded[:, idx : idx + arr.shape[1]]
    return out


def augment_background(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply only weak appearance changes; no rotation, flip, warp, or strong color jitter."""
    out = arr.astype(np.float32, copy=True)
    mean = float(out.mean())

    gain = float(rng.uniform(0.975, 1.025))
    bias = float(rng.uniform(-0.010, 0.010))
    out = out * gain + bias

    contrast = float(rng.uniform(0.95, 1.05))
    mean = float(out.mean())
    out = mean + (out - mean) * contrast
    out = np.clip(out, 0.0, 1.0)

    gamma = float(rng.uniform(0.965, 1.035))
    out = np.power(np.clip(out, 1e-6, 1.0), gamma)

    h, w = out.shape
    if rng.random() < 0.35:
        row_amp = float(rng.uniform(0.0003, 0.0015))
        row = lowfreq_1d(h, int(rng.integers(4, 8)), rng)[:, None] * row_amp
        out = out + row
    if rng.random() < 0.30:
        col_amp = float(rng.uniform(0.0003, 0.0012))
        col = lowfreq_1d(w, int(rng.integers(4, 9)), rng)[None, :] * col_amp
        out = out + col

    noise_sigma = float(rng.uniform(0.0008, 0.0030))
    out = out + rng.normal(0.0, noise_sigma, out.shape).astype(np.float32)

    if rng.random() < 0.75:
        out = blur_float(out, float(rng.uniform(0.05, 0.35)))

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def make_grid(image_paths: list[Path], out_path: Path, cols: int = 7, max_images: int = 35) -> None:
    if not image_paths:
        return
    chosen = image_paths[:max_images]
    first = load_luma(chosen[0])
    aspect = first.shape[1] / max(first.shape[0], 1)
    thumb_w = 220
    thumb_h = max(64, int(round(thumb_w / aspect)))
    rows = int(np.ceil(len(chosen) / cols))
    grid = Image.new("L", (cols * thumb_w, rows * thumb_h), color=18)
    for i, path in enumerate(chosen):
        arr = load_luma(path)
        img = Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="L")
        img = ImageOps.fit(img, (thumb_w, thumb_h), method=Image.Resampling.BICUBIC)
        x = (i % cols) * thumb_w
        y = (i // cols) * thumb_h
        grid.paste(img, (x, y))
    ensure_dir(out_path.parent)
    grid.save(out_path)


def generate_background_bank(
    input_dir: Path,
    out_dir: Path,
    per_source: int = 30,
    seed: int = 0,
    target_size: tuple[int, int] | None = None,
    preview_per_source: int = 4,
    crop_edges: dict[str, int] | None = None,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    clean_dir = out_dir / "backgrounds_clean"
    aug_dir = out_dir / "backgrounds_aug"
    preview_dir = out_dir / "previews"
    ensure_dir(clean_dir)
    ensure_dir(aug_dir)
    ensure_dir(preview_dir)

    source_paths = list_images(input_dir)
    if not source_paths:
        raise FileNotFoundError(f"no image files found in {input_dir}")

    if target_size is None:
        with Image.open(source_paths[0]) as first:
            first_arr = _image_to_luma_float(first)
        first_arr = crop_luma_edges(first_arr, crop_edges)
        target_size = (int(first_arr.shape[1]), int(first_arr.shape[0]))

    rng = np.random.default_rng(seed)
    clean_records: list[dict[str, Any]] = []
    aug_records: list[dict[str, Any]] = []
    preview_paths: list[Path] = []

    for source_index, src in enumerate(source_paths):
        source_id = f"bg{source_index:02d}"
        with Image.open(src) as raw:
            original_size = list(raw.size)
            clean_raw = _image_to_luma_float(raw)
        clean_cropped = crop_luma_edges(clean_raw, crop_edges)
        cropped_size = [int(clean_cropped.shape[1]), int(clean_cropped.shape[0])]
        clean = clean_cropped
        if (clean.shape[1], clean.shape[0]) != target_size:
            clean = _resize_float_luma(clean, target_size)
        clean_path = clean_dir / f"{source_id}_clean.png"
        save_luma_png(clean, clean_path)
        clean_records.append(
            {
                "source_id": source_id,
                "source_index": source_index,
                "source_file": str(src),
                "clean_path": str(clean_path),
                "original_size": original_size,
                "crop_edges": crop_edges or {"top": 0, "bottom": 0, "left": 0, "right": 0},
                "cropped_size": cropped_size,
                "target_size": list(target_size),
            }
        )
        preview_paths.append(clean_path)

        for aug_index in range(per_source):
            aug_id = f"{source_id}_aug{aug_index:03d}"
            aug = augment_background(clean, rng)
            aug_path = aug_dir / f"{aug_id}.png"
            save_luma_png(aug, aug_path)
            record = {
                "source_id": source_id,
                "source_index": source_index,
                "source_file": str(src),
                "clean_path": str(clean_path),
                "aug_id": aug_id,
                "aug_index": aug_index,
                "aug_path": str(aug_path),
                "crop_edges": crop_edges or {"top": 0, "bottom": 0, "left": 0, "right": 0},
                "cropped_size": cropped_size,
                "target_size": list(target_size),
            }
            aug_records.append(record)
            if aug_index < preview_per_source:
                preview_paths.append(aug_path)

    with open(out_dir / "backgrounds_manifest.jsonl", "w", encoding="utf-8") as f:
        for row in aug_records:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    make_grid(
        preview_paths,
        preview_dir / "background_augmentation_preview_grid.png",
        cols=max(1, min(7, preview_per_source + 1)),
        max_images=len(preview_paths),
    )

    return {
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "per_source": per_source,
        "seed": seed,
        "crop_edges": crop_edges or {"top": 0, "bottom": 0, "left": 0, "right": 0},
        "target_size": list(target_size),
        "clean": clean_records,
        "augmented": aug_records,
        "preview": str(preview_dir / "background_augmentation_preview_grid.png"),
    }


def parse_size(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT")
    return int(parts[0]), int(parts[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="real_bg_7_fixedcrop", help="folder with clean cropped backgrounds")
    ap.add_argument("--out", default="data/hybrid_mura", help="hybrid dataset root")
    ap.add_argument("--per_source", type=int, default=30, help="augmentations per source background")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target_size", type=parse_size, default=None, help="optional WIDTHxHEIGHT resize")
    ap.add_argument("--preview_per_source", type=int, default=4)
    ap.add_argument("--crop_top", type=int, default=0)
    ap.add_argument("--crop_bottom", type=int, default=0)
    ap.add_argument("--crop_left", type=int, default=0)
    ap.add_argument("--crop_right", type=int, default=0)
    a = ap.parse_args()

    crop_edges = {"top": a.crop_top, "bottom": a.crop_bottom, "left": a.crop_left, "right": a.crop_right}
    result = generate_background_bank(
        input_dir=Path(a.input),
        out_dir=Path(a.out),
        per_source=a.per_source,
        seed=a.seed,
        target_size=a.target_size,
        preview_per_source=a.preview_per_source,
        crop_edges=crop_edges,
    )
    print(
        f"wrote {len(result['clean'])} clean backgrounds and "
        f"{len(result['augmented'])} augmented backgrounds to {a.out}"
    )
    print(f"preview: {result['preview']}")


if __name__ == "__main__":
    main()
