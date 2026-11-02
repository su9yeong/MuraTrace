from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from mura_rca_data import MASK_TYPE_TO_IDX, assigned_split, load_manifest, rel_path
except ModuleNotFoundError:
    MASK_TYPES = ["spot", "line", "region", "gradient", "band"]
    MASK_TYPE_TO_IDX = {v: i for i, v in enumerate(MASK_TYPES)}

    def rel_path(root: Path, value: str) -> Path:
        return root / str(value).replace("\\", "/")

    def load_manifest(root: Path) -> list[dict[str, Any]]:
        rows = []
        with open(root / "manifest.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def assigned_split(row: dict[str, Any], split_mode: str, train_ratio: float = 0.6, val_ratio: float = 0.2) -> str:
        if split_mode == "manifest":
            return str(row["split"])
        import hashlib

        key = str(row["panel_id"] if split_mode == "random_panel" else row["glass_id"])
        u = (int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % 1_000_000) / 1_000_000.0
        if u < train_ratio:
            return "train"
        if u < train_ratio + val_ratio:
            return "val"
        return "test"


def row_path(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    raise KeyError(f"manifest row is missing all path keys: {keys}")


def to_uint8_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if np.nanmax(arr) <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def export_split(
    rows: list[dict[str, Any]],
    data_root: Path,
    out_root: Path,
    split_mode: str,
    include_empty: bool,
    boxes_key: str,
) -> dict[str, int]:
    counts = {"images": 0, "labels": 0, "boxes": 0}
    for row in rows:
        split = assigned_split(row, split_mode)
        image_path = rel_path(data_root, row_path(row, "image", "image_npy"))
        box_path = rel_path(data_root, row_path(row, boxes_key, "boxes", "box_json", "box"))
        image_out_dir = out_root / "images" / split
        label_out_dir = out_root / "labels" / split
        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)

        with open(box_path, encoding="utf-8") as f:
            box_data = json.load(f)
        boxes = box_data.get("boxes", [])
        if not boxes and not include_empty:
            continue

        arr = np.load(image_path)
        image_u8 = to_uint8_image(arr)
        panel_id = str(row["panel_id"])
        out_image = image_out_dir / f"{panel_id}.png"
        Image.fromarray(image_u8, mode="L").save(out_image)

        label_lines = []
        for box in boxes:
            mask_type = str(box.get("mask_type", ""))
            if mask_type not in MASK_TYPE_TO_IDX:
                continue
            cls = MASK_TYPE_TO_IDX[mask_type]
            yolo = box.get("yolo")
            if not yolo or len(yolo) != 4:
                continue
            x, y, w, h = [float(v) for v in yolo]
            if w <= 0 or h <= 0:
                continue
            label_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            counts["boxes"] += 1

        out_label = label_out_dir / f"{panel_id}.txt"
        out_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        counts["images"] += 1
        counts["labels"] += 1
    return counts


def write_yaml(out_root: Path) -> None:
    names = ["spot", "line", "region", "gradient", "band"]
    lines = [
        f"path: {out_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(names)}",
        "names:",
    ]
    lines += [f"  {idx}: {name}" for idx, name in enumerate(names)]
    (out_root / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="data/mura_rca_yolo_det")
    ap.add_argument("--split_mode", default="manifest", choices=["manifest", "random_panel", "random_glass"])
    ap.add_argument("--include_empty", action="store_true")
    ap.add_argument("--boxes_key", default="train_boxes", help="manifest boxes key to export; falls back to boxes")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_dir)
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(data_root)
    counts = export_split(rows, data_root, out_root, args.split_mode, args.include_empty, args.boxes_key)
    write_yaml(out_root)
    (out_root / "export_summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
