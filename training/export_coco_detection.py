from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from export_yolo_detection import MASK_TYPE_TO_IDX, assigned_split, load_manifest, rel_path, row_path, to_uint8_image


NAMES = ["spot", "line", "region", "gradient", "band"]


def export_coco(
    rows: list[dict[str, Any]],
    data_root: Path,
    out_root: Path,
    split_mode: str,
    include_empty: bool,
    boxes_key: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split_name in ["train", "val", "test"]:
        images = []
        annotations = []
        ann_id = 1
        image_id = 1
        image_dir = out_root / "images" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            if assigned_split(row, split_mode) != split_name:
                continue
            box_path = rel_path(data_root, row_path(row, boxes_key, "boxes", "box_json", "box"))
            with open(box_path, encoding="utf-8") as f:
                box_data = json.load(f)
            boxes = box_data.get("boxes", [])
            if not boxes and not include_empty:
                continue

            image_path = rel_path(data_root, row_path(row, "image", "image_npy"))
            arr = np.load(image_path)
            image_u8 = to_uint8_image(arr)
            h, w = image_u8.shape[:2]
            file_name = f"{row['panel_id']}.png"
            Image.fromarray(image_u8, mode="L").save(image_dir / file_name)
            images.append({"id": image_id, "file_name": file_name, "width": int(w), "height": int(h)})

            for box in boxes:
                mask_type = str(box.get("mask_type", ""))
                if mask_type not in MASK_TYPE_TO_IDX:
                    continue
                x0, y0, x1, y1 = [float(v) for v in box["bbox"]]
                bw = max(0.0, x1 - x0)
                bh = max(0.0, y1 - y0)
                if bw <= 0 or bh <= 0:
                    continue
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": int(MASK_TYPE_TO_IDX[mask_type]) + 1,
                        "bbox": [x0, y0, bw, bh],
                        "area": float(bw * bh),
                        "iscrowd": 0,
                        "attributes": {
                            "instance_id": box.get("instance_id"),
                            "extent": box.get("extent"),
                            "subtype": box.get("subtype"),
                            "polarity": box.get("polarity"),
                            "semu": box.get("semu"),
                            "is_reject": box.get("is_reject"),
                        },
                    }
                )
                ann_id += 1
            image_id += 1

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": idx + 1, "name": name} for idx, name in enumerate(NAMES)],
        }
        ann_dir = out_root / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        (ann_dir / f"instances_{split_name}.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
        summary[split_name] = {"images": len(images), "annotations": len(annotations)}
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="data/mura_rca_coco_det")
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
    summary = export_coco(load_manifest(data_root), data_root, out_root, args.split_mode, args.include_empty, args.boxes_key)
    (out_root / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
