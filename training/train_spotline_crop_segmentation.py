from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

from mura_rca_data import assigned_split, load_manifest, normalize_image, rel_path, stable_hash_int
from mura_rca_models import build_segmentation_model
from train_segmentation import metrics_from_confusion, run_epoch, write_json


SPOT_LINE_IDS = {1, 2}


def row_path(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    raise KeyError(f"manifest row is missing all path keys: {keys}")


def crop_box_from_bbox(
    bbox: list[float],
    width: int,
    height: int,
    crop_scale: float,
    min_side: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = max(float(min_side), max(bw, bh) * float(crop_scale))
    nx0 = int(math.floor(cx - side / 2.0))
    ny0 = int(math.floor(cy - side / 2.0))
    nx1 = int(math.ceil(cx + side / 2.0))
    ny1 = int(math.ceil(cy + side / 2.0))
    nx0 = max(0, min(nx0, max(width - 1, 0)))
    ny0 = max(0, min(ny0, max(height - 1, 0)))
    nx1 = max(nx0 + 1, min(nx1, width))
    ny1 = max(ny0 + 1, min(ny1, height))
    return nx0, ny0, nx1, ny1


class SpotLineCropDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 256,
        split_mode: str = "manifest",
        crop_scale: float = 5.0,
        min_side: int = 96,
        negative_ratio: float = 0.35,
        mask_key: str = "train_mask",
        boxes_key: str = "train_boxes",
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = int(image_size)
        self.crop_scale = float(crop_scale)
        self.min_side = int(min_side)
        self.mask_key = str(mask_key)
        self.boxes_key = str(boxes_key)
        rows = []
        positives: list[tuple[int, int]] = []
        for row in load_manifest(self.root):
            if assigned_split(row, split_mode) != split:
                continue
            row_idx = len(rows)
            rows.append(row)
            with open(rel_path(self.root, row_path(row, self.boxes_key, "boxes", "box_json", "box")), encoding="utf-8") as f:
                boxes = list(json.load(f).get("boxes", []))
            for box_idx, box in enumerate(boxes):
                if str(box.get("mask_type", "")) in {"spot", "line"}:
                    positives.append((row_idx, box_idx))
        if limit:
            positives = positives[: int(limit)]
        self.rows = rows
        self.samples: list[tuple[int, int]] = list(positives)
        n_neg = int(round(len(positives) * float(negative_ratio)))
        if rows and n_neg > 0:
            order = sorted(range(len(rows)), key=lambda i: stable_hash_int(str(rows[i].get("panel_id", i))))
            for i in range(n_neg):
                self.samples.append((order[i % len(order)], -1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row_idx, box_idx = self.samples[idx]
        row = self.rows[row_idx]
        image = normalize_image(np.load(rel_path(self.root, row_path(row, "image", "image_npy"))))
        mask = np.load(rel_path(self.root, row_path(row, self.mask_key, "mask", "mask_npy"))).astype(np.int64)
        h, w = image.shape[:2]

        if box_idx >= 0:
            with open(rel_path(self.root, row_path(row, self.boxes_key, "boxes", "box_json", "box")), encoding="utf-8") as f:
                box = list(json.load(f).get("boxes", []))[box_idx]
            x0, y0, x1, y1 = crop_box_from_bbox(box["bbox"], w, h, self.crop_scale, self.min_side)
        else:
            side = min(max(self.min_side, self.image_size), max(w, h))
            key = stable_hash_int(f"{row.get('panel_id')}:{idx}")
            max_x = max(w - side, 0)
            max_y = max(h - side, 0)
            x0 = int(key % (max_x + 1)) if max_x > 0 else 0
            y0 = int((key // 997) % (max_y + 1)) if max_y > 0 else 0
            x1 = min(w, x0 + side)
            y1 = min(h, y0 + side)

        crop_img = image[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
        crop_mask = np.where(np.isin(crop_mask, list(SPOT_LINE_IDS)), crop_mask, 0).astype(np.uint8)

        img_u8 = np.clip(crop_img * 255.0 + 0.5, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_u8, mode="L").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        pil_mask = Image.fromarray(crop_mask, mode="L").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        x = torch.from_numpy(np.asarray(pil_img, dtype=np.float32) / 255.0)[None]
        y = torch.from_numpy(np.asarray(pil_mask, dtype=np.int64)).long()
        return {
            "image": x,
            "mask": y,
            "panel_id": row["panel_id"],
            "glass_id": row["glass_id"],
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="runs/spotline_crop_deeplabv3plus")
    ap.add_argument("--model", default="deeplabv3plus", choices=["unet", "deeplabv3plus", "hrnet", "smp_unet"])
    ap.add_argument("--encoder_name", default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--crop_scale", type=float, default=5.0)
    ap.add_argument("--min_side", type=int, default=96)
    ap.add_argument("--negative_ratio", type=float, default=0.35)
    ap.add_argument("--mask_key", default="train_mask", help="manifest mask key for crop target; falls back to mask")
    ap.add_argument("--boxes_key", default="train_boxes", help="manifest boxes key for crop positives; falls back to boxes")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--split_mode", default="manifest", choices=["manifest", "random_panel", "random_glass"])
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)
    ap.add_argument("--focal_gamma", type=float, default=1.5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    reports_dir = out_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = SpotLineCropDataset(
        args.data_root,
        "train",
        image_size=args.image_size,
        split_mode=args.split_mode,
        crop_scale=args.crop_scale,
        min_side=args.min_side,
        negative_ratio=args.negative_ratio,
        mask_key=args.mask_key,
        boxes_key=args.boxes_key,
        limit=args.limit_train or None,
    )
    val_ds = SpotLineCropDataset(
        args.data_root,
        "val",
        image_size=args.image_size,
        split_mode=args.split_mode,
        crop_scale=args.crop_scale,
        min_side=args.min_side,
        negative_ratio=args.negative_ratio,
        mask_key=args.mask_key,
        boxes_key=args.boxes_key,
        limit=args.limit_val or None,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = build_segmentation_model(
        args.model,
        num_classes=6,
        in_ch=1,
        image_size=args.image_size,
        encoder_name=args.encoder_name,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)
    class_weights = torch.tensor([0.25, 12.0, 8.0, 0.05, 0.05, 0.05], dtype=torch.float32)
    write_json(
        out_dir / "run_config.json",
        {
            "args": vars(args),
            "device": str(device),
            "train_crops": len(train_ds),
            "val_crops": len(val_ds),
            "class_weights": class_weights.tolist(),
        },
    )
    best = -1.0
    with open(out_dir / "metrics.jsonl", "w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            train_metrics, train_report = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                scaler,
                args.amp,
                grad_clip=args.grad_clip,
                class_weights=class_weights,
                focal_gamma=args.focal_gamma,
                dice_weight=1.0,
            )
            val_metrics, val_report = run_epoch(
                model,
                val_loader,
                None,
                device,
                None,
                args.amp,
                class_weights=class_weights,
                focal_gamma=args.focal_gamma,
                dice_weight=1.0,
            )
            row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row, indent=2), flush=True)
            write_json(reports_dir / f"train_epoch_{epoch:04d}.json", train_report)
            write_json(reports_dir / f"val_epoch_{epoch:04d}.json", val_report)
            score = 0.5 * float(val_metrics.get("iou_1", 0.0)) + 0.5 * float(val_metrics.get("iou_2", 0.0))
            ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args), "score": score, "val": val_metrics}
            torch.save(ckpt, out_dir / "last.pt")
            if score > best:
                best = score
                torch.save(ckpt, out_dir / "best.pt")
                write_json(out_dir / "best_point.json", {"epoch": epoch, "score": score, "metric": "mean_iou_spot_line", "val_metrics": val_metrics})
                write_json(reports_dir / "val_best.json", val_report)

    # Keep this import path used enough for static checkers when reports are inspected.
    _ = metrics_from_confusion


if __name__ == "__main__":
    main()
