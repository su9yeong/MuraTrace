from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

from mura_rca_data import MASK_TYPE_TO_IDX, assigned_split, load_manifest, normalize_image, rel_path, stable_hash_int
from train_maskrcnn import collate_detection, move_targets, run_train_epoch, run_val_loss, write_json


NUM_CLASSES = 6


def row_path(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    raise KeyError(f"manifest row is missing all path keys: {keys}")


def intersects(box: list[float], tile: tuple[int, int, int, int], min_area: float = 1.0) -> bool:
    x0, y0, x1, y1 = [float(v) for v in box]
    tx0, ty0, tx1, ty1 = tile
    ix0 = max(x0, tx0)
    iy0 = max(y0, ty0)
    ix1 = min(x1, tx1)
    iy1 = min(y1, ty1)
    return (ix1 - ix0) * (iy1 - iy0) >= min_area


class TiledPanelInstanceDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        split_mode: str = "manifest",
        tile_size: int = 384,
        overlap: float = 0.35,
        include_empty_tiles: bool = True,
        empty_tile_fraction: float = 0.04,
        focus_types: set[str] | None = None,
        boxes_key: str = "train_boxes",
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.tile_size = int(tile_size)
        self.focus_types = focus_types
        self.boxes_key = str(boxes_key)
        rows = []
        for row in load_manifest(self.root):
            if assigned_split(row, split_mode) == split:
                rows.append(row)
        if limit:
            rows = rows[: int(limit)]
        self.rows = rows
        self.tiles: list[tuple[int, int, int, int, int]] = []
        for row_idx, row in enumerate(self.rows):
            with open(rel_path(self.root, row_path(row, self.boxes_key, "boxes", "box_json", "box")), encoding="utf-8") as f:
                boxes = list(json.load(f).get("boxes", []))
            if self.focus_types:
                boxes = [b for b in boxes if str(b.get("mask_type", "")) in self.focus_types]
            width, height = [int(v) for v in row.get("panel_size", [0, 0])]
            if width <= 0 or height <= 0:
                sample = normalize_image(np.load(rel_path(self.root, row_path(row, "image", "image_npy"))))
                height, width = sample.shape[:2]
            stride = max(1, int(round(self.tile_size * (1.0 - float(overlap)))))
            xs = list(range(0, max(width - self.tile_size, 0) + 1, stride))
            ys = list(range(0, max(height - self.tile_size, 0) + 1, stride))
            if not xs or xs[-1] != max(width - self.tile_size, 0):
                xs.append(max(width - self.tile_size, 0))
            if not ys or ys[-1] != max(height - self.tile_size, 0):
                ys.append(max(height - self.tile_size, 0))
            for y0 in ys:
                for x0 in xs:
                    x1 = min(width, x0 + self.tile_size)
                    y1 = min(height, y0 + self.tile_size)
                    tile = (x0, y0, x1, y1)
                    has_target = any(intersects(list(b["bbox"]), tile, min_area=2.0) for b in boxes)
                    if has_target:
                        self.tiles.append((row_idx, x0, y0, x1, y1))
                    elif include_empty_tiles:
                        key = f"{row.get('panel_id')}:{x0}:{y0}"
                        u = (stable_hash_int(key) % 1_000_000) / 1_000_000.0
                        if u < float(empty_tile_fraction):
                            self.tiles.append((row_idx, x0, y0, x1, y1))

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        row_idx, tx0, ty0, tx1, ty1 = self.tiles[idx]
        row = self.rows[row_idx]
        image = normalize_image(np.load(rel_path(self.root, row_path(row, "image", "image_npy"))))
        inst = np.load(rel_path(self.root, row_path(row, "inst", "instance", "instance_map"))).astype(np.uint16)
        with open(rel_path(self.root, row_path(row, self.boxes_key, "boxes", "box_json", "box")), encoding="utf-8") as f:
            boxes = list(json.load(f).get("boxes", []))

        tile_img = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
        tile_inst = np.zeros((self.tile_size, self.tile_size), dtype=np.uint16)
        crop_img = image[ty0:ty1, tx0:tx1]
        crop_inst = inst[ty0:ty1, tx0:tx1]
        tile_img[: crop_img.shape[0], : crop_img.shape[1]] = crop_img
        tile_inst[: crop_inst.shape[0], : crop_inst.shape[1]] = crop_inst

        target_boxes = []
        labels = []
        masks = []
        areas = []
        for box in boxes:
            mask_type = str(box.get("mask_type", ""))
            if self.focus_types and mask_type not in self.focus_types:
                continue
            if mask_type not in MASK_TYPE_TO_IDX:
                continue
            instance_id = int(box.get("instance_id", -1))
            mask = tile_inst == instance_id
            if not np.any(mask):
                continue
            ys, xs = np.where(mask)
            x0 = float(xs.min())
            y0 = float(ys.min())
            x1 = float(xs.max() + 1)
            y1 = float(ys.max() + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            target_boxes.append([x0, y0, x1, y1])
            labels.append(int(MASK_TYPE_TO_IDX[mask_type]) + 1)
            masks.append(mask.astype(np.uint8))
            areas.append((x1 - x0) * (y1 - y0))

        image_tensor = torch.from_numpy(tile_img)[None].repeat(3, 1, 1)
        if target_boxes:
            boxes_tensor = torch.tensor(target_boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            masks_tensor = torch.from_numpy(np.stack(masks, axis=0)).to(torch.uint8)
            areas_tensor = torch.tensor(areas, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            masks_tensor = torch.zeros((0, self.tile_size, self.tile_size), dtype=torch.uint8)
            areas_tensor = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": areas_tensor,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        return image_tensor, target


def build_tiled_maskrcnn(pretrained: bool = False) -> torch.nn.Module:
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    if pretrained:
        try:
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

            weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        except Exception:
            weights = "DEFAULT"
        model = maskrcnn_resnet50_fpn(weights=weights)
    else:
        model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((4, 8, 16), (8, 16, 32), (16, 32, 64), (32, 64, 128), (64, 128, 256)),
        aspect_ratios=((0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 20.0),) * 5,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, NUM_CLASSES)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="runs/maskrcnn_tiled")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--tile_size", type=int, default=384)
    ap.add_argument("--overlap", type=float, default=0.35)
    ap.add_argument("--empty_tile_fraction", type=float, default=0.04)
    ap.add_argument("--focus_spot_line", action="store_true")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--split_mode", default="manifest", choices=["manifest", "random_panel", "random_glass"])
    ap.add_argument("--boxes_key", default="train_boxes", help="manifest boxes key for training; falls back to boxes")
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    focus_types = {"spot", "line"} if args.focus_spot_line else None
    train_ds = TiledPanelInstanceDataset(
        args.data_root,
        "train",
        split_mode=args.split_mode,
        tile_size=args.tile_size,
        overlap=args.overlap,
        include_empty_tiles=True,
        empty_tile_fraction=args.empty_tile_fraction,
        focus_types=focus_types,
        boxes_key=args.boxes_key,
        limit=args.limit_train or None,
    )
    val_ds = TiledPanelInstanceDataset(
        args.data_root,
        "val",
        split_mode=args.split_mode,
        tile_size=args.tile_size,
        overlap=args.overlap,
        include_empty_tiles=True,
        empty_tile_fraction=args.empty_tile_fraction,
        focus_types=focus_types,
        boxes_key=args.boxes_key,
        limit=args.limit_val or None,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_detection)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_detection)
    model = build_tiled_maskrcnn(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)
    write_json(
        out_dir / "run_config.json",
        {
            "args": vars(args),
            "device": str(device),
            "num_classes": NUM_CLASSES,
            "train_tiles": len(train_ds),
            "val_tiles": len(val_ds),
        },
    )
    best = float("inf")
    with open(out_dir / "metrics.jsonl", "w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_train_epoch(model, train_loader, optimizer, device, scaler, args.amp)
            val_metrics = run_val_loss(model, val_loader, device, args.amp)
            row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(
                f"ep{epoch:03d} train loss={train_metrics.get('loss', 0.0):.4f} "
                f"val loss={val_metrics.get('loss', 0.0):.4f}",
                flush=True,
            )
            ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args), "val": val_metrics}
            torch.save(ckpt, out_dir / "last.pt")
            if val_metrics.get("loss", float("inf")) < best:
                best = val_metrics.get("loss", float("inf"))
                torch.save(ckpt, out_dir / "best.pt")
                write_json(out_dir / "best_point.json", {"epoch": epoch, "score": best, "metric": "val_loss"})


if __name__ == "__main__":
    main()
