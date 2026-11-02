from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from mura_rca_data import MASK_TYPE_TO_IDX, assigned_split, load_manifest, normalize_image, rel_path


NUM_CLASSES = 6  # background + five mask types


def row_path(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    raise KeyError(f"manifest row is missing all path keys: {keys}")


class PanelInstanceDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        split_mode: str = "manifest",
        image_size: int | None = None,
        include_empty: bool = True,
        boxes_key: str = "train_boxes",
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.boxes_key = str(boxes_key)
        rows = []
        for row in load_manifest(self.root):
            if assigned_split(row, split_mode) != split:
                continue
            if not include_empty and int(row.get("train_boxes_count", row.get("boxes_count", 0))) == 0:
                continue
            rows.append(row)
        if limit:
            rows = rows[: int(limit)]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def _resize(self, image: np.ndarray, inst: np.ndarray, boxes: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if self.image_size is None:
            return image, inst, boxes
        from PIL import Image

        h, w = image.shape[:2]
        size = int(self.image_size)
        sx = size / max(w, 1)
        sy = size / max(h, 1)
        img_pil = Image.fromarray(np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="L")
        inst_pil = Image.fromarray(inst.astype(np.uint16), mode="I;16")
        image = np.asarray(img_pil.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        inst = np.asarray(inst_pil.resize((size, size), Image.Resampling.NEAREST), dtype=np.uint16)
        scaled_boxes = []
        for box in boxes:
            x0, y0, x1, y1 = [float(v) for v in box["bbox"]]
            new_box = dict(box)
            new_box["bbox"] = [x0 * sx, y0 * sy, x1 * sx, y1 * sy]
            scaled_boxes.append(new_box)
        return image, inst, scaled_boxes

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        row = self.rows[idx]
        image = normalize_image(np.load(rel_path(self.root, row_path(row, "image", "image_npy"))))
        inst = np.load(rel_path(self.root, row_path(row, "inst", "instance", "instance_map"))).astype(np.uint16)
        with open(rel_path(self.root, row_path(row, self.boxes_key, "boxes", "box_json", "box")), encoding="utf-8") as f:
            boxes_obj = json.load(f)
        boxes = list(boxes_obj.get("boxes", []))
        image, inst, boxes = self._resize(image, inst, boxes)

        image_tensor = torch.from_numpy(image.astype(np.float32))[None].repeat(3, 1, 1)
        target_boxes = []
        labels = []
        masks = []
        areas = []
        for box in boxes:
            instance_id = int(box.get("instance_id", -1))
            mask = inst == instance_id
            if not np.any(mask):
                continue
            x0, y0, x1, y1 = [float(v) for v in box["bbox"]]
            if x1 <= x0 or y1 <= y0:
                continue
            mask_type = str(box.get("mask_type", ""))
            if mask_type not in MASK_TYPE_TO_IDX:
                continue
            target_boxes.append([x0, y0, x1, y1])
            labels.append(int(MASK_TYPE_TO_IDX[mask_type]) + 1)
            masks.append(mask.astype(np.uint8))
            areas.append((x1 - x0) * (y1 - y0))

        if target_boxes:
            boxes_tensor = torch.tensor(target_boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            masks_tensor = torch.from_numpy(np.stack(masks, axis=0)).to(torch.uint8)
            areas_tensor = torch.tensor(areas, dtype=torch.float32)
        else:
            h, w = image.shape[:2]
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            masks_tensor = torch.zeros((0, h, w), dtype=torch.uint8)
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


def collate_detection(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    return [b[0] for b in batch], [b[1] for b in batch]


def build_model(pretrained: bool = False) -> torch.nn.Module:
    from torchvision.models.detection import maskrcnn_resnet50_fpn
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

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)
    return model


def move_targets(targets: list[dict[str, torch.Tensor]], device: torch.device) -> list[dict[str, torch.Tensor]]:
    return [{k: v.to(device) for k, v in target.items()} for target in targets]


def run_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None,
    amp: bool,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    n = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = move_targets(targets, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            losses = model(images, targets)
            loss = sum(losses.values())
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item())
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().item())
        n += 1
    return {key: value / max(n, 1) for key, value in totals.items()}


@torch.no_grad()
def run_val_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> dict[str, float]:
    # Torchvision detection models return predictions in eval mode, not losses.
    # Keep train mode but disable gradients to get validation losses.
    was_training = model.training
    model.train()
    totals: dict[str, float] = {}
    n = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = move_targets(targets, device)
        with autocast(enabled=amp):
            losses = model(images, targets)
            loss = sum(losses.values())
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().item())
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().item())
        n += 1
    model.train(was_training)
    return {key: value / max(n, 1) for key, value in totals.items()}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="runs/maskrcnn")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--include_empty", action="store_true")
    ap.add_argument("--split_mode", default="manifest", choices=["manifest", "random_panel", "random_glass"])
    ap.add_argument("--boxes_key", default="train_boxes", help="manifest boxes key for training; falls back to boxes")
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = PanelInstanceDataset(
        args.data_root,
        "train",
        split_mode=args.split_mode,
        image_size=args.image_size,
        include_empty=args.include_empty,
        boxes_key=args.boxes_key,
        limit=args.limit_train or None,
    )
    val_ds = PanelInstanceDataset(
        args.data_root,
        "val",
        split_mode=args.split_mode,
        image_size=args.image_size,
        include_empty=args.include_empty,
        boxes_key=args.boxes_key,
        limit=args.limit_val or None,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_detection,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_detection,
    )
    model = build_model(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)
    write_json(
        out_dir / "run_config.json",
        {"args": vars(args), "device": str(device), "num_classes": NUM_CLASSES, "train": len(train_ds), "val": len(val_ds)},
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
                f"val loss={val_metrics.get('loss', 0.0):.4f}"
            )
            ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args), "val": val_metrics}
            torch.save(ckpt, out_dir / "last.pt")
            if val_metrics.get("loss", float("inf")) < best:
                best = val_metrics.get("loss", float("inf"))
                torch.save(ckpt, out_dir / "best.pt")
                write_json(out_dir / "best_point.json", {"epoch": epoch, "score": best, "metric": "val_loss"})


if __name__ == "__main__":
    main()
