"""Mask2Former (mask-classification) trainer for MuraTrace semantic segmentation.

Option 2 in the analysis: instead of per-pixel classification (DeepLabV3+),
Mask2Former predicts a set of mask queries and classifies each. This paradigm
is stronger on blurry, large-area "stuff"-like classes (region, gradient) that
the per-pixel models struggle with.

Implemented on HuggingFace transformers' Mask2FormerForUniversalSegmentation.
Panel masks (ids 0..5, ignore 255) are converted to per-class binary masks for
training. Reporting reuses the shared confusion-matrix metrics so numbers are
directly comparable to the other segmentation runs (miou_defect, per-class IoU).

Checkpoint schema matches the other trainers (best.pt / last.pt with
ckpt["args"], ckpt["model"], ckpt["epoch"], ckpt["score"]) and stores
`hf_model_id` so the paired evaluator rebuilds the same config.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from mura_rca_data import PanelSegDataset, SEG_IGNORE_INDEX
from train_segmentation import SEG_NAMES, metrics_from_confusion, update_confusion, write_json

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
NUM_CLASSES = 6


def build_mask2former(hf_model_id: str | None, pretrained: bool):
    try:
        from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Install transformers to use Mask2Former: pip install transformers"
        ) from exc
    model_id = hf_model_id or "facebook/mask2former-swin-tiny-ade-semantic"
    if pretrained:
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_id, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
        )
    else:
        config = Mask2FormerConfig.from_pretrained(model_id, num_labels=NUM_CLASSES)
        model = Mask2FormerForUniversalSegmentation(config)
    return model


def to_mask_labels(masks: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Convert [B,H,W] id masks into per-image (mask_labels, class_labels).

    One binary mask per class present in the image; ignore-index pixels are
    dropped. Images with no labeled class get an empty target set.
    """
    mask_labels: list[torch.Tensor] = []
    class_labels: list[torch.Tensor] = []
    for m in masks:
        present = [c for c in range(NUM_CLASSES) if bool((m == c).any())]
        if present:
            stack = torch.stack([(m == c).float() for c in present], dim=0)
            cls = torch.tensor(present, dtype=torch.long)
        else:
            stack = torch.zeros((0, m.shape[0], m.shape[1]), dtype=torch.float32)
            cls = torch.zeros((0,), dtype=torch.long)
        mask_labels.append(stack)
        class_labels.append(cls)
    return mask_labels, class_labels


def prep_pixels(images: torch.Tensor) -> torch.Tensor:
    """[B,1,H,W] gray in [0,1] -> [B,3,H,W] ImageNet-normalized."""
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    return (images - IMAGENET_MEAN.to(images.device)) / IMAGENET_STD.to(images.device)


@torch.no_grad()
def semantic_scores(outputs, size: tuple[int, int]) -> torch.Tensor:
    """Combine query class/mask logits into per-pixel class scores [B,C,H,W]."""
    class_logits = outputs.class_queries_logits  # [B,Q,C+1]
    mask_logits = outputs.masks_queries_logits  # [B,Q,h,w]
    mask_logits = F.interpolate(mask_logits, size=size, mode="bilinear", align_corners=False)
    class_prob = class_logits.softmax(dim=-1)[..., :NUM_CLASSES]  # drop "no-object"
    mask_prob = mask_logits.sigmoid()
    return torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)


def run_epoch(model, loader, optimizer, device, scaler, amp):
    train = optimizer is not None
    model.train(train)
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        pixel_values = prep_pixels(images)
        mask_labels, class_labels = to_mask_labels(masks)
        mask_labels = [m.to(device) for m in mask_labels]
        class_labels = [c.to(device) for c in class_labels]
        with torch.set_grad_enabled(train):
            with autocast(enabled=amp, dtype=torch.bfloat16):
                outputs = model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)
                loss = outputs.loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        scores = semantic_scores(outputs, size=masks.shape[-2:])
        # treat ignore pixels as background for confusion bookkeeping
        masks_for_cm = torch.where(masks == int(SEG_IGNORE_INDEX), torch.zeros_like(masks), masks)
        confusion = update_confusion(confusion, scores, masks_for_cm)
        total_loss += float(loss.item())
        n_batches += 1
    flat, report = metrics_from_confusion(confusion)
    flat["loss"] = total_loss / max(n_batches, 1)
    report["losses"] = {"loss": flat["loss"]}
    return flat, report


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Mask2Former for MuraTrace panel semantic segmentation.")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="runs/seg_mask2former")
    ap.add_argument("--hf_model_id", default="facebook/mask2former-swin-tiny-ade-semantic")
    ap.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    ap.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--split_mode", default="manifest", choices=["manifest", "random_panel", "random_glass"])
    ap.add_argument("--mask_key", default="train_mask")
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    reports_dir = out_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_json(out_dir / "run_config.json", {"args": vars(args), "device": str(device), "label_names": SEG_NAMES})

    train_ds = PanelSegDataset(
        args.data_root, "train", image_size=args.image_size, split_mode=args.split_mode,
        defective_only=False, mask_key=args.mask_key, limit=args.limit_train or None,
    )
    val_ds = PanelSegDataset(
        args.data_root, "val", image_size=args.image_size, split_mode=args.split_mode,
        defective_only=False, mask_key=args.mask_key, limit=args.limit_val or None,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_mask2former(args.hf_model_id, args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # bfloat16 autocast needs no gradient scaling
    scaler = GradScaler(enabled=False)

    best = -1.0
    best_point: dict[str, Any] = {}
    with open(out_dir / "metrics.jsonl", "w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            train_metrics, train_report = run_epoch(model, train_loader, optimizer, device, scaler, args.amp)
            val_metrics, val_report = run_epoch(model, val_loader, None, device, None, args.amp)
            row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row, indent=2), flush=True)
            score = float(val_metrics.get("miou_defect", 0.0))
            write_json(reports_dir / f"train_epoch_{epoch:04d}.json", train_report)
            write_json(reports_dir / f"val_epoch_{epoch:04d}.json", val_report)
            ckpt = {"epoch": epoch, "model": model.state_dict(), "args": vars(args), "score": score, "val": val_metrics}
            torch.save(ckpt, out_dir / "last.pt")
            if score > best:
                best = score
                best_point = {"epoch": epoch, "score": score, "checkpoint": "best.pt", "val_metrics": val_metrics}
                torch.save(ckpt, out_dir / "best.pt")
                write_json(out_dir / "best_point.json", best_point)
                write_json(reports_dir / "val_best.json", val_report)
    if best_point:
        write_json(out_dir / "best_point.json", best_point)


if __name__ == "__main__":
    main()
