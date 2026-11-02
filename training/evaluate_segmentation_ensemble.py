"""Specialist-ensemble evaluator (option 5).

Class-wise routing between two full-panel semantic-seg checkpoints:
  - a "coarse" model that owns the large/low-frequency classes (region, gradient,
    band, background)
  - a "fine" model that owns the small/high-frequency classes (spot, line)

Both checkpoints may come from the legacy trainer (train_segmentation.py) or the
advanced trainer (train_segmentation_advanced.py); the loader auto-detects via
seg_advanced_models.build_model_from_ckpt_args. Per pixel, the coarse argmax is
the base prediction; where the fine model predicts one of --fine_classes, it
overrides. Output is written in the shared schema so summarize picks it up.

This pairs as an eval-only methodology: no training, it composes already-trained
checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mura_rca_data import PanelSegDataset, SEG_IGNORE_INDEX
from seg_advanced_models import build_model_from_ckpt_args
from train_segmentation import metrics_from_confusion, write_json

NUM_CLASSES = 6


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = dict(ckpt.get("args", {}))
    model = build_model_from_ckpt_args(ckpt_args, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt_args, ckpt


def confusion_from_preds(confusion: torch.Tensor, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.view(-1)
    true = target.view(-1)
    valid = (true >= 0) & (true < NUM_CLASSES)
    encoded = true[valid] * NUM_CLASSES + pred[valid]
    counts = torch.bincount(encoded, minlength=NUM_CLASSES * NUM_CLASSES)
    return confusion + counts.reshape(NUM_CLASSES, NUM_CLASSES).cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description="Class-wise specialist ensemble of two segmentation checkpoints.")
    ap.add_argument("--coarse_ckpt", required=True, help="model owning region/gradient/band")
    ap.add_argument("--fine_ckpt", required=True, help="model owning spot/line")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--mask_key", default="train_mask")
    ap.add_argument("--fine_classes", default="1,2", help="comma class ids the fine model overrides (default spot,line)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fine_classes = {int(c) for c in str(args.fine_classes).split(",") if c.strip() != ""}

    coarse, coarse_args, coarse_ck = load_model(args.coarse_ckpt, device)
    fine, fine_args, fine_ck = load_model(args.fine_ckpt, device)

    eval_ds = PanelSegDataset(
        args.data_root, args.split, image_size=args.image_size,
        split_mode=str(coarse_args.get("split_mode", "manifest")),
        defective_only=False, mask_key=args.mask_key,
    )
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"]
            coarse_pred = torch.argmax(coarse(image), dim=1).cpu()
            fine_pred = torch.argmax(fine(image), dim=1).cpu()
            merged = coarse_pred.clone()
            override = torch.zeros_like(fine_pred, dtype=torch.bool)
            for c in fine_classes:
                override |= fine_pred == c
            merged[override] = fine_pred[override]
            masks_for_cm = torch.where(mask == int(SEG_IGNORE_INDEX), torch.zeros_like(mask), mask)
            confusion = confusion_from_preds(confusion, merged, masks_for_cm)

    metrics, report = metrics_from_confusion(confusion)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "coarse_checkpoint": args.coarse_ckpt,
        "fine_checkpoint": args.fine_ckpt,
        "fine_classes": sorted(fine_classes),
        "checkpoint_epoch": coarse_ck.get("epoch"),
        "checkpoint_val_score": coarse_ck.get("score"),
        "split": args.split,
        "n_items": len(eval_ds),
        "score": float(metrics.get("miou_defect", 0.0)),
        "metrics": metrics,
        "mask_key": args.mask_key,
        "args": {"coarse": coarse_args, "fine": fine_args, "image_size": args.image_size},
    }
    write_json(out_dir / f"{args.split}_summary.json", summary)
    write_json(out_dir / f"{args.split}_report.json", report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
