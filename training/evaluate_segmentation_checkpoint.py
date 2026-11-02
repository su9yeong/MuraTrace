from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from mura_rca_data import PanelSegDataset
from mura_rca_models import build_segmentation_model
from train_segmentation import (
    compute_auto_class_weights,
    run_epoch,
    small_defect_weights,
    write_json,
)


def checkpoint_args(ckpt: dict[str, Any]) -> dict[str, Any]:
    args = dict(ckpt.get("args", {}))
    if not args:
        raise ValueError("checkpoint does not contain training args")
    return args


def resolve_data_root(args: argparse.Namespace, ckpt_args: dict[str, Any]) -> str:
    data_root = args.data_root or ckpt_args.get("data_root")
    if not data_root:
        raise ValueError("pass --data_root or use a checkpoint with args.data_root")
    return str(data_root)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a segmentation best.pt/last.pt checkpoint on val/test split.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--mask_key", default=None, help="override manifest mask key; default follows checkpoint args")
    args = ap.parse_args()

    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = checkpoint_args(ckpt)
    data_root = resolve_data_root(args, ckpt_args)
    run_dir = checkpoint_path.parent
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "test_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    image_size = int(ckpt_args.get("image_size", 512))
    split_mode = str(ckpt_args.get("split_mode", "manifest"))
    mask_key = str(args.mask_key or ckpt_args.get("mask_key", "train_mask"))
    batch_size = int(args.batch_size or ckpt_args.get("batch_size", 4))
    amp = bool(args.amp or ckpt_args.get("amp", False))

    eval_ds = PanelSegDataset(
        data_root,
        args.split,
        image_size=image_size,
        split_mode=split_mode,
        defective_only=False,
        mask_key=mask_key,
        limit=args.limit or None,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    class_weights = None
    class_weight_mode = str(ckpt_args.get("class_weights", "none"))
    if class_weight_mode == "small_defect":
        class_weights = small_defect_weights(num_classes=6)
    elif class_weight_mode == "auto":
        train_ds = PanelSegDataset(
            data_root,
            "train",
            image_size=image_size,
            split_mode=split_mode,
            defective_only=bool(ckpt_args.get("defective_only", False)),
            mask_key=mask_key,
            limit=int(ckpt_args.get("limit_train", 0)) or None,
        )
        class_weights = compute_auto_class_weights(
            train_ds,
            num_classes=6,
            max_items=int(ckpt_args.get("class_weight_max_items", 0) or 0),
            cap=float(ckpt_args.get("class_weight_cap", 12.0)),
        )

    model = build_segmentation_model(
        str(ckpt_args.get("model", "deeplabv3plus")),
        num_classes=6,
        in_ch=1,
        image_size=image_size,
        encoder_name=ckpt_args.get("encoder_name"),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    metrics, report = run_epoch(
        model,
        eval_loader,
        None,
        device,
        None,
        amp,
        class_weights=class_weights,
        focal_gamma=float(ckpt_args.get("focal_gamma", 0.0)),
        dice_weight=float(ckpt_args.get("dice_weight", 1.0)),
    )
    score = float(metrics.get("miou_defect", 0.0))
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_score": ckpt.get("score"),
        "split": args.split,
        "n_items": len(eval_ds),
        "score": score,
        "metrics": metrics,
        "mask_key": mask_key,
        "args": ckpt_args,
    }
    write_json(out_dir / f"{args.split}_summary.json", summary)
    write_json(out_dir / f"{args.split}_report.json", report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
