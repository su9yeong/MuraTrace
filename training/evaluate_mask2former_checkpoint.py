"""Evaluator paired with train_mask2former.py.

Rebuilds the Mask2Former config from the checkpoint args, loads weights, runs
the held-out split, and writes `<split>_summary.json` / `<split>_report.json`
in the shared schema so summarize_wacv_test_eval.py picks it up.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mura_rca_data import PanelSegDataset
from train_mask2former import build_mask2former, run_epoch
from train_segmentation import write_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a Mask2Former best.pt/last.pt on a split.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--mask_key", default=None)
    args = ap.parse_args()

    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = dict(ckpt.get("args", {}))
    if not ckpt_args:
        raise ValueError("checkpoint does not contain training args")
    data_root = args.data_root or ckpt_args.get("data_root")
    if not data_root:
        raise ValueError("pass --data_root or use a checkpoint with args.data_root")
    out_dir = Path(args.out_dir) if args.out_dir else checkpoint_path.parent / "test_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    image_size = int(ckpt_args.get("image_size", 512))
    split_mode = str(ckpt_args.get("split_mode", "manifest"))
    mask_key = str(args.mask_key or ckpt_args.get("mask_key", "train_mask"))
    batch_size = int(args.batch_size or ckpt_args.get("batch_size", 4))
    amp = bool(args.amp or ckpt_args.get("amp", False))

    eval_ds = PanelSegDataset(
        data_root, args.split, image_size=image_size, split_mode=split_mode,
        defective_only=False, mask_key=mask_key, limit=args.limit or None,
    )
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # rebuild with pretrained=False (weights come from the checkpoint)
    model = build_mask2former(str(ckpt_args.get("hf_model_id") or "facebook/mask2former-swin-tiny-ade-semantic"), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    metrics, report = run_epoch(model, eval_loader, None, device, None, amp)
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
