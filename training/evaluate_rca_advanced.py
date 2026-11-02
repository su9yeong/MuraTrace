"""Evaluator paired with train_rca_advanced.py (rebuilds DefectRCAModelV2)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from mura_rca_data import DefectRCADataset, collate_dict
from mura_rca_models_v2 import DefectRCAModelV2, RCAModelConfig
from train_rca import default_loss_weights, run_epoch, write_json, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate an advanced RCA best.pt/last.pt on a split.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
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
    image_size = int(ckpt_args.get("image_size", 224))
    n_time_bins = int(ckpt_args.get("n_time_bins", 10))
    split_mode = str(ckpt_args.get("split_mode", "manifest"))
    batch_size = int(args.batch_size or ckpt_args.get("batch_size", 64))
    amp = bool(args.amp or ckpt_args.get("amp", False))

    eval_ds = DefectRCADataset(
        data_root, args.split, image_size=image_size, split_mode=split_mode,
        route_shuffle=bool(ckpt_args.get("route_shuffle", False)), n_time_bins=n_time_bins,
        limit=args.limit or None, route_swap=bool(ckpt_args.get("route_swap", False)),
    )
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, collate_fn=collate_dict)

    model = DefectRCAModelV2(
        RCAModelConfig(mode=str(ckpt_args.get("mode", "visionroute")),
                       embed_dim=int(ckpt_args.get("embed_dim", 256)), n_time_bins=n_time_bins),
        image_encoder=str(ckpt_args.get("image_encoder", "smallcnn")),
        image_pretrained=False,  # weights come from checkpoint
        image_init_backbone=None,
        image_size=image_size,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    weights = dict(ckpt.get("loss_weights") or default_loss_weights(
        str(ckpt_args.get("task", "joint")), train_reject_head=bool(ckpt_args.get("train_reject_head", False))))

    metrics, report, predictions = run_epoch(model, eval_loader, None, device, None, amp,
                                             weights, n_time_bins, collect_predictions=True)
    score = (float(metrics.get("tool_id_hit1", 0.0)) + float(metrics.get("process_step_hit1", 0.0))
             + float(metrics.get("fault_mode_hit1", 0.0)) + float(metrics.get("time_bin_hit1", 0.0))) / 4.0
    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_score": ckpt.get("score"),
        "split": args.split,
        "n_items": len(eval_ds),
        "score": score,
        "metrics": metrics,
        "args": ckpt_args,
    }
    write_json(out_dir / f"{args.split}_summary.json", summary)
    write_json(out_dir / f"{args.split}_report.json", report)
    write_jsonl(out_dir / f"{args.split}_predictions.jsonl", predictions)
    print(json.dumps({k: summary[k] for k in ("checkpoint_epoch", "split", "n_items", "score")}, indent=2))


if __name__ == "__main__":
    main()
