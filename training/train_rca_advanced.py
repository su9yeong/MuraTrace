"""RCA trainer with a configurable (pretrained) image encoder + image_route mode.

Reuses the epoch loop / metrics / reporting from train_rca.py; only the model
(DefectRCAModelV2) and a few CLI args differ. The checkpoint stores
image_encoder / image_pretrained / image_init_backbone so the paired evaluator
(evaluate_rca_advanced.py) rebuilds the same model.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from mura_rca_data import (
    EXTENTS, FAULT_MODES, MASK_TYPES, POLARITIES, PROCESS_STEPS, SUBTYPES, TOOL_IDS,
    DefectRCADataset, collate_dict, summarize_defect_dataset,
)
from mura_rca_models_v2 import DefectRCAModelV2, RCAModelConfig
from train_rca import (
    append_metrics_csv, default_loss_weights, format_epoch_line, head_names,
    run_epoch, write_json, write_jsonl,
)

MODES = ["image_only", "image_position", "route_only", "image_route", "late_fusion", "rcfilm", "visionroute"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="runs/rca_adv")
    ap.add_argument("--mode", default="visionroute", choices=MODES)
    ap.add_argument("--task", default="joint", choices=["visual", "rca", "joint"])
    ap.add_argument("--image_encoder", default="convnext",
                    help="smallcnn | resnet34 | convnext | swin | ... (see rca_image_encoders.ENCODER_ALIAS)")
    ap.add_argument("--image_pretrained", dest="image_pretrained", action="store_true", default=True)
    ap.add_argument("--no_image_pretrained", dest="image_pretrained", action="store_false")
    ap.add_argument("--image_init_backbone", default=None,
                    help="our SimMIM/MAE backbone ckpt (domain pretraining on real_background)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--n_time_bins", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--route_shuffle", action="store_true")
    ap.add_argument("--route_swap", action="store_true",
                    help="control: feed each defect the route graph of a different glass (fixed derangement)")
    ap.add_argument("--split_mode", default="manifest", choices=["manifest", "random_panel", "random_glass"])
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)
    ap.add_argument("--save_epoch_predictions", action="store_true")
    ap.add_argument("--train_reject_head", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    reports_dir = out_dir / "reports"
    predictions_dir = out_dir / "predictions"
    for d in (out_dir, reports_dir, predictions_dir):
        d.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = DefectRCADataset(args.data_root, "train", image_size=args.image_size, split_mode=args.split_mode,
                                route_shuffle=args.route_shuffle, n_time_bins=args.n_time_bins, limit=args.limit_train or None,
                                route_swap=args.route_swap)
    val_ds = DefectRCADataset(args.data_root, "val", image_size=args.image_size, split_mode=args.split_mode,
                              route_shuffle=args.route_shuffle, n_time_bins=args.n_time_bins, limit=args.limit_val or None,
                              route_swap=args.route_swap)
    write_json(out_dir / "dataset_summary.json", summarize_defect_dataset(args.data_root))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, collate_fn=collate_dict)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, collate_fn=collate_dict)

    model = DefectRCAModelV2(
        RCAModelConfig(mode=args.mode, embed_dim=args.embed_dim, n_time_bins=args.n_time_bins),
        image_encoder=args.image_encoder, image_pretrained=args.image_pretrained,
        image_init_backbone=args.image_init_backbone, image_size=args.image_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)
    weights = default_loss_weights(args.task, train_reject_head=args.train_reject_head)
    write_json(out_dir / "run_config.json", {"args": vars(args), "loss_weights": weights, "device": str(device)})

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} mode={args.mode} image_encoder={args.image_encoder} "
          f"pretrained={args.image_pretrained} params={n_params/1e6:.1f}M "
          f"train={len(train_ds)} val={len(val_ds)}")

    best = -1.0
    best_point: dict[str, Any] = {}
    metrics_csv = out_dir / "metrics.csv"
    with open(out_dir / "metrics.jsonl", "w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_metrics, train_report, _ = run_epoch(model, train_loader, optimizer, device, scaler, args.amp,
                                                       weights, args.n_time_bins, collect_predictions=False)
            val_metrics, val_report, val_predictions = run_epoch(model, val_loader, None, device, None, args.amp,
                                                                 weights, args.n_time_bins, collect_predictions=True)
            score = (val_metrics.get("tool_id_hit1", 0.0) + val_metrics.get("process_step_hit1", 0.0)
                     + val_metrics.get("fault_mode_hit1", 0.0) + val_metrics.get("time_bin_hit1", 0.0)) / 4.0
            log.write(json.dumps({"epoch": epoch, "mode": args.mode, "score": score,
                                  "train": train_metrics, "val": val_metrics}) + "\n")
            log.flush()
            append_metrics_csv(metrics_csv, epoch, train_metrics, val_metrics)
            write_json(reports_dir / f"val_epoch_{epoch:04d}.json", val_report)
            write_jsonl(predictions_dir / "val_last.jsonl", val_predictions)
            if args.save_epoch_predictions:
                write_jsonl(predictions_dir / f"val_epoch_{epoch:04d}.jsonl", val_predictions)

            ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "args": vars(args), "loss_weights": weights, "score": score, "val": val_metrics}
            torch.save(ckpt, out_dir / "last.pt")
            best_updated = False
            if score > best:
                best, best_updated = score, True
                best_point = {"epoch": epoch, "score": score, "checkpoint": "best.pt", "val_metrics": val_metrics}
                torch.save(ckpt, out_dir / "best.pt")
                write_json(out_dir / "best_point.json", best_point)
                write_json(reports_dir / "val_best.json", val_report)
                write_jsonl(predictions_dir / "val_best.jsonl", val_predictions)
            print(format_epoch_line(epoch, time.time() - t0, train_metrics, val_metrics, best_updated), flush=True)

    if best_point:
        write_json(out_dir / "best_point.json", best_point)


if __name__ == "__main__":
    main()
