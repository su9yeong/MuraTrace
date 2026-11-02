"""Advanced per-pixel segmentation trainer for MuraTrace.

Single flexible trainer that realizes the recommended per-pixel options:
  - 0: ImageNet-pretrained encoders  (--pretrained, default on)
  - 1: global-context backbones      (--arch segformer/upernet, --encoder_name mit_b2 / tu-convnext_small / tu-swin_*)
  - 3: high-resolution HRNet          (--encoder_name tu-hrnet_w32)
  - 4: consume an MAE/SimMIM backbone (--init_backbone <ckpt>)

Each batch invocation = one methodology run = one out_dir. The checkpoint
format matches train_segmentation.py and additionally stores `arch`,
`encoder_name`, `pretrained`, and `in_channels` so the paired evaluator
(evaluate_segmentation_advanced.py) rebuilds the exact architecture.

Loss, metrics, class weighting, and the epoch loop are reused from
train_segmentation.py so results stay comparable to the legacy runs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from mura_rca_data import PanelSegDataset
from seg_advanced_models import build_advanced_segmentation_model, encoder_module, load_backbone_weights
from train_segmentation import (
    SEG_NAMES,
    compute_auto_class_weights,
    run_epoch,
    small_defect_weights,
    write_json,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Advanced (pretrained / global-context) segmentation trainer.")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="runs/seg_advanced")
    ap.add_argument(
        "--arch",
        default="deeplabv3plus",
        choices=["deeplabv3plus", "deeplabv3", "unet", "unetplusplus", "upernet", "manet", "pan", "fpn", "pspnet", "linknet", "segformer"],
    )
    ap.add_argument("--encoder_name", default="resnet34", help="smp/timm encoder, e.g. resnet34, mit_b2, tu-convnext_small, tu-swin_tiny_patch4_window7_224, tu-hrnet_w32")
    ap.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    ap.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--in_channels", type=int, default=3, choices=[1, 3], help="3 = gray->RGB adapter to reuse ImageNet weights; 1 = native single channel")
    ap.add_argument("--init_backbone", default=None, help="path to an MAE/SimMIM backbone checkpoint to seed the encoder")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4, help="base (decoder/head) lr; encoder lr is scaled by --encoder_lr_scale")
    ap.add_argument("--encoder_lr_scale", type=float, default=0.1, help="multiplier on lr for the pretrained encoder (lower = more stable fine-tuning)")
    ap.add_argument("--warmup_steps", type=int, default=500, help="linear LR warmup steps; critical for stable pretrained fine-tuning")
    ap.add_argument("--scheduler", default="cosine", choices=["cosine", "none"])
    ap.add_argument("--min_lr_ratio", type=float, default=0.01, help="cosine floor as a fraction of base lr")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp_dtype", default="bfloat16", choices=["bfloat16", "float16"], help="bfloat16 avoids fp16 overflow/NaN on ConvNeXt/transformer encoders")
    ap.add_argument("--skip_nonfinite", dest="skip_nonfinite", action="store_true", default=True, help="skip a batch if loss is non-finite instead of crashing")
    ap.add_argument("--no_skip_nonfinite", dest="skip_nonfinite", action="store_false")
    ap.add_argument("--defective_only", action="store_true")
    ap.add_argument("--split_mode", default="manifest", choices=["manifest", "random_panel", "random_glass"])
    ap.add_argument("--mask_key", default="train_mask")
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)
    ap.add_argument("--class_weights", default="auto", choices=["none", "auto", "small_defect"])
    ap.add_argument("--class_weight_cap", type=float, default=4.0)
    ap.add_argument("--class_weight_max_items", type=int, default=0)
    ap.add_argument("--focal_gamma", type=float, default=0.0)
    ap.add_argument("--dice_weight", type=float, default=1.0)
    # --- camera-ready background ablation options (all off by default = original behaviour) ---
    ap.add_argument("--train_sources", default="", help="comma-separated real-background source ids to train on (e.g. bg00,bg02); empty = all")
    ap.add_argument("--train_subsample", type=int, default=0, help="random panel-count-matched subset of the training rows (0 = off)")
    ap.add_argument("--subsample_seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=-1, help="seed for torch/numpy/random (-1 = unseeded, as in the original runs)")
    ap.add_argument("--max_steps", type=int, default=0, help="fixed optimizer-step budget; overrides --epochs for the schedule and stops training when reached")
    ap.add_argument("--val_every_steps", type=int, default=0, help="validate roughly every N steps (rounded to whole epochs); 0 = every epoch")
    args = ap.parse_args()

    if args.seed >= 0:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    reports_dir = out_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_json(out_dir / "run_config.json", {"args": vars(args), "device": str(device), "label_names": SEG_NAMES})

    train_sources = [s.strip() for s in args.train_sources.split(",") if s.strip()] or None
    train_ds = PanelSegDataset(
        args.data_root, "train", image_size=args.image_size, split_mode=args.split_mode,
        defective_only=args.defective_only, mask_key=args.mask_key, limit=args.limit_train or None,
        source_ids=train_sources, subsample=args.train_subsample or None, subsample_seed=args.subsample_seed,
    )
    print(f"train rows={len(train_ds)} (sources={train_sources or 'all'}, subsample={args.train_subsample or 'off'})", flush=True)
    val_ds = PanelSegDataset(
        args.data_root, "val", image_size=args.image_size, split_mode=args.split_mode,
        defective_only=False, mask_key=args.mask_key, limit=args.limit_val or None,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    class_weights = None
    if args.class_weights == "small_defect":
        class_weights = small_defect_weights(num_classes=6)
    elif args.class_weights == "auto":
        class_weights = compute_auto_class_weights(train_ds, num_classes=6, max_items=args.class_weight_max_items, cap=args.class_weight_cap)
    if class_weights is not None:
        print(f"class_weights={class_weights.tolist()}", flush=True)

    model = build_advanced_segmentation_model(
        arch=args.arch,
        encoder_name=args.encoder_name,
        num_classes=6,
        pretrained=args.pretrained,
        in_channels=args.in_channels,
        image_size=args.image_size,
    ).to(device)

    if args.init_backbone:
        report = load_backbone_weights(model, args.init_backbone)
        print(f"init_backbone={args.init_backbone} matched={report['matched_tensors']}/{report['encoder_tensors']}", flush=True)
        write_json(out_dir / "init_backbone_report.json", report)

    # discriminative LR: pretrained encoder fine-tunes slower than the fresh decoder/head
    enc_param_ids = {id(p) for p in encoder_module(model).parameters()}
    encoder_params = [p for p in model.parameters() if id(p) in enc_param_ids and p.requires_grad]
    other_params = [p for p in model.parameters() if id(p) not in enc_param_ids and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.lr * args.encoder_lr_scale},
            {"params": other_params, "lr": args.lr},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, args.max_steps) if args.max_steps > 0 else max(1, args.epochs * steps_per_epoch)
    n_epochs = int(math.ceil(total_steps / steps_per_epoch)) if args.max_steps > 0 else args.epochs
    val_every_epochs = max(1, int(round(args.val_every_steps / steps_per_epoch))) if args.val_every_steps > 0 else 1
    warmup = min(max(0, args.warmup_steps), max(1, total_steps - 1))

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        if args.scheduler == "cosine":
            prog = float(step - warmup) / float(max(1, total_steps - warmup))
            cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
            return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cos
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    # GradScaler is only needed for float16; bfloat16 has fp32 dynamic range and needs no scaling
    use_scaler = bool(args.amp) and amp_dtype == torch.float16
    scaler = GradScaler(enabled=use_scaler)
    print(
        f"lr base={args.lr} encoder_lr={args.lr * args.encoder_lr_scale} warmup={warmup} "
        f"total_steps={total_steps} sched={args.scheduler} amp_dtype={args.amp_dtype} "
        f"use_scaler={use_scaler} skip_nonfinite={args.skip_nonfinite}",
        flush=True,
    )
    class _StepBudgetLoader:
        """Yields at most `n` batches of the wrapped loader (fixed optimizer-step budgets)."""

        def __init__(self, loader: DataLoader, n: int) -> None:
            self.loader, self.n = loader, int(n)

        def __len__(self) -> int:
            return min(len(self.loader), self.n)

        def __iter__(self):
            for i, batch in enumerate(self.loader):
                if i >= self.n:
                    break
                yield batch

    best = -1.0
    best_point: dict[str, Any] = {}
    steps_done = 0
    with open(out_dir / "metrics.jsonl", "w", encoding="utf-8") as log:
        for epoch in range(1, n_epochs + 1):
            epoch_loader: Any = train_loader
            if args.max_steps > 0:
                remaining = total_steps - steps_done
                if remaining <= 0:
                    break
                epoch_loader = _StepBudgetLoader(train_loader, remaining)
            train_metrics, train_report = run_epoch(
                model, epoch_loader, optimizer, device, scaler, args.amp,
                grad_clip=args.grad_clip, class_weights=class_weights,
                focal_gamma=args.focal_gamma, dice_weight=args.dice_weight,
                scheduler=scheduler, amp_dtype=amp_dtype, skip_nonfinite=args.skip_nonfinite,
            )
            steps_done += len(epoch_loader)
            last_epoch = (epoch == n_epochs) or (args.max_steps > 0 and steps_done >= total_steps)
            if epoch % val_every_epochs != 0 and not last_epoch:
                log.write(json.dumps({"epoch": epoch, "steps_done": steps_done, "train": train_metrics}) + "\n")
                log.flush()
                print(f"epoch {epoch} steps {steps_done}/{total_steps} train miou_def={train_metrics.get('miou_defect', 0.0):.4f} (no val)", flush=True)
                continue
            val_metrics, val_report = run_epoch(
                model, val_loader, None, device, None, args.amp,
                class_weights=class_weights, focal_gamma=args.focal_gamma, dice_weight=args.dice_weight,
                amp_dtype=amp_dtype, skip_nonfinite=args.skip_nonfinite,
            )
            row = {"epoch": epoch, "steps_done": steps_done, "train": train_metrics, "val": val_metrics}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row, indent=2), flush=True)
            score = float(val_metrics.get("miou_defect", 0.0))
            write_json(reports_dir / f"train_epoch_{epoch:04d}.json", train_report)
            write_json(reports_dir / f"val_epoch_{epoch:04d}.json", val_report)
            ckpt = {
                "epoch": epoch,
                "steps_done": steps_done,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "score": score,
                "val": val_metrics,
            }
            torch.save(ckpt, out_dir / "last.pt")
            if score > best:
                best = score
                best_point = {"epoch": epoch, "steps_done": steps_done, "score": score, "checkpoint": "best.pt", "val_metrics": val_metrics}
                torch.save(ckpt, out_dir / "best.pt")
                write_json(out_dir / "best_point.json", best_point)
                write_json(reports_dir / "val_best.json", val_report)
            if last_epoch:
                break
    if best_point:
        write_json(out_dir / "best_point.json", best_point)


if __name__ == "__main__":
    main()
