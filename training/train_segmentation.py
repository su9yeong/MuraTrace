from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from mura_rca_data import MASK_TYPES, SEG_IGNORE_INDEX, PanelSegDataset
from mura_rca_models import build_segmentation_model


SEG_NAMES = ["background", *MASK_TYPES]


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 6,
    eps: float = 1e-6,
    ignore_index: int = SEG_IGNORE_INDEX,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    valid = (target >= 0) & (target < num_classes) & (target != int(ignore_index))
    if not bool(valid.any()):
        return logits.sum() * 0.0
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    onehot = F.one_hot(safe_target.clamp_min(0), num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid_f = valid[:, None].float()
    probs = probs * valid_f
    onehot = onehot * valid_f
    dims = (0, 2, 3)
    inter = torch.sum(probs * onehot, dims)
    denom = torch.sum(probs + onehot, dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def focal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    gamma: float = 2.0,
    ignore_index: int = SEG_IGNORE_INDEX,
) -> torch.Tensor:
    valid = target != int(ignore_index)
    ce = F.cross_entropy(logits, target, weight=weight, ignore_index=int(ignore_index), reduction="none")
    if not bool(valid.any()):
        return ce.sum() * 0.0
    pt = torch.exp(-ce.detach())
    return torch.mean((((1.0 - pt) ** float(gamma)) * ce)[valid])


def small_defect_weights(num_classes: int = 6) -> torch.Tensor:
    # background, spot, line, region, gradient, band
    values = [0.25, 12.0, 8.0, 2.0, 1.5, 1.0]
    if num_classes != len(values):
        values = [1.0] * num_classes
    return torch.tensor(values, dtype=torch.float32)


def compute_auto_class_weights(
    dataset: PanelSegDataset,
    num_classes: int = 6,
    max_items: int = 0,
    cap: float = 12.0,
) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    n = len(dataset) if max_items <= 0 else min(len(dataset), int(max_items))
    for idx in range(n):
        mask = dataset[idx]["mask"].view(-1)
        valid = (mask >= 0) & (mask < num_classes)
        counts += torch.bincount(mask[valid].clamp(0, num_classes - 1), minlength=num_classes).to(torch.float64)
    freq = counts / max(float(counts.sum().item()), 1.0)
    weights = torch.rsqrt(freq.clamp_min(1e-9))
    present = counts > 0
    if bool(present.any()):
        weights = weights / weights[present].mean().clamp_min(1e-9)
    weights = torch.clamp(weights, min=0.05, max=float(cap))
    if num_classes > 0:
        weights[0] = min(float(weights[0]), 0.35)
    return weights.to(torch.float32)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def update_confusion(confusion: torch.Tensor, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num_classes = int(logits.shape[1])
    pred = torch.argmax(logits.detach(), dim=1).view(-1)
    true = target.detach().view(-1)
    valid = (true >= 0) & (true < num_classes)
    encoded = true[valid] * num_classes + pred[valid]
    counts = torch.bincount(encoded, minlength=num_classes * num_classes)
    return confusion + counts.reshape(num_classes, num_classes).cpu()


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def metrics_from_confusion(confusion: torch.Tensor) -> tuple[dict[str, float], dict[str, Any]]:
    cm = confusion.to(torch.float64)
    n = int(cm.shape[0])
    per_class: dict[str, dict[str, float]] = {}
    total = float(cm.sum().item())
    correct = float(torch.diag(cm).sum().item())
    macro_f1 = 0.0
    macro_iou = 0.0
    defect_ious = []
    for idx in range(n):
        tp = float(cm[idx, idx].item())
        fp = float(cm[:, idx].sum().item() - tp)
        fn = float(cm[idx, :].sum().item() - tp)
        support = float(cm[idx, :].sum().item())
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        iou = safe_div(tp, tp + fp + fn)
        macro_f1 += f1
        macro_iou += iou
        if idx > 0 and support > 0:
            defect_ious.append(iou)
        name = SEG_NAMES[idx] if idx < len(SEG_NAMES) else str(idx)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "support": support,
        }
    flat = {
        "pixel_acc": safe_div(correct, total),
        "macro_f1": safe_div(macro_f1, float(n)),
        "macro_iou": safe_div(macro_iou, float(n)),
        "miou_defect": safe_div(sum(defect_ious), float(len(defect_ious))),
    }
    for idx, name in enumerate(SEG_NAMES):
        item = per_class[name]
        flat[f"iou_{idx}"] = item["iou"]
        flat[f"f1_{idx}"] = item["f1"]
    report = {
        "labels": SEG_NAMES,
        "confusion_matrix": confusion.long().tolist(),
        "per_class": per_class,
        "summary": flat,
    }
    return flat, report


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: GradScaler | None,
    amp: bool,
    grad_clip: float = 0.0,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
    dice_weight: float = 1.0,
    scheduler: Any | None = None,
    amp_dtype: Any | None = None,
    skip_nonfinite: bool = False,
) -> tuple[dict[str, float], dict[str, Any]]:
    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {}
    n_batches = 0
    confusion = torch.zeros(6, 6, dtype=torch.long)
    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(train):
            with autocast(enabled=amp, dtype=amp_dtype or torch.float16):
                logits = model(batch["image"])
                logits_for_loss = logits.float()
                weights = class_weights.to(logits_for_loss.device) if class_weights is not None else None
                if focal_gamma > 0:
                    ce = focal_cross_entropy(
                        logits_for_loss,
                        batch["mask"],
                        weight=weights,
                        gamma=focal_gamma,
                        ignore_index=SEG_IGNORE_INDEX,
                    )
                else:
                    ce = F.cross_entropy(logits_for_loss, batch["mask"], weight=weights, ignore_index=SEG_IGNORE_INDEX)
                dl = dice_loss(
                    logits_for_loss,
                    batch["mask"],
                    num_classes=logits.shape[1],
                    ignore_index=SEG_IGNORE_INDEX,
                )
                loss = ce + float(dice_weight) * dl
            if not torch.isfinite(loss):
                panel_ids = batch.get("panel_id", [])
                if isinstance(panel_ids, (list, tuple)):
                    panel_ids = list(panel_ids)[:8]
                finite_bits = {
                    "image_finite": bool(torch.isfinite(batch["image"]).all().detach().cpu()),
                    "logits_finite": bool(torch.isfinite(logits_for_loss).all().detach().cpu()),
                    "mask_min": int(batch["mask"].min().detach().cpu()),
                    "mask_max": int(batch["mask"].max().detach().cpu()),
                }
                message = (
                    f"non-finite segmentation loss at batch {batch_idx}: "
                    f"loss={float(loss.detach().cpu())}, ce={float(ce.detach().cpu())}, "
                    f"dice={float(dl.detach().cpu())}, diagnostics={finite_bits}, "
                    f"panel_ids={panel_ids}"
                )
                if skip_nonfinite:
                    if train and optimizer is not None:
                        optimizer.zero_grad(set_to_none=True)
                    print(f"WARNING: skipping batch (non-finite loss). {message}", flush=True)
                    continue
                raise FloatingPointError(message)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        confusion = update_confusion(confusion, logits, batch["mask"])
        metrics = {"loss": float(loss.item()), "ce": float(ce.item()), "dice_loss": float(dl.item())}
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        n_batches += 1
    flat, report = metrics_from_confusion(confusion)
    flat.update({k: v / max(n_batches, 1) for k, v in totals.items()})
    report["losses"] = {k: v / max(n_batches, 1) for k, v in totals.items()}
    return flat, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="runs/seg_unet")
    ap.add_argument("--model", default="unet", choices=["unet", "smp_unet", "segformer", "swin_unetr", "vmamba", "hrnet", "deeplabv3plus"])
    ap.add_argument("--encoder_name", default=None, help="optional timm encoder name, mainly for --model vmamba")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--defective_only", action="store_true")
    ap.add_argument("--split_mode", default="manifest", choices=["manifest", "random_panel", "random_glass"])
    ap.add_argument("--mask_key", default="train_mask", help="manifest mask key for training/evaluation; falls back to mask")
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)
    ap.add_argument("--class_weights", default="none", choices=["none", "auto", "small_defect"])
    ap.add_argument("--class_weight_cap", type=float, default=12.0)
    ap.add_argument("--class_weight_max_items", type=int, default=0)
    ap.add_argument("--focal_gamma", type=float, default=0.0)
    ap.add_argument("--dice_weight", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    reports_dir = out_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_json(out_dir / "run_config.json", {"args": vars(args), "device": str(device), "label_names": SEG_NAMES})

    train_ds = PanelSegDataset(
        args.data_root,
        "train",
        image_size=args.image_size,
        split_mode=args.split_mode,
        defective_only=args.defective_only,
        mask_key=args.mask_key,
        limit=args.limit_train or None,
    )
    val_ds = PanelSegDataset(
        args.data_root,
        "val",
        image_size=args.image_size,
        split_mode=args.split_mode,
        defective_only=False,
        mask_key=args.mask_key,
        limit=args.limit_val or None,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    class_weights = None
    if args.class_weights == "small_defect":
        class_weights = small_defect_weights(num_classes=6)
    elif args.class_weights == "auto":
        class_weights = compute_auto_class_weights(
            train_ds,
            num_classes=6,
            max_items=args.class_weight_max_items,
            cap=args.class_weight_cap,
        )
    if class_weights is not None:
        print(f"class_weights={class_weights.tolist()}", flush=True)
    model = build_segmentation_model(
        args.model,
        num_classes=6,
        in_ch=1,
        image_size=args.image_size,
        encoder_name=args.encoder_name,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)
    best = -1.0
    best_point: dict[str, Any] = {}
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
                dice_weight=args.dice_weight,
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
                dice_weight=args.dice_weight,
            )
            row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row, indent=2), flush=True)
            score = float(val_metrics.get("miou_defect", 0.0))
            write_json(reports_dir / f"train_epoch_{epoch:04d}.json", train_report)
            write_json(reports_dir / f"val_epoch_{epoch:04d}.json", val_report)
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "score": score,
                "val": val_metrics,
            }
            torch.save(ckpt, out_dir / "last.pt")
            if score > best:
                best = score
                best_point = {
                    "epoch": epoch,
                    "score": score,
                    "checkpoint": "best.pt",
                    "val_metrics": val_metrics,
                }
                torch.save(ckpt, out_dir / "best.pt")
                write_json(out_dir / "best_point.json", best_point)
                write_json(reports_dir / "val_best.json", val_report)
    if best_point:
        write_json(out_dir / "best_point.json", best_point)


if __name__ == "__main__":
    main()
