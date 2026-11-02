from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from mura_rca_data import (
    EXTENTS,
    FAULT_MODES,
    MASK_TYPES,
    POLARITIES,
    PROCESS_STEPS,
    SUBTYPES,
    TOOL_IDS,
    DefectRCADataset,
    collate_dict,
    summarize_defect_dataset,
)
from mura_rca_models import DefectRCAModel, RCAModelConfig


CE_HEADS = ["mask_type", "extent", "subtype", "polarity", "process_step", "tool_id", "fault_mode", "time_bin"]
RCA_HEADS = ["process_step", "tool_id", "fault_mode", "time_bin"]


def head_names(head: str, n_time_bins: int) -> list[str]:
    if head == "mask_type":
        return MASK_TYPES
    if head == "extent":
        return EXTENTS
    if head == "subtype":
        return SUBTYPES
    if head == "polarity":
        return POLARITIES
    if head == "process_step":
        return PROCESS_STEPS
    if head == "tool_id":
        return TOOL_IDS
    if head == "fault_mode":
        return FAULT_MODES
    if head == "time_bin":
        return [f"bin_{i}" for i in range(n_time_bins)] + ["none"]
    raise KeyError(head)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def compute_losses(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, float]) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    for head in CE_HEADS:
        losses[head] = F.cross_entropy(out[head], batch[head])
    losses["semu"] = F.smooth_l1_loss(out["semu"], batch["semu"])
    losses["is_reject"] = F.binary_cross_entropy_with_logits(out["is_reject"], batch["is_reject"])
    total = torch.zeros((), device=out["semu"].device)
    for name, loss in losses.items():
        total = total + float(weights.get(name, 1.0)) * loss
    losses["loss"] = total
    return losses


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def classification_report(y_true: list[int], y_pred: list[int], names: list[str]) -> dict[str, Any]:
    n = len(names)
    cm = [[0 for _ in range(n)] for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < n and 0 <= int(p) < n:
            cm[int(t)][int(p)] += 1

    per_class: dict[str, dict[str, float]] = {}
    total = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(n))
    macro_f1 = 0.0
    weighted_f1 = 0.0
    for i, name in enumerate(names):
        tp = float(cm[i][i])
        fp = float(sum(cm[r][i] for r in range(n)) - cm[i][i])
        fn = float(sum(cm[i][c] for c in range(n)) - cm[i][i])
        support = float(sum(cm[i]))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        macro_f1 += f1
        weighted_f1 += f1 * support
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return {
        "accuracy": safe_div(float(correct), float(total)),
        "macro_f1": safe_div(macro_f1, float(n)),
        "weighted_f1": safe_div(weighted_f1, float(total)),
        "support": float(total),
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": names,
    }


def binary_report(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = safe_div(float(tp), float(tp + fp))
    recall = safe_div(float(tp), float(tp + fn))
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": safe_div(float(tp + tn), float(tp + tn + fp + fn)),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "support": float(tp + tn + fp + fn),
    }


def update_rank_metrics(accum: dict[str, dict[str, float]], head: str, logits: torch.Tensor, target: torch.Tensor) -> None:
    order = torch.argsort(logits.detach(), dim=1, descending=True)
    target = target.detach()
    match = order == target[:, None]
    ranks = match.float().argmax(dim=1).float() + 1.0
    item = accum.setdefault(head, {"hit1": 0.0, "hit3": 0.0, "mrr": 0.0, "n": 0.0})
    item["hit1"] += float((ranks <= 1).sum().item())
    item["hit3"] += float((ranks <= 3).sum().item())
    item["mrr"] += float((1.0 / ranks).sum().item())
    item["n"] += float(target.numel())


def finalize_rank_metrics(accum: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for head, item in accum.items():
        n = max(item.get("n", 0.0), 1.0)
        out[head] = {
            "hit1": item.get("hit1", 0.0) / n,
            "hit3": item.get("hit3", 0.0) / n,
            "mrr": item.get("mrr", 0.0) / n,
            "support": item.get("n", 0.0),
        }
    return out


def default_loss_weights(task: str, train_reject_head: bool = False) -> dict[str, float]:
    if task == "visual":
        weights = {
            "mask_type": 1.0,
            "extent": 1.0,
            "subtype": 1.2,
            "polarity": 0.5,
            "semu": 0.3,
            "is_reject": 0.8 if train_reject_head else 0.0,
            "process_step": 0.0,
            "tool_id": 0.0,
            "fault_mode": 0.0,
            "time_bin": 0.0,
        }
        return weights
    if task == "rca":
        weights = {
            "mask_type": 0.2,
            "extent": 0.2,
            "subtype": 0.4,
            "polarity": 0.1,
            "semu": 0.1,
            "is_reject": 0.2 if train_reject_head else 0.0,
            "process_step": 1.0,
            "tool_id": 1.4,
            "fault_mode": 0.8,
            "time_bin": 0.8,
        }
        return weights
    weights = {
        "mask_type": 0.7,
        "extent": 0.5,
        "subtype": 0.8,
        "polarity": 0.3,
        "semu": 0.2,
        "is_reject": 0.5 if train_reject_head else 0.0,
        "process_step": 1.0,
        "tool_id": 1.2,
        "fault_mode": 0.8,
        "time_bin": 0.8,
    }
    return weights


def build_prediction_rows(
    batch: dict[str, Any],
    out: dict[str, torch.Tensor],
    n_time_bins: int,
) -> list[dict[str, Any]]:
    batch_size = int(out["semu"].shape[0])
    labels_by_head: dict[str, torch.Tensor] = {head: batch[head].detach().cpu() for head in CE_HEADS}
    pred_by_head: dict[str, torch.Tensor] = {head: torch.argmax(out[head].detach().cpu(), dim=1) for head in CE_HEADS}
    top_by_head: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for head in CE_HEADS:
        probs = torch.softmax(out[head].detach().cpu(), dim=1)
        k = min(3, probs.shape[1])
        top_by_head[head] = torch.topk(probs, k=k, dim=1)

    semu_pred = out["semu"].detach().cpu()
    semu_true = batch["semu"].detach().cpu()
    reject_threshold = batch["reject_visible_threshold"].detach().cpu()
    reject_prob = torch.sigmoid(out["is_reject"].detach().cpu())
    reject_true = batch["is_reject"].detach().cpu()

    rows: list[dict[str, Any]] = []
    for i in range(batch_size):
        labels = {}
        predictions = {}
        top3 = {}
        for head in CE_HEADS:
            names = head_names(head, n_time_bins)
            label_idx = int(labels_by_head[head][i].item())
            pred_idx = int(pred_by_head[head][i].item())
            labels[head] = {"id": label_idx, "name": names[label_idx] if label_idx < len(names) else str(label_idx)}
            predictions[head] = {"id": pred_idx, "name": names[pred_idx] if pred_idx < len(names) else str(pred_idx)}
            values, indices = top_by_head[head]
            top3[head] = [
                {
                    "id": int(indices[i, j].item()),
                    "name": names[int(indices[i, j].item())] if int(indices[i, j].item()) < len(names) else str(int(indices[i, j].item())),
                    "prob": float(values[i, j].item()),
                }
                for j in range(indices.shape[1])
            ]
        rows.append(
            {
                "glass_id": batch["glass_id"][i],
                "panel_id": batch["panel_id"][i],
                "instance_id": int(batch["instance_id"][i]),
                "labels": labels,
                "predictions": predictions,
                "top3": top3,
                "semu_true": float(semu_true[i].item()),
                "semu_pred": float(semu_pred[i].item()),
                "reject_visible_threshold": float(reject_threshold[i].item()),
                "reject_true": float(reject_true[i].item()),
                "reject_prob": float(reject_prob[i].item()),
                "reject_pred_from_semu": int(float(semu_pred[i].item()) >= float(reject_threshold[i].item())),
                "reject_pred_from_head": int(float(reject_prob[i].item()) >= 0.5),
            }
        )
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def flatten_csv_row(epoch: int, split: str, metrics: dict[str, float]) -> dict[str, Any]:
    row: dict[str, Any] = {"epoch": epoch, "split": split}
    preferred = [
        "loss_loss",
        "mask_type_macro_f1",
        "extent_macro_f1",
        "subtype_macro_f1",
        "polarity_macro_f1",
        "semu_mae",
        "reject_f1",
        "reject_semu_f1",
        "reject_head_f1",
        "process_step_hit1",
        "process_step_hit3",
        "process_step_mrr",
        "tool_id_hit1",
        "tool_id_hit3",
        "tool_id_mrr",
        "fault_mode_hit1",
        "fault_mode_hit3",
        "fault_mode_mrr",
        "time_bin_hit1",
        "time_bin_hit3",
        "time_bin_mrr",
    ]
    class_keys = []
    for name in MASK_TYPES:
        class_keys.append(f"mask_type_f1/{name}")
    for name in EXTENTS:
        class_keys.append(f"extent_f1/{name}")
    for name in SUBTYPES:
        class_keys.append(f"subtype_f1/{name}")
        class_keys.append(f"reject_semu_f1_by_subtype/{name}")
    for key in preferred + class_keys:
        if key in metrics:
            row[key] = metrics[key]
    return row


def append_metrics_csv(path: Path, epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
    rows = [
        flatten_csv_row(epoch, "train", train_metrics),
        flatten_csv_row(epoch, "val", val_metrics),
    ]
    columns = sorted({key for row in rows for key in row.keys()})
    columns = ["epoch", "split"] + [key for key in columns if key not in {"epoch", "split"}]
    existing_rows: list[dict[str, Any]] = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            for key in reader.fieldnames or []:
                if key not in columns:
                    columns.append(key)
            for key in columns:
                for row in existing_rows:
                    row.setdefault(key, "")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in existing_rows + rows:
            writer.writerow(row)


def format_epoch_line(
    epoch: int,
    elapsed: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_updated: bool,
) -> str:
    class_bits = []
    for name in MASK_TYPES:
        value = val_metrics.get(f"mask_type_f1/{name}", 0.0)
        class_bits.append(f"{name}={value:.2f}")
    rca_hit1 = (
        val_metrics.get("process_step_hit1", 0.0)
        + val_metrics.get("tool_id_hit1", 0.0)
        + val_metrics.get("fault_mode_hit1", 0.0)
        + val_metrics.get("time_bin_hit1", 0.0)
    ) / 4.0
    rca_hit3 = (
        val_metrics.get("process_step_hit3", 0.0)
        + val_metrics.get("tool_id_hit3", 0.0)
        + val_metrics.get("fault_mode_hit3", 0.0)
        + val_metrics.get("time_bin_hit3", 0.0)
    ) / 4.0
    rca_mrr = (
        val_metrics.get("process_step_mrr", 0.0)
        + val_metrics.get("tool_id_mrr", 0.0)
        + val_metrics.get("fault_mode_mrr", 0.0)
        + val_metrics.get("time_bin_mrr", 0.0)
    ) / 4.0
    marker = " *best" if best_updated else ""
    return (
        f"ep{epoch:03d} ({elapsed:.0f}s) "
        f"train subtypeF1={train_metrics.get('subtype_macro_f1', 0.0):.3f} "
        f"maskF1={train_metrics.get('mask_type_macro_f1', 0.0):.3f} | "
        f"val subtypeF1={val_metrics.get('subtype_macro_f1', 0.0):.3f} "
        f"maskF1={val_metrics.get('mask_type_macro_f1', 0.0):.3f} "
        f"[{' '.join(class_bits)}] "
        f"rca H@1={rca_hit1:.3f} H@3={rca_hit3:.3f} MRR={rca_mrr:.3f} "
        f"reject(SEMU)F1={val_metrics.get('reject_semu_f1', 0.0):.3f}{marker}"
    )


def build_epoch_report(
    y_true: dict[str, list[int]],
    y_pred: dict[str, list[int]],
    semu_true: list[float],
    semu_pred: list[float],
    reject_true: list[int],
    reject_prob: list[float],
    reject_thresholds: list[float],
    rank_accum: dict[str, dict[str, float]],
    n_time_bins: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    report: dict[str, Any] = {"classification": {}, "reject_by_subtype": {}, "rca_ranking": {}}
    flat: dict[str, float] = {}
    for head in CE_HEADS:
        item = classification_report(y_true[head], y_pred[head], head_names(head, n_time_bins))
        report["classification"][head] = item
        flat[f"{head}_acc"] = float(item["accuracy"])
        flat[f"{head}_macro_f1"] = float(item["macro_f1"])
        flat[f"{head}_weighted_f1"] = float(item["weighted_f1"])
        for name, per_class in item["per_class"].items():
            flat[f"{head}_f1/{name}"] = float(per_class["f1"])

    rank_report = finalize_rank_metrics(rank_accum)
    report["rca_ranking"] = rank_report
    for head, item in rank_report.items():
        flat[f"{head}_hit1"] = float(item["hit1"])
        flat[f"{head}_hit3"] = float(item["hit3"])
        flat[f"{head}_mrr"] = float(item["mrr"])

    semu_abs = [abs(a - b) for a, b in zip(semu_true, semu_pred)]
    flat["semu_mae"] = sum(semu_abs) / max(len(semu_abs), 1)
    reject_pred_head = [1 if p >= 0.5 else 0 for p in reject_prob]
    reject_pred_semu = [1 if p >= t else 0 for p, t in zip(semu_pred, reject_thresholds)]
    reject_head_report = binary_report(reject_true, reject_pred_head)
    reject_semu_report = binary_report(reject_true, reject_pred_semu)
    report["reject_head"] = reject_head_report
    report["reject_from_semu_pred"] = reject_semu_report
    flat["reject_head_precision"] = reject_head_report["precision"]
    flat["reject_head_recall"] = reject_head_report["recall"]
    flat["reject_head_f1"] = reject_head_report["f1"]
    flat["reject_head_acc"] = reject_head_report["accuracy"]
    flat["reject_semu_precision"] = reject_semu_report["precision"]
    flat["reject_semu_recall"] = reject_semu_report["recall"]
    flat["reject_semu_f1"] = reject_semu_report["f1"]
    flat["reject_semu_acc"] = reject_semu_report["accuracy"]
    flat["reject_precision"] = reject_semu_report["precision"]
    flat["reject_recall"] = reject_semu_report["recall"]
    flat["reject_f1"] = reject_semu_report["f1"]
    flat["reject_acc"] = reject_semu_report["accuracy"]

    subtype_names = head_names("subtype", n_time_bins)
    subtype_true = y_true["subtype"]
    for subtype_idx, subtype_name in enumerate(subtype_names):
        idxs = [i for i, value in enumerate(subtype_true) if value == subtype_idx]
        if not idxs:
            continue
        by_true = [reject_true[i] for i in idxs]
        by_pred_head = [reject_pred_head[i] for i in idxs]
        by_pred_semu = [reject_pred_semu[i] for i in idxs]
        head_item = binary_report(by_true, by_pred_head)
        semu_item = binary_report(by_true, by_pred_semu)
        report["reject_by_subtype"][subtype_name] = {
            "from_reject_head": head_item,
            "from_semu_pred": semu_item,
        }
        flat[f"reject_head_f1_by_subtype/{subtype_name}"] = float(head_item["f1"])
        flat[f"reject_semu_f1_by_subtype/{subtype_name}"] = float(semu_item["f1"])
        flat[f"reject_f1_by_subtype/{subtype_name}"] = float(semu_item["f1"])
    return report, flat


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: GradScaler | None,
    amp: bool,
    weights: dict[str, float],
    n_time_bins: int,
    collect_predictions: bool = False,
) -> tuple[dict[str, float], dict[str, Any], list[dict[str, Any]]]:
    train = optimizer is not None
    model.train(train)
    loss_totals: dict[str, float] = {}
    n_batches = 0
    n_items = 0
    y_true: dict[str, list[int]] = {head: [] for head in CE_HEADS}
    y_pred: dict[str, list[int]] = {head: [] for head in CE_HEADS}
    semu_true: list[float] = []
    semu_pred: list[float] = []
    reject_true: list[int] = []
    reject_prob: list[float] = []
    reject_thresholds: list[float] = []
    rank_accum: dict[str, dict[str, float]] = {}
    prediction_rows: list[dict[str, Any]] = []

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(train):
            with autocast(enabled=amp):
                out = model(batch)
                losses = compute_losses(out, batch, weights)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(losses["loss"]).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["loss"].backward()
                    optimizer.step()

        out_detached = {k: v.detach() for k, v in out.items()}
        bsz = int(batch["image"].shape[0])
        for k, v in losses.items():
            loss_totals[k] = loss_totals.get(k, 0.0) + float(v.detach().item())

        for head in CE_HEADS:
            pred = torch.argmax(out_detached[head], dim=1)
            y_true[head].extend([int(v) for v in batch[head].detach().cpu().tolist()])
            y_pred[head].extend([int(v) for v in pred.detach().cpu().tolist()])
            if head in RCA_HEADS:
                update_rank_metrics(rank_accum, head, out_detached[head], batch[head])

        semu_true.extend([float(v) for v in batch["semu"].detach().cpu().tolist()])
        semu_pred.extend([float(v) for v in out_detached["semu"].detach().cpu().tolist()])
        reject_true.extend([int(v >= 0.5) for v in batch["is_reject"].detach().cpu().tolist()])
        reject_prob.extend([float(v) for v in torch.sigmoid(out_detached["is_reject"]).detach().cpu().tolist()])
        reject_thresholds.extend([float(v) for v in batch["reject_visible_threshold"].detach().cpu().tolist()])
        if collect_predictions:
            cpu_batch = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}
            cpu_out = {k: v.detach().cpu() for k, v in out_detached.items()}
            prediction_rows.extend(build_prediction_rows(cpu_batch, cpu_out, n_time_bins))

        n_items += bsz
        n_batches += 1

    report, metrics = build_epoch_report(
        y_true,
        y_pred,
        semu_true,
        semu_pred,
        reject_true,
        reject_prob,
        reject_thresholds,
        rank_accum,
        n_time_bins,
    )
    metrics.update({f"loss_{k}": v / max(n_batches, 1) for k, v in loss_totals.items()})
    metrics["n_items"] = float(n_items)
    return metrics, report, prediction_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="runs/rca_visionroute")
    ap.add_argument(
        "--mode",
        default="visionroute",
        choices=["image_only", "image_position", "route_only", "late_fusion", "rcfilm", "visionroute"],
    )
    ap.add_argument("--task", default="joint", choices=["visual", "rca", "joint"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--n_time_bins", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--route_shuffle", action="store_true")
    ap.add_argument("--split_mode", default="manifest", choices=["manifest", "random_panel", "random_glass"])
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)
    ap.add_argument("--save_epoch_predictions", action="store_true")
    ap.add_argument(
        "--train_reject_head",
        action="store_true",
        help="Train the reject BCE head. By default reject is evaluated from predicted SEMU and reject_visible_threshold.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    reports_dir = out_dir / "reports"
    predictions_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = DefectRCADataset(
        args.data_root,
        "train",
        image_size=args.image_size,
        split_mode=args.split_mode,
        route_shuffle=args.route_shuffle,
        n_time_bins=args.n_time_bins,
        limit=args.limit_train or None,
    )
    val_ds = DefectRCADataset(
        args.data_root,
        "val",
        image_size=args.image_size,
        split_mode=args.split_mode,
        route_shuffle=args.route_shuffle,
        n_time_bins=args.n_time_bins,
        limit=args.limit_val or None,
    )

    dataset_summary = summarize_defect_dataset(args.data_root)
    write_json(out_dir / "dataset_summary.json", dataset_summary)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_dict,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_dict,
    )

    model = DefectRCAModel(
        RCAModelConfig(mode=args.mode, embed_dim=args.embed_dim, n_time_bins=args.n_time_bins)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)
    weights = default_loss_weights(args.task, train_reject_head=args.train_reject_head)
    write_json(
        out_dir / "run_config.json",
        {
            "args": vars(args),
            "loss_weights": weights,
            "device": str(device),
            "label_names": {
                "mask_type": MASK_TYPES,
                "extent": EXTENTS,
                "subtype": SUBTYPES,
                "polarity": POLARITIES,
                "process_step": PROCESS_STEPS,
                "tool_id": TOOL_IDS,
                "fault_mode": FAULT_MODES,
                "time_bin": head_names("time_bin", args.n_time_bins),
            },
        },
    )

    best = -1.0
    best_point: dict[str, Any] = {}
    metrics_csv = out_dir / "metrics.csv"
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"device={device} mode={args.mode} task={args.task} "
        f"reject_train={args.train_reject_head} params={n_params / 1e6:.1f}M "
        f"train={len(train_ds)} val={len(val_ds)}"
    )
    with open(out_dir / "metrics.jsonl", "w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_metrics, train_report, _ = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                scaler,
                args.amp,
                weights,
                args.n_time_bins,
                collect_predictions=False,
            )
            val_metrics, val_report, val_predictions = run_epoch(
                model,
                val_loader,
                None,
                device,
                None,
                args.amp,
                weights,
                args.n_time_bins,
                collect_predictions=True,
            )
            score = (
                val_metrics.get("tool_id_hit1", 0.0)
                + val_metrics.get("process_step_hit1", 0.0)
                + val_metrics.get("fault_mode_hit1", 0.0)
                + val_metrics.get("time_bin_hit1", 0.0)
            ) / 4.0
            row = {
                "epoch": epoch,
                "mode": args.mode,
                "task": args.task,
                "score": score,
                "train": train_metrics,
                "val": val_metrics,
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            append_metrics_csv(metrics_csv, epoch, train_metrics, val_metrics)

            write_json(reports_dir / f"train_epoch_{epoch:04d}.json", train_report)
            write_json(reports_dir / f"val_epoch_{epoch:04d}.json", val_report)
            write_jsonl(predictions_dir / "val_last.jsonl", val_predictions)
            if args.save_epoch_predictions:
                write_jsonl(predictions_dir / f"val_epoch_{epoch:04d}.jsonl", val_predictions)

            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "loss_weights": weights,
                "score": score,
                "val": val_metrics,
            }
            torch.save(ckpt, out_dir / "last.pt")
            best_updated = False
            if score > best:
                best = score
                best_updated = True
                best_point = {
                    "epoch": epoch,
                    "score": score,
                    "checkpoint": "best.pt",
                    "val_metrics": val_metrics,
                }
                torch.save(ckpt, out_dir / "best.pt")
                write_json(out_dir / "best_point.json", best_point)
                write_json(reports_dir / "val_best.json", val_report)
                write_jsonl(predictions_dir / "val_best.jsonl", val_predictions)
            elapsed = time.time() - t0
            print(format_epoch_line(epoch, elapsed, train_metrics, val_metrics, best_updated), flush=True)

    if best_point:
        write_json(out_dir / "best_point.json", best_point)


if __name__ == "__main__":
    main()
