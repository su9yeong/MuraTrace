#!/usr/bin/env python3
"""Rebuild train_mask/train_boxes/train_yolo from full support labels and SEMU thresholds."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


IGNORE_LABEL = 255


def read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rel_path(root: Path, value: str) -> Path:
    return root / str(value).replace("\\", "/")


def row_path(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    raise KeyError(f"manifest row is missing all path keys: {keys}")


def visibility_threshold(thresholds: dict[str, Any], mask_type: str, subtype: str) -> dict[str, float]:
    d = dict(thresholds.get("default", {"engineer_visible": 0.5, "reject_visible": 1.0}))
    d.update(thresholds.get("by_mask_type", {}).get(mask_type, {}))
    d.update(thresholds.get("by_subtype", {}).get(subtype, {}))
    return {"engineer_visible": float(d["engineer_visible"]), "reject_visible": float(d["reject_visible"])}


def write_yolo(path: Path, boxes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for box in boxes:
            yolo = box.get("yolo")
            if not yolo or len(yolo) != 4:
                continue
            cls = int(box["class_id"]) - 1
            cx, cy, w, h = [float(v) for v in yolo]
            if w <= 0 or h <= 0:
                continue
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def update_defect_flags(defect: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    out = dict(defect)
    mask_type = str(out.get("mask_type", ""))
    subtype = str(out.get("subtype", ""))
    thr = visibility_threshold(thresholds, mask_type, subtype)
    semu = float(out.get("semu", out.get("panel_label_semu", 0.0)) or 0.0)
    is_visible = semu >= thr["engineer_visible"]
    is_reject = semu >= thr["reject_visible"]
    label_area = int(out.get("target_label_mask_area_px", 0) or 0)
    out.update(
        {
            "engineer_visible_threshold": thr["engineer_visible"],
            "reject_visible_threshold": thr["reject_visible"],
            "panel_is_locally_visible": bool(is_visible),
            "panel_is_locally_reject": bool(is_reject),
            "segmentation_train_enabled": bool(is_visible),
            "is_reject": bool(is_reject),
            "train_mask_ignore_label": IGNORE_LABEL,
            "train_label_mask_area_px": label_area if is_visible else 0,
            "train_ignore_mask_area_px": 0 if is_visible else label_area,
        }
    )
    return out


def box_with_threshold_flags(box: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    out = dict(box)
    mask_type = str(out.get("mask_type", ""))
    subtype = str(out.get("subtype", ""))
    thr = visibility_threshold(thresholds, mask_type, subtype)
    semu = float(out.get("semu", out.get("panel_label_semu", 0.0)) or 0.0)
    is_visible = semu >= thr["engineer_visible"]
    is_reject = semu >= thr["reject_visible"]
    out.update(
        {
            "engineer_visible_threshold": thr["engineer_visible"],
            "reject_visible_threshold": thr["reject_visible"],
            "panel_is_locally_visible": bool(is_visible),
            "panel_is_locally_reject": bool(is_reject),
            "segmentation_train_enabled": bool(is_visible),
            "is_reject": bool(is_reject),
        }
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild train_mask/train_boxes/train_yolo from a new SEMU threshold file.")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--backup_manifest", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_root)
    thresholds = read_json(Path(args.thresholds))
    manifest_path = root / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if args.backup_manifest:
        backup = root / "manifest.before_training_target_update.jsonl"
        if not backup.exists():
            shutil.copy2(manifest_path, backup)

    updated_rows: list[dict[str, Any]] = []
    counts = {"panels": 0, "train_boxes": 0, "ignored_instances": 0, "train_pixels": 0, "ignore_pixels": 0}
    for row in rows:
        row = dict(row)
        mask = np.load(rel_path(root, row_path(row, "mask"))).astype(np.uint8)
        inst = np.load(rel_path(root, row_path(row, "inst"))).astype(np.uint16)
        train_mask = np.zeros_like(mask, dtype=np.uint8)

        box_path = rel_path(root, row_path(row, "boxes"))
        box_obj = read_json(box_path)
        full_boxes = [box_with_threshold_flags(box, thresholds) for box in box_obj.get("boxes", [])]
        train_boxes = []
        box_obj["boxes"] = full_boxes
        with open(box_path, "w", encoding="utf-8") as f:
            json.dump(box_obj, f, indent=2, ensure_ascii=True)

        defects = []
        for defect in row.get("defects", []):
            defects.append(update_defect_flags(defect, thresholds))
        defect_by_id = {int(d.get("instance_id", -1)): d for d in defects}

        for box in full_boxes:
            instance_id = int(box.get("instance_id", -1))
            pixels = inst == instance_id
            if not bool(np.any(pixels)):
                continue
            visible = bool(box.get("panel_is_locally_visible", False))
            if visible:
                train_mask[pixels] = int(box.get("class_id", mask[pixels][0] if np.any(pixels) else 0))
                train_boxes.append(box)
                counts["train_boxes"] += 1
            else:
                train_mask[pixels] = IGNORE_LABEL
                counts["ignored_instances"] += 1
            if instance_id in defect_by_id:
                defect_by_id[instance_id]["segmentation_train_enabled"] = visible

        panel_id = str(row["panel_id"])
        train_mask_rel = row.get("train_mask") or f"panels/train_masks/{panel_id}_train_mask.npy"
        train_box_rel = row.get("train_boxes") or f"panels/train_boxes/{panel_id}_train_boxes.json"
        train_yolo_rel = row.get("train_yolo") or f"panels/train_yolo/{panel_id}.txt"
        train_mask_path = rel_path(root, train_mask_rel)
        train_box_path = rel_path(root, train_box_rel)
        train_yolo_path = rel_path(root, train_yolo_rel)
        train_mask_path.parent.mkdir(parents=True, exist_ok=True)
        train_box_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(train_mask_path, train_mask)
        with open(train_box_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "panel_id": panel_id,
                    "panel_size": row.get("panel_size"),
                    "boxes": train_boxes,
                    "source_boxes": str(row.get("boxes", "")),
                    "filter": "semu >= engineer_visible_threshold",
                    "threshold_file": str(Path(args.thresholds)),
                },
                f,
                indent=2,
                ensure_ascii=True,
            )
        write_yolo(train_yolo_path, train_boxes)

        row["train_mask"] = str(train_mask_rel).replace("\\", "/")
        row["train_boxes"] = str(train_box_rel).replace("\\", "/")
        row["train_yolo"] = str(train_yolo_rel).replace("\\", "/")
        row["defects"] = defects
        row["train_boxes_count"] = len(train_boxes)
        row["panel_has_train_defect"] = bool(np.any((train_mask > 0) & (train_mask < IGNORE_LABEL)))
        row["panel_has_ignored_defect"] = bool(np.any(train_mask == IGNORE_LABEL))
        updated_rows.append(row)
        counts["panels"] += 1
        counts["train_pixels"] += int(np.count_nonzero((train_mask > 0) & (train_mask < IGNORE_LABEL)))
        counts["ignore_pixels"] += int(np.count_nonzero(train_mask == IGNORE_LABEL))

    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in updated_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "training_target_update_summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
