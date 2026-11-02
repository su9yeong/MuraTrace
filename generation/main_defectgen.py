#!/usr/bin/env python3
"""Build a multimodal real-background hybrid Mura RCA benchmark.

This generator creates a dataset layout for five visual mask
classes (spot, line, region, gradient, band) and stores scale/subtype/polarity,
SEMU, reject, process, tool, fault mode, time window, and route graph as annotations.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from main_augmentbg import generate_background_bank, load_luma, save_luma_png, blur_float  # noqa: E402


MASK_COLORS = {
    1: (255, 80, 80),
    2: (80, 220, 120),
    3: (80, 150, 255),
    4: (255, 190, 40),
    5: (190, 90, 255),
}
TRAIN_IGNORE_LABEL = 255


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_size(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT")
    return int(parts[0]), int(parts[1])


def list_image_sizes(input_dir: Path) -> list[tuple[int, int]]:
    sizes = []
    for p in sorted(input_dir.iterdir()):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        with Image.open(p) as img:
            sizes.append(img.size)
    return sizes


def background_crop_edges(cfg: dict[str, Any]) -> dict[str, int]:
    crop = cfg.get("background", {}).get("crop_edges", {})
    return {
        "top": max(0, int(crop.get("top", 0))),
        "bottom": max(0, int(crop.get("bottom", 0))),
        "left": max(0, int(crop.get("left", 0))),
        "right": max(0, int(crop.get("right", 0))),
    }


def choose_target_size(input_dir: Path, cfg: dict[str, Any], cli_size: tuple[int, int] | None) -> tuple[int, int]:
    if cli_size is not None:
        return cli_size
    configured = cfg["geometry"].get("target_size")
    if configured:
        return int(configured[0]), int(configured[1])
    sizes = list_image_sizes(input_dir)
    if not sizes:
        raise FileNotFoundError(f"no background images found in {input_dir}")
    w, h = sizes[0]
    crop = background_crop_edges(cfg)
    w = max(1, w - crop["left"] - crop["right"])
    h = max(1, h - crop["top"] - crop["bottom"])
    factor = float(cfg["geometry"].get("resize_factor", 1.0))
    return max(32, int(round(w * factor))), max(32, int(round(h * factor)))


def prepare_output(root: Path, overwrite: bool) -> dict[str, Path]:
    dirs = {
        "backgrounds_clean": root / "backgrounds_clean",
        "backgrounds_aug": root / "backgrounds_aug",
        "glass_images": root / "glass" / "images",
        "glass_overlays": root / "glass" / "overlays",
        "glass_route_graphs": root / "glass" / "route_graphs",
        "panel_images": root / "panels" / "images",
        "panel_masks": root / "panels" / "masks",
        "panel_train_masks": root / "panels" / "train_masks",
        "panel_inst": root / "panels" / "inst",
        "panel_boxes": root / "panels" / "boxes",
        "panel_train_boxes": root / "panels" / "train_boxes",
        "panel_yolo": root / "panels" / "yolo",
        "panel_train_yolo": root / "panels" / "train_yolo",
        "reports": root / "reports",
        "previews": root / "previews",
        "preview_class": root / "previews" / "class_examples",
        "preview_semu": root / "previews" / "semu_sweep",
        "preview_rca": root / "previews" / "rca_examples",
        "preview_glass": root / "previews" / "glass_examples",
    }
    if root.exists() and overwrite:
        for sub in [
            "backgrounds_clean",
            "backgrounds_aug",
            "glass",
            "panels",
            "reports",
            "previews",
        ]:
            path = root / sub
            if path.exists():
                shutil.rmtree(path)
        for name in ["manifest.jsonl", "config.json", "dataset_card.md", "backgrounds_manifest.jsonl"]:
            path = root / name
            if path.exists():
                path.unlink()
    if root.exists() and not overwrite and (root / "manifest.jsonl").exists():
        raise FileExistsError(f"{root} already has manifest.jsonl; use --overwrite")
    for path in dirs.values():
        ensure_dir(path)
    return dirs


def assign_source_splits(source_ids: list[str], ratios: list[float], seed: int) -> dict[str, str]:
    ids = list(source_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    if n < 3:
        raise ValueError("source-level split needs at least 3 real background sources")
    ratios = [float(x) for x in ratios]
    total = sum(ratios)
    ratios = [x / total for x in ratios]
    n_train = max(1, int(round(n * ratios[0])))
    n_val = max(1, int(round(n * ratios[1])))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    out = {}
    for sid in ids[:n_train]:
        out[sid] = "train"
    for sid in ids[n_train : n_train + n_val]:
        out[sid] = "val"
    for sid in ids[n_train + n_val :]:
        out[sid] = "test"
    return out


def step_index(step: str, steps: list[str]) -> int:
    return int(steps.index(step))


def equip_index_map(cfg: dict[str, Any]) -> dict[str, int]:
    out = {}
    idx = 0
    for s_idx, n_tools in enumerate(cfg["process"]["tools_per_step"], start=1):
        for t_idx in range(1, int(n_tools) + 1):
            out[f"S{s_idx}_T{t_idx}"] = idx
            idx += 1
    return out


def make_route_graph(
    cfg: dict[str, Any],
    rng: np.random.Generator,
    required_tool_ids: list[str] | None = None,
    required_tool_cycles: list[tuple[str, int | None]] | None = None,
) -> dict[str, Any]:
    steps = list(cfg["process"]["steps"])
    tools_per_step = list(cfg["process"]["tools_per_step"])
    n_cycles = 3
    equip_index = equip_index_map(cfg)
    nodes = []
    route_node_idx = []
    for cycle in range(n_cycles):
        for step_i, step in enumerate(steps):
            tool = int(rng.integers(1, int(tools_per_step[step_i]) + 1))
            equip_id = f"S{step_i + 1}_T{tool}"
            node_pos = cycle * len(steps) + step_i
            route_node_idx.append(equip_index[equip_id])
            nodes.append(
                {
                    "cycle": cycle,
                    "step": step,
                    "step_index": step_i,
                    "tool_id": equip_id,
                    "equip_idx": equip_index[equip_id],
                    "node_pos": node_pos,
                }
            )
    placements: list[tuple[str, int | None]] = []
    if required_tool_cycles:
        placements.extend((str(tool), cycle) for tool, cycle in required_tool_cycles if tool)
    else:
        placements.extend((str(tool), None) for tool in sorted({t for t in (required_tool_ids or []) if t}))

    forced_slots: dict[int, str] = {}
    for tool_id, preferred_cycle in placements:
        try:
            step_part, tool_part = tool_id.split("_T", 1)
            step_i = int(step_part.removeprefix("S")) - 1
            tool = int(tool_part)
        except ValueError:
            continue
        if not (0 <= step_i < len(steps)):
            continue
        if not (1 <= tool <= int(tools_per_step[step_i])):
            continue
        cycle_order = list(range(n_cycles))
        if preferred_cycle is not None:
            preferred = int(np.clip(int(preferred_cycle), 0, n_cycles - 1))
            cycle_order = [preferred] + [c for c in cycle_order if c != preferred]
        else:
            start = int(rng.integers(0, n_cycles))
            cycle_order = [start] + [c for c in cycle_order if c != start]
        cycle: int | None = None
        for candidate_cycle in cycle_order:
            slot = candidate_cycle * len(steps) + step_i
            if slot not in forced_slots or forced_slots[slot] == tool_id:
                cycle = int(candidate_cycle)
                break
        if cycle is None:
            continue
        node_pos = cycle * len(steps) + step_i
        equip_id = f"S{step_i + 1}_T{tool}"
        forced_slots[node_pos] = equip_id
        nodes[node_pos].update(
            {
                "step": steps[step_i],
                "step_index": step_i,
                "tool_id": equip_id,
                "equip_idx": equip_index[equip_id],
            }
        )
        route_node_idx[node_pos] = equip_index[equip_id]
    edges = []
    for cycle in range(n_cycles):
        base = cycle * len(steps)
        for i in range(len(steps) - 1):
            edges.append([base + i, base + i + 1])
        if cycle < n_cycles - 1:
            edges.append([base + len(steps) - 1, (cycle + 1) * len(steps)])
    return {"nodes": nodes, "edges": edges, "route_node_idx": route_node_idx}


def active_faults(cfg: dict[str, Any], t_norm: float, n_total: int) -> list[dict[str, Any]]:
    faults = []
    for f in cfg["fault_library"].get("persistent", []):
        row = dict(f)
        row["fault_mode"] = "persistent"
        row["time_window"] = None
        row["time_window_frac"] = None
        faults.append(row)
    for f in cfg["fault_library"].get("excursion", []):
        lo, hi = [float(x) for x in f["window_frac"]]
        if lo <= t_norm <= hi:
            row = dict(f)
            row["fault_mode"] = "excursion"
            row["time_window_frac"] = [lo, hi]
            row["time_window"] = [int(round(lo * max(n_total - 1, 1))), int(round(hi * max(n_total - 1, 1)))]
            faults.append(row)
    return faults


def sample_weighted(items: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    weights = np.array([float(x.get("weight", 1.0)) for x in items], dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-12)
    return dict(items[int(rng.choice(len(items), p=weights))])


def sample_defect_count(cfg: dict[str, Any], rng: np.random.Generator) -> int:
    generation = cfg.get("generation", {})
    clean_prob = float(generation.get("clean_glass_probability", 0.0))
    if rng.random() < clean_prob:
        return 0

    profiles = generation.get("defect_count_profiles", {})
    active_name = str(generation.get("active_count_profile", "standard"))
    if active_name in profiles:
        profile = profiles[active_name]
        counts = np.array([int(k) for k in profile.keys()], dtype=np.int64)
        probs = np.array([float(profile[str(k)]) for k in counts], dtype=np.float64)
        probs = probs / max(float(probs.sum()), 1e-12)
        return int(rng.choice(counts, p=probs))

    fallback = generation.get("defects_per_glass", {"min": 1, "max": 4, "lambda": 2.1})
    return int(
        np.clip(
            rng.poisson(float(fallback.get("lambda", 2.1))) + 1,
            int(fallback.get("min", 1)),
            int(fallback.get("max", 4)),
        )
    )


def fault_allowed(candidate: dict[str, Any], selected: list[dict[str, Any]], constraints: dict[str, Any], strict: bool) -> bool:
    if not selected:
        return True
    if strict and constraints.get("no_repeat_subtype_per_glass", True):
        if str(candidate.get("subtype")) in {str(x.get("subtype")) for x in selected}:
            return False

    future = [*selected, candidate]
    max_global = int(constraints.get("max_global_defects", 99))
    if sum(1 for x in future if x.get("extent") == "global") > max_global:
        return False

    max_excursion = int(constraints.get("max_excursion_defects", 99))
    if sum(1 for x in future if x.get("fault_mode") == "excursion") > max_excursion:
        return False

    tools = {str(x.get("tool_id")) for x in future if x.get("tool_id")}
    if len(tools) > int(constraints.get("max_root_cause_tools", 99)):
        return False

    max_same_process = int(constraints.get("max_same_process", 99))
    processes = Counter(str(x.get("process_step")) for x in future if x.get("process_step"))
    if any(v > max_same_process for v in processes.values()):
        return False

    max_same_type = int(constraints.get("max_same_mask_type", 99))
    types = Counter(str(x.get("mask_type")) for x in future)
    if any(v > max_same_type for v in types.values()):
        return False
    return True


def sample_fault_for_defect(
    active_faults_for_time: list[dict[str, Any]],
    cfg: dict[str, Any],
    selected_faults: list[dict[str, Any]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    constraints = cfg.get("generation", {}).get("multi_cause_constraints", {})
    bg_prob = float(cfg["fault_library"].get("background_unattributed_probability", 0.0))

    for strict in [True, False]:
        for _ in range(80):
            fault = sample_weighted(active_faults_for_time, rng)
            if rng.random() < bg_prob:
                fault["fault_mode"] = "background"
                fault["tool_id"] = None
                fault["process_step"] = None
                fault["time_window"] = None
                fault["time_window_frac"] = None
            if fault_allowed(fault, selected_faults, constraints, strict=strict):
                return fault
    return sample_weighted(active_faults_for_time, rng)


def subtype_quota_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("generation", {}).get("subtype_quota_mode", {}))


def subtype_quota_enabled(cfg: dict[str, Any]) -> bool:
    return bool(subtype_quota_cfg(cfg).get("enabled", False))


def all_subtypes(cfg: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for values in cfg.get("subtypes", {}).values():
        out.extend(str(x) for x in values)
    return out


def quota_target_maps(cfg: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    q = subtype_quota_cfg(cfg)
    subtypes = all_subtypes(cfg)
    unique_default = int(q.get("target_unique_per_subtype", 0))
    reject_default = int(q.get("target_reject_per_subtype", 0))
    unique_by_subtype = q.get("target_unique_by_subtype", {})
    reject_by_subtype = q.get("target_reject_by_subtype", {})
    unique_targets = {s: int(unique_by_subtype.get(s, unique_default)) for s in subtypes}
    reject_targets = {s: int(reject_by_subtype.get(s, reject_default)) for s in subtypes}
    return unique_targets, reject_targets


def quota_remaining_total(quota_state: dict[str, Counter[str]], unique_targets: dict[str, int]) -> int:
    return int(sum(max(0, int(target) - int(quota_state["unique"][subtype])) for subtype, target in unique_targets.items()))


def sample_quota_defect_count(
    cfg: dict[str, Any],
    quota_state: dict[str, Counter[str]],
    rng: np.random.Generator,
) -> int:
    q = subtype_quota_cfg(cfg)
    unique_targets, _reject_targets = quota_target_maps(cfg)
    remaining = quota_remaining_total(quota_state, unique_targets)
    if remaining <= 0:
        return int(sample_defect_count(cfg, rng)) if q.get("continue_after_quota", False) else 0
    count_cfg = q.get("defects_per_glass", {"min": 6, "max": 6})
    lo = int(count_cfg.get("min", count_cfg.get("max", 6)))
    hi = int(count_cfg.get("max", lo))
    if hi < lo:
        hi = lo
    return int(min(remaining, int(rng.integers(lo, hi + 1))))


def quota_constraints(cfg: dict[str, Any]) -> dict[str, Any]:
    constraints = dict(cfg.get("generation", {}).get("multi_cause_constraints", {}))
    constraints.update(subtype_quota_cfg(cfg).get("multi_cause_constraints", {}))
    return constraints


def quota_fault_allowed(
    candidate: dict[str, Any],
    selected: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> bool:
    q = subtype_quota_cfg(cfg)
    per_subtype_max = {str(k): int(v) for k, v in q.get("max_instances_per_subtype_per_glass", {}).items()}
    subtype = str(candidate.get("subtype"))
    future_count = sum(1 for x in selected if str(x.get("subtype")) == subtype) + 1
    if subtype in per_subtype_max and future_count > per_subtype_max[subtype]:
        return False
    return fault_allowed(candidate, selected, quota_constraints(cfg), strict=False)


def quota_future_capacity(
    candidate_faults: list[dict[str, Any]],
    subtype: str,
    cfg: dict[str, Any],
    gidx: int,
    n_total: int,
) -> int:
    q = subtype_quota_cfg(cfg)
    per_subtype_max = {str(k): int(v) for k, v in q.get("max_instances_per_subtype_per_glass", {}).items()}
    max_per_glass = int(per_subtype_max.get(subtype, 1))
    if any(f.get("time_window") is None for f in candidate_faults):
        return max(1, (n_total - gidx) * max_per_glass)

    intervals: list[tuple[int, int]] = []
    for fault in candidate_faults:
        tw = fault.get("time_window")
        if not tw:
            continue
        lo, hi = int(tw[0]), int(tw[1])
        lo = max(lo, gidx)
        hi = min(hi, n_total - 1)
        if lo <= hi:
            intervals.append((lo, hi))
    if not intervals:
        return max_per_glass
    intervals.sort()
    merged: list[list[int]] = []
    for lo, hi in intervals:
        if not merged or lo > merged[-1][1] + 1:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)
    future_glasses = sum(hi - lo + 1 for lo, hi in merged)
    return max(1, future_glasses * max_per_glass)


def sample_fault_for_quota(
    active_faults_for_time: list[dict[str, Any]],
    cfg: dict[str, Any],
    selected_faults: list[dict[str, Any]],
    quota_state: dict[str, Counter[str]],
    rng: np.random.Generator,
    gidx: int,
    n_total: int,
) -> dict[str, Any] | None:
    unique_targets, reject_targets = quota_target_maps(cfg)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fault in active_faults_for_time:
        subtype = str(fault.get("subtype"))
        if int(quota_state["unique"][subtype]) >= int(unique_targets.get(subtype, 0)):
            continue
        if not quota_fault_allowed(fault, selected_faults, cfg):
            continue
        grouped[subtype].append(dict(fault))
    if not grouped:
        return None

    subtypes = sorted(grouped)
    weights = []
    for subtype in subtypes:
        unique_remaining = max(0, int(unique_targets[subtype]) - int(quota_state["unique"][subtype]))
        reject_remaining = max(0, int(reject_targets.get(subtype, 0)) - int(quota_state["reject"][subtype]))
        future_capacity = quota_future_capacity(grouped[subtype], subtype, cfg, gidx, n_total)
        urgency = float(unique_remaining) / max(float(future_capacity), 1.0)
        weights.append((float(unique_remaining) + 0.35 * float(reject_remaining)) * (1.0 + 12.0 * urgency * urgency))
    probs = np.array(weights, dtype=np.float64)
    probs = probs / max(float(probs.sum()), 1e-12)
    subtype = subtypes[int(rng.choice(len(subtypes), p=probs))]
    return sample_weighted(grouped[subtype], rng)


def semu_sample_for_quota(
    thr: dict[str, float],
    subtype: str,
    quota_state: dict[str, Counter[str]],
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> float:
    unique_targets, reject_targets = quota_target_maps(cfg)
    unique_remaining = max(1, int(unique_targets.get(subtype, 0)) - int(quota_state["unique"][subtype]))
    reject_remaining = max(0, int(reject_targets.get(subtype, 0)) - int(quota_state["reject"][subtype]))
    reject_probability = float(np.clip(reject_remaining / unique_remaining, 0.0, 1.0))
    visible = float(thr["engineer_visible"])
    reject = float(thr["reject_visible"])
    if rng.random() < reject_probability:
        return float(rng.uniform(reject, max(reject + 0.2, reject * 2.4)))
    return float(rng.uniform(visible, max(visible + 0.1, reject * 0.98)))


def visibility_threshold(thresholds: dict[str, Any], mask_type: str, subtype: str) -> dict[str, float]:
    d = dict(thresholds.get("default", {"engineer_visible": 0.5, "reject_visible": 1.0}))
    d.update(thresholds.get("by_mask_type", {}).get(mask_type, {}))
    d.update(thresholds.get("by_subtype", {}).get(subtype, {}))
    return {"engineer_visible": float(d["engineer_visible"]), "reject_visible": float(d["reject_visible"])}


def semu_sample(thr: dict[str, float], rng: np.random.Generator, reject_bias: float = 0.45) -> float:
    visible = float(thr["engineer_visible"])
    reject = float(thr["reject_visible"])
    if rng.random() < reject_bias:
        return float(rng.uniform(reject, max(reject + 0.2, reject * 3.0)))
    return float(rng.uniform(visible, max(visible + 0.1, reject * 0.98)))


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo + 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def soft_window(h: int, w: int, y0: int, y1: int, x0: int, x1: int, feather: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    f = max(1.0, float(feather))
    wy = np.clip((yy - y0) / f, 0, 1) * np.clip((y1 - 1 - yy) / f, 0, 1)
    wx = np.clip((xx - x0) / f, 0, 1) * np.clip((x1 - 1 - xx) / f, 0, 1)
    return (wy * wx).astype(np.float32)


def biased_center(
    rows: int,
    cols: int,
    ph: int,
    pw: int,
    rng: np.random.Generator,
    signature: dict[str, Any] | None = None,
) -> tuple[float, float]:
    h, w = rows * ph, cols * pw
    if not signature or "center_norm" not in signature:
        return float(rng.uniform(0.18 * h, 0.82 * h)), float(rng.uniform(0.18 * w, 0.82 * w))
    cy_n, cx_n = [float(x) for x in signature["center_norm"]]
    jy_n, jx_n = [float(x) for x in signature.get("jitter_norm", [0.12, 0.12])]
    cy = np.clip(rng.normal(cy_n, jy_n) * h, 0.05 * h, 0.95 * h)
    cx = np.clip(rng.normal(cx_n, jx_n) * w, 0.05 * w, 0.95 * w)
    return float(cy), float(cx)


def choose_extent_box(
    extent: str,
    rows: int,
    cols: int,
    ph: int,
    pw: int,
    rng: np.random.Generator,
    signature: dict[str, Any] | None = None,
) -> tuple[int, int, int, int]:
    if extent == "global":
        return 0, rows * ph, 0, cols * pw
    if extent == "regional":
        rh = int(rng.integers(1, min(3, rows) + 1))
        cw = int(rng.integers(2, min(4, cols) + 1))
    else:
        rh, cw = 1, 1
    cy, cx = biased_center(rows, cols, ph, pw, rng, signature)
    r0 = int(np.clip(round(cy / ph - rh / 2), 0, rows - rh))
    c0 = int(np.clip(round(cx / pw - cw / 2), 0, cols - cw))
    return r0 * ph, (r0 + rh) * ph, c0 * pw, (c0 + cw) * pw


def shape_spot(shape: tuple[int, int], box: tuple[int, int, int, int], panel_min: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    y0, y1, x0, x1 = box
    yy, xx = np.mgrid[0:h, 0:w]
    cx = float(rng.uniform(x0 + 0.25 * (x1 - x0), x0 + 0.75 * (x1 - x0)))
    cy = float(rng.uniform(y0 + 0.25 * (y1 - y0), y0 + 0.75 * (y1 - y0)))
    sigma = float(rng.uniform(0.006, 0.014) * panel_min)
    m = np.exp(-0.5 * (((xx - cx) / sigma) ** 2 + ((yy - cy) / sigma) ** 2)).astype(np.float32)
    return m, m > 0.20


def box_center_from_signature(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
    rng: np.random.Generator,
    signature: dict[str, Any] | None = None,
) -> tuple[float, float]:
    h, w = shape
    y0, y1, x0, x1 = box
    if signature and "center_norm" in signature:
        cy_n, cx_n = [float(x) for x in signature["center_norm"]]
        jy_n, jx_n = [float(x) for x in signature.get("jitter_norm", [0.12, 0.12])]
        cy = float(np.clip(rng.normal(cy_n, jy_n) * h, y0 + 0.12 * (y1 - y0), y1 - 0.12 * (y1 - y0)))
        cx = float(np.clip(rng.normal(cx_n, jx_n) * w, x0 + 0.12 * (x1 - x0), x1 - 0.12 * (x1 - x0)))
        return cy, cx
    return (
        float(rng.uniform(y0 + 0.25 * (y1 - y0), y0 + 0.75 * (y1 - y0))),
        float(rng.uniform(x0 + 0.25 * (x1 - x0), x0 + 0.75 * (x1 - x0))),
    )


def shape_line(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
    panel_min: int,
    subtype: str,
    extent: str,
    cols: int,
    pw: int,
    rng: np.random.Generator,
    signature: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    y0, y1, x0, x1 = box
    yy, xx = np.mgrid[0:h, 0:w]
    width = max(1.0, float(rng.uniform(0.0025, 0.0055) * panel_min))
    if subtype == "stitch_line":
        seam = int(rng.integers(1, cols)) * pw
        off_lo, off_hi = [float(x) for x in (signature or {}).get("stitch_offset_frac", [0.08, 0.18])]
        jitter = float((signature or {}).get("jitter_norm", [0.02, 0.02])[1]) * pw
        shift = float(rng.choice([-1.0, 1.0]) * rng.uniform(off_lo, off_hi) * pw)
        pos = seam + shift + float(rng.normal(0.0, jitter))
        line = np.exp(-0.5 * ((xx - pos) / width) ** 2)
        y0, y1 = 0, h
        x0, x1 = max(0, int(pos - 0.35 * pw)), min(w, int(pos + 0.35 * pw))
    else:
        vertical = bool(rng.random() < 0.65)
        if vertical:
            _cy, cx = box_center_from_signature(shape, box, rng, signature)
            pos = cx
            line = np.exp(-0.5 * ((xx - pos) / width) ** 2)
        else:
            cy, _cx = box_center_from_signature(shape, box, rng, signature)
            pos = cy
            line = np.exp(-0.5 * ((yy - pos) / width) ** 2)
    win = soft_window(h, w, y0, y1, x0, x1, feather=0.18 * panel_min)
    m = (line * win).astype(np.float32)
    return m, m > 0.16


def shape_region(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
    subtype: str,
    rng: np.random.Generator,
    signature: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    y0, y1, x0, x1 = box
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = box_center_from_signature(shape, box, rng, signature)
    if subtype == "thermal_blob":
        # Broad, smooth thermal non-uniformity. Keep it visually distinct from
        # puddle defects by avoiding small lobes and irregular edges.
        sx = float(rng.uniform(0.22, 0.34) * max(x1 - x0, 1))
        sy = float(rng.uniform(0.20, 0.32) * max(y1 - y0, 1))
        theta = float(rng.uniform(0, math.pi))
        xr = (xx - cx) * math.cos(theta) + (yy - cy) * math.sin(theta)
        yr = -(xx - cx) * math.sin(theta) + (yy - cy) * math.cos(theta)
        m = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2)).astype(np.float32)
        m = blur_float(m, radius=max(2.0, 0.010 * min(h, w)))
        m = normalize(m)
        mask_threshold = float((signature or {}).get("thermal_mask_threshold", 0.10))
        return m, m > mask_threshold

    # Localized wet/puddle-like blob: smaller, irregular, and not glass-wide.
    sx = float(rng.uniform(0.055, 0.105) * max(x1 - x0, 1))
    sy = float(rng.uniform(0.060, 0.120) * max(y1 - y0, 1))
    theta = float(rng.uniform(0, math.pi))
    xr = (xx - cx) * math.cos(theta) + (yy - cy) * math.sin(theta)
    yr = -(xx - cx) * math.sin(theta) + (yy - cy) * math.cos(theta)
    m = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2)).astype(np.float32)
    for _ in range(int(rng.integers(1, 4))):
        ox = float(rng.normal(0.0, 0.65 * sx))
        oy = float(rng.normal(0.0, 0.65 * sy))
        sx2 = sx * float(rng.uniform(0.35, 0.70))
        sy2 = sy * float(rng.uniform(0.35, 0.70))
        xr2 = (xx - (cx + ox)) * math.cos(theta) + (yy - (cy + oy)) * math.sin(theta)
        yr2 = -(xx - (cx + ox)) * math.sin(theta) + (yy - (cy + oy)) * math.cos(theta)
        m += float(rng.uniform(0.18, 0.38)) * np.exp(-0.5 * ((xr2 / sx2) ** 2 + (yr2 / sy2) ** 2))
    noise = blur_float(rng.normal(0.0, 1.0, (h, w)).astype(np.float32), radius=max(2.0, 0.012 * min(h, w)))
    noise = 0.82 + 0.30 * normalize(noise)
    m = normalize(m * noise)
    m *= soft_window(h, w, y0, y1, x0, x1, feather=0.10 * min(y1 - y0, x1 - x0))
    m = normalize(m)
    return m, m > 0.16


def shape_gradient(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
    rng: np.random.Generator,
    signature: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    edge_width_frac = float((signature or {}).get("edge_width_frac", 0.14))
    sigma_frac = float((signature or {}).get("edge_radial_sigma_frac", 0.34))
    corner_gate_floor = float((signature or {}).get("corner_edge_gate_floor", 0.42))
    blur_frac = float((signature or {}).get("edge_gradient_blur_frac", 0.012))
    panel_min = float(min(h, w))
    edge_width = max(8.0, edge_width_frac * panel_min)
    sigma = max(edge_width, sigma_frac * panel_min)
    anchors = [
        (0.0, 0.0),
        (0.0, float(w - 1)),
        (float(h - 1), 0.0),
        (float(h - 1), float(w - 1)),
        (0.0, float(w - 1) * 0.5),
        (float(h - 1), float(w - 1) * 0.5),
        (float(h - 1) * 0.5, 0.0),
        (float(h - 1) * 0.5, float(w - 1)),
    ]
    ay, ax = anchors[int(rng.integers(0, len(anchors)))]
    is_corner_anchor = (ay in {0.0, float(h - 1)}) and (ax in {0.0, float(w - 1)})
    dist = np.sqrt((yy.astype(np.float32) - ay) ** 2 + (xx.astype(np.float32) - ax) ** 2)
    radial = np.exp(-0.5 * (dist / sigma) ** 2).astype(np.float32)
    edge_dist = np.minimum.reduce(
        [
            yy.astype(np.float32),
            xx.astype(np.float32),
            float(h - 1) - yy.astype(np.float32),
            float(w - 1) - xx.astype(np.float32),
        ]
    )
    edge_gate = np.clip(1.0 - edge_dist / edge_width, 0.0, 1.0)
    edge_gate = edge_gate * edge_gate * (3.0 - 2.0 * edge_gate)
    if is_corner_anchor:
        # Corner anneal gradients should look like a smooth quadrant field.
        # A hard edge gate creates an L-shaped bottom/right pair with a weak
        # diagonal valley in the joining panel, which is not a physical boundary.
        gate = corner_gate_floor + (1.0 - corner_gate_floor) * edge_gate
    else:
        gate = edge_gate
    m = radial * gate.astype(np.float32)
    m = blur_float(m.astype(np.float32), radius=max(2.0, blur_frac * panel_min))
    m = normalize(m)
    return m, m > float((signature or {}).get("edge_mask_threshold", 0.12))


def shape_band(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
    subtype: str,
    panel_min: int,
    panel_long: int,
    rng: np.random.Generator,
    signature: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    h, w = shape
    y0, y1, x0, x1 = box
    yy, xx = np.mgrid[0:h, 0:w]
    win = soft_window(h, w, y0, y1, x0, x1, feather=0.18 * panel_min)
    meta: dict[str, Any] = {}
    width_range = (signature or {}).get("band_width_frac_long_side_range", [0.025, 0.075])
    if not isinstance(width_range, list) or len(width_range) != 2:
        width_range = [0.025, 0.075]
    width_lo = max(0.0, float(width_range[0]))
    width_hi = max(width_lo, float(width_range[1]))
    width_max_frac = float((signature or {}).get("band_width_max_frac_long_side", 0.095))
    width_period_cap = float((signature or {}).get("band_width_max_frac_period", 0.18))
    width_min_px = float((signature or {}).get("band_width_min_px", 3.0))

    def sample_band_width(period_px: float) -> float:
        width = float(rng.uniform(width_lo, width_hi) * max(float(panel_long), 1.0))
        width = min(width, max(width_min_px, width_max_frac * max(float(panel_long), 1.0)))
        width = min(width, max(width_min_px, width_period_cap * max(float(period_px), 1.0)))
        return max(width_min_px, width)

    if subtype == "row_banding":
        x0, x1 = 0, w
        win = soft_window(h, w, y0, y1, x0, x1, feather=0.16 * panel_min)
        n_bands = int(rng.integers(2, 5))
        length = max(float(y1 - y0), 1.0)
        period = length / float(n_bands + rng.uniform(0.15, 0.55))
        width = sample_band_width(period)
        phase = float(rng.uniform(0.20, 0.65) * period)
        centers = np.arange(y0 + phase, y1 + period, period)[: max(2, n_bands + 1)]
        m = np.zeros((h, w), dtype=np.float32)
        for c in centers:
            m += np.exp(-0.5 * ((yy - c) / width) ** 2)
        meta = {"band_orientation": "row", "band_period_px": period, "band_width_px": width, "band_count": int(len(centers))}
    elif subtype == "column_banding":
        y0, y1 = 0, h
        win = soft_window(h, w, y0, y1, x0, x1, feather=0.16 * panel_min)
        n_bands = int(rng.integers(2, 5))
        length = max(float(x1 - x0), 1.0)
        period = length / float(n_bands + rng.uniform(0.15, 0.55))
        width = sample_band_width(period)
        phase = float(rng.uniform(0.20, 0.65) * period)
        centers = np.arange(x0 + phase, x1 + period, period)[: max(2, n_bands + 1)]
        m = np.zeros((h, w), dtype=np.float32)
        for c in centers:
            m += np.exp(-0.5 * ((xx - c) / width) ** 2)
        meta = {"band_orientation": "column", "band_period_px": period, "band_width_px": width, "band_count": int(len(centers))}
    else:
        # Low-frequency banding must be periodic: at least two broad
        # peak/trough bands, not a single ramp.
        orientation = str((signature or {}).get("band_orientation", "lowfreq"))
        n_bands = int(rng.integers(2, 4))
        horizontal = bool(rng.random() < 0.65)
        axis = yy if horizontal else xx
        lo, hi = (y0, y1) if horizontal else (x0, x1)
        length = max(float(hi - lo), 1.0)
        period = length / float(n_bands + rng.uniform(0.15, 0.55))
        width = sample_band_width(period)
        phase = float(rng.uniform(0.2, 0.8) * period)
        centers = np.arange(lo + phase, hi + period, period)
        m = np.zeros((h, w), dtype=np.float32)
        for idx, c in enumerate(centers[: max(2, min(len(centers), 4))]):
            amp = 1.0 if idx % 2 == 0 else 0.72
            m += amp * np.exp(-0.5 * ((axis - c) / width) ** 2)
        m = blur_float(m.astype(np.float32), radius=max(2.0, 0.010 * panel_min))
        meta = {
            "band_orientation": orientation,
            "band_period_px": period,
            "band_width_px": width,
            "band_count": int(min(len(centers), 4)),
        }
    m = normalize(m * win)
    mask_threshold = float((signature or {}).get("band_mask_threshold", 0.20))
    return m, (m > mask_threshold) & (win > 0.05), meta


def make_shape(
    mask_type: str,
    subtype: str,
    extent: str,
    rows: int,
    cols: int,
    ph: int,
    pw: int,
    rng: np.random.Generator,
    signature: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    glass_shape = (rows * ph, cols * pw)
    box = choose_extent_box(extent, rows, cols, ph, pw, rng, signature)
    panel_min = min(ph, pw)
    panel_long = max(ph, pw)
    if mask_type == "spot":
        shape, mask = shape_spot(glass_shape, box, panel_min, rng)
        meta = {}
    elif mask_type == "line":
        shape, mask = shape_line(glass_shape, box, panel_min, subtype, extent, cols, pw, rng, signature)
        meta = {}
    elif mask_type == "region":
        shape, mask = shape_region(glass_shape, box, subtype, rng, signature)
        meta = {}
    elif mask_type == "gradient":
        shape, mask = shape_gradient(glass_shape, box, rng, signature)
        meta = {}
    elif mask_type == "band":
        shape, mask, meta = shape_band(glass_shape, box, subtype, panel_min, panel_long, rng, signature)
    else:
        raise ValueError(f"unknown mask_type {mask_type}")
    shape = normalize(shape)
    ys, xs = np.where(mask)
    if len(xs):
        meta["centroid_norm"] = [round(float(ys.mean()) / max(glass_shape[0], 1), 5), round(float(xs.mean()) / max(glass_shape[1], 1), 5)]
    else:
        meta["centroid_norm"] = [None, None]
    meta["extent_box_px"] = [int(box[2]), int(box[0]), int(box[3]), int(box[1])]
    return shape.astype(np.float32), mask.astype(bool), meta


def defect_zorder_key(item: tuple[np.ndarray, np.ndarray, dict[str, Any]]) -> tuple[int, float]:
    """Class-priority z-order: smooth-field defects are drawn first (bottom) and
    structural/compact defects last (top).

    Class priority is the PRIMARY key so overlap labels are deterministic and do
    not flip with total area: e.g. a tall column band is no longer overwritten by
    an overlapping panel-filling gradient. Within the same class, larger area is
    drawn first so smaller cores remain visible on top. Later defects overwrite
    earlier labels in the semantic/instance maps; the image composition follows
    the same order.
    """
    _shape_map, hard_mask, defect = item
    type_rank = {
        "gradient": 0,  # smooth luminance field -> bottom
        "band": 1,
        "region": 2,
        "line": 3,
        "spot": 4,      # compact -> top
    }
    area = float(np.count_nonzero(hard_mask))
    return (type_rank.get(str(defect.get("mask_type")), 2), -area)


def alpha_from_semu(cfg: dict[str, Any], mask_type: str, semu: float, subtype: str | None = None) -> float:
    comp = cfg["composition"]
    per = float(comp.get("alpha_per_semu", {}).get(mask_type, 0.005))
    cap = float(comp.get("alpha_cap", {}).get(mask_type, 0.05))
    if subtype:
        per = float(comp.get("alpha_per_semu_subtype", {}).get(subtype, per))
        cap = float(comp.get("alpha_cap_subtype", {}).get(subtype, cap))
    return float(np.clip(per * float(semu), 0.001, cap))


def alpha_per_semu(cfg: dict[str, Any], mask_type: str, subtype: str | None = None) -> float:
    comp = cfg["composition"]
    per = float(comp.get("alpha_per_semu", {}).get(mask_type, 0.005))
    if subtype:
        per = float(comp.get("alpha_per_semu_subtype", {}).get(subtype, per))
    return float(max(per, 1e-12))


def local_semu_stats(
    local_semu: np.ndarray,
    hard_mask: np.ndarray,
    visible_threshold: float,
    reject_threshold: float,
) -> dict[str, Any]:
    hard = hard_mask.astype(bool)
    visible = hard & (local_semu >= float(visible_threshold))
    hard_values = local_semu[hard]
    visible_values = local_semu[visible]
    panel_peak = float(hard_values.max()) if hard_values.size else 0.0
    hard_p95 = float(np.percentile(hard_values, 95)) if hard_values.size else 0.0
    panel_instance_semu = float(np.percentile(visible_values, 95)) if visible_values.size else 0.0
    return {
        "visible_mask": visible,
        "panel_semu_peak": panel_peak,
        "panel_semu_p95_hard_mask": hard_p95,
        "panel_semu_p95_visible_core": panel_instance_semu,
        "panel_instance_semu": panel_instance_semu,
        "is_visible": bool(visible_values.size > 0),
        "is_reject": bool(panel_instance_semu >= float(reject_threshold)),
        "hard_mask_area_px": int(np.count_nonzero(hard)),
        "visible_mask_area_px": int(np.count_nonzero(visible)),
        "hard_mask_area_fraction": float(np.count_nonzero(hard) / hard.size),
        "visible_mask_area_fraction": float(np.count_nonzero(visible) / visible.size),
    }


def panel_label_semu_from_stats(stats: dict[str, Any]) -> float:
    return float(stats.get("panel_semu_p95_hard_mask", 0.0))


def label_tail_fraction(cfg: dict[str, Any], defect: dict[str, Any]) -> float:
    generation = cfg.get("generation", {})
    frac = float(generation.get("label_mask_tail_fraction", 0.70))
    by_type = generation.get("label_mask_tail_fraction_by_mask_type", {})
    by_subtype = generation.get("label_mask_tail_fraction_by_subtype", {})
    mask_type = str(defect.get("mask_type", ""))
    subtype = str(defect.get("subtype", ""))
    if mask_type in by_type:
        frac = float(by_type[mask_type])
    if subtype in by_subtype:
        frac = float(by_subtype[subtype])
    return float(np.clip(frac, 0.05, 1.0))


def label_max_expansion_ratio(cfg: dict[str, Any], defect: dict[str, Any]) -> float:
    generation = cfg.get("generation", {})
    ratio = float(generation.get("label_mask_max_expansion_ratio", 4.0))
    by_type = generation.get("label_mask_max_expansion_ratio_by_mask_type", {})
    by_subtype = generation.get("label_mask_max_expansion_ratio_by_subtype", {})
    mask_type = str(defect.get("mask_type", ""))
    subtype = str(defect.get("subtype", ""))
    if mask_type in by_type:
        ratio = float(by_type[mask_type])
    if subtype in by_subtype:
        ratio = float(by_subtype[subtype])
    return float(max(1.0, ratio))


def label_tail_radius_px(cfg: dict[str, Any], defect: dict[str, Any]) -> int:
    generation = cfg.get("generation", {})
    radius = int(generation.get("label_mask_tail_radius_px", 8))
    by_type = generation.get("label_mask_tail_radius_px_by_mask_type", {})
    by_subtype = generation.get("label_mask_tail_radius_px_by_subtype", {})
    mask_type = str(defect.get("mask_type", ""))
    subtype = str(defect.get("subtype", ""))
    if mask_type in by_type:
        radius = int(by_type[mask_type])
    if subtype in by_subtype:
        radius = int(by_subtype[subtype])
    return max(0, radius)


def dilate_once(mask: np.ndarray) -> np.ndarray:
    src = mask.astype(bool)
    out = src.copy()
    out[1:, :] |= src[:-1, :]
    out[:-1, :] |= src[1:, :]
    out[:, 1:] |= src[:, :-1]
    out[:, :-1] |= src[:, 1:]
    out[1:, 1:] |= src[:-1, :-1]
    out[1:, :-1] |= src[:-1, 1:]
    out[:-1, 1:] |= src[1:, :-1]
    out[:-1, :-1] |= src[1:, 1:]
    return out


def cap_by_local_semu(mask: np.ndarray, local_semu: np.ndarray, max_area: int, keep_mask: np.ndarray) -> np.ndarray:
    if max_area <= 0 or int(np.count_nonzero(mask)) <= max_area:
        return mask
    keep = keep_mask.astype(bool)
    out = keep.copy()
    remaining = max_area - int(np.count_nonzero(out))
    if remaining <= 0:
        return out
    extra = mask.astype(bool) & ~out
    values = local_semu[extra]
    if values.size <= remaining:
        out |= extra
        return out
    cutoff = float(np.partition(values, values.size - remaining)[values.size - remaining])
    out |= extra & (local_semu >= cutoff)
    if int(np.count_nonzero(out)) > max_area:
        extra = out & ~keep
        ys, xs = np.where(extra)
        if len(xs) > 0:
            order = np.argsort(local_semu[ys, xs])[::-1]
            chosen = order[:remaining]
            capped = keep.copy()
            capped[ys[chosen], xs[chosen]] = True
            out = capped
    return out


def grow_from_visible_core(
    visible_mask: np.ndarray,
    allowed_mask: np.ndarray,
    local_semu: np.ndarray,
    radius_px: int,
    max_expansion_ratio: float | None,
) -> np.ndarray:
    visible = visible_mask.astype(bool)
    allowed = allowed_mask.astype(bool) | visible
    visible_area = int(np.count_nonzero(visible))
    if visible_area == 0 or radius_px <= 0:
        return visible
    max_area = 0
    if max_expansion_ratio is not None:
        max_area = int(math.ceil(visible_area * float(max_expansion_ratio)))
    grown = visible.copy()
    for _ in range(int(radius_px)):
        expanded = dilate_once(grown) & allowed
        if np.array_equal(expanded, grown):
            break
        if max_area > 0 and int(np.count_nonzero(expanded)) > max_area:
            return cap_by_local_semu(expanded, local_semu, max_area, visible)
        grown = expanded
    if max_area > 0:
        grown = cap_by_local_semu(grown, local_semu, max_area, visible)
    return grown


def label_mask_from_policy(
    policy: str,
    hard_mask: np.ndarray,
    visible_mask: np.ndarray,
    local_semu: np.ndarray | None = None,
    visible_threshold: float | None = None,
    tail_fraction: float = 0.70,
    max_expansion_ratio: float | None = None,
    tail_radius_px: int = 0,
) -> np.ndarray:
    hard = hard_mask.astype(bool)
    if policy in {"hard", "hard_mask", "hard_visible_instance"}:
        return hard
    if policy in {"local_visible", "local_visible_semu", "semu_visible"}:
        return visible_mask.astype(bool)
    if policy in {"semu_tail", "semu_tail_visible_instance", "local_semu_tail"}:
        if local_semu is None or visible_threshold is None:
            return visible_mask.astype(bool)
        tail_threshold = float(visible_threshold) * float(tail_fraction)
        candidate = hard & (local_semu >= tail_threshold)
        visible = visible_mask.astype(bool)
        visible_area = int(np.count_nonzero(visible))
        if max_expansion_ratio is not None and visible_area > 0:
            max_area = int(math.ceil(visible_area * float(max_expansion_ratio)))
            candidate_area = int(np.count_nonzero(candidate))
            if candidate_area > max_area:
                values = local_semu[candidate]
                if values.size > max_area:
                    cutoff = float(np.partition(values, values.size - max_area)[values.size - max_area])
                    candidate = candidate & (local_semu >= cutoff)
                    candidate |= visible
        return candidate
    if policy in {"core_tail", "core_tail_visible_instance", "visible_core_tail"}:
        if local_semu is None or visible_threshold is None:
            return visible_mask.astype(bool)
        tail_threshold = float(visible_threshold) * float(tail_fraction)
        allowed = hard & (local_semu >= tail_threshold)
        return grow_from_visible_core(
            visible_mask.astype(bool),
            allowed,
            local_semu,
            int(tail_radius_px),
            max_expansion_ratio,
        )
    raise ValueError(f"unknown label_mask_policy: {policy}")


def compose(background: np.ndarray, shape_map: np.ndarray, polarity: str, alpha: float) -> np.ndarray:
    if polarity == "bright":
        return np.clip(background * (1.0 + alpha * shape_map), 0.0, 1.0)
    return np.clip(background * (1.0 - alpha * shape_map), 0.0, 1.0)


def match_panel_luminance(panel: np.ndarray, ref_mean: float, ref_std: float) -> np.ndarray:
    """Align random panel backgrounds so glass assembly does not look tiled.

    Texture is preserved, but large source-to-source mean/contrast jumps are
    reduced before synthetic defects are composed.
    """
    src_mean = float(panel.mean())
    src_std = float(panel.std())
    if src_std < 1e-6:
        return np.clip(panel - src_mean + ref_mean, 0.0, 1.0).astype(np.float32)
    scale = float(np.clip(ref_std / src_std, 0.65, 1.45))
    out = (panel - src_mean) * scale + ref_mean
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def prepare_synthesis_background(panel: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """Optionally flatten strong low-frequency background before defect composition.

    Real panel captures can contain gradients as strong as the synthetic gradient/band defects. 
    For clean weak synthesis, keep sensor texture/noise but suppress broad shading so SEMU controls defect visibility.
    """
    bg_cfg = cfg.get("background", {})
    profile = str(bg_cfg.get("synthesis_profile", "real"))
    if profile in {"synthetic_clean_weak", "synthetic_cleanweak"}:
        params = dict(bg_cfg.get("clean_weak", {}))
        h, w = panel.shape[:2]
        target_mean = float(params.get("target_mean", 0.52))
        noise_sigma = float(params.get("noise_sigma", 0.006))
        out = np.full((h, w), target_mean, dtype=np.float32)
        row_amp = float(params.get("row_amp", 0.0005))
        col_amp = float(params.get("col_amp", 0.0005))
        if row_amp > 0:
            row = blur_float(rng.normal(0.0, 1.0, (h, 1)).astype(np.float32), radius=max(2.0, 0.05 * h))
            row = row / max(float(np.max(np.abs(row))), 1e-6)
            out = out + row_amp * row
        if col_amp > 0:
            col = blur_float(rng.normal(0.0, 1.0, (1, w)).astype(np.float32), radius=max(2.0, 0.05 * w))
            col = col / max(float(np.max(np.abs(col))), 1e-6)
            out = out + col_amp * col
        if noise_sigma > 0:
            out = out + rng.normal(0.0, noise_sigma, out.shape).astype(np.float32)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    if profile not in {"clean_weak", "cleanweak"}:
        return np.clip(panel, 0.0, 1.0).astype(np.float32)

    params = dict(bg_cfg.get("clean_weak", {}))
    out = panel.astype(np.float32, copy=True)
    h, w = out.shape[:2]
    radius_frac = float(params.get("lowfreq_radius_frac", 0.18))
    radius = max(8.0, radius_frac * float(min(h, w)))
    low = blur_float(out, radius=radius)
    mean = float(out.mean())
    texture_gain = float(params.get("texture_gain", 0.35))
    lowfreq_gain = float(params.get("lowfreq_gain", 0.04))
    out = mean + texture_gain * (out - low) + lowfreq_gain * (low - float(low.mean()))

    row_amp = float(params.get("row_amp", 0.0))
    col_amp = float(params.get("col_amp", 0.0))
    if row_amp > 0:
        row = blur_float(rng.normal(0.0, 1.0, (h, 1)).astype(np.float32), radius=max(2.0, 0.04 * h))
        row = row / max(float(np.max(np.abs(row))), 1e-6)
        out = out + row_amp * row
    if col_amp > 0:
        col = blur_float(rng.normal(0.0, 1.0, (1, w)).astype(np.float32), radius=max(2.0, 0.04 * w))
        col = col / max(float(np.max(np.abs(col))), 1e-6)
        out = out + col_amp * col

    noise_sigma = float(params.get("noise_sigma", 0.0018))
    if noise_sigma > 0:
        out = out + rng.normal(0.0, noise_sigma, out.shape).astype(np.float32)

    target_mean = params.get("target_mean")
    if target_mean is not None:
        out = out - float(out.mean()) + float(target_mean)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def choose_primary_route_node(
    route_graph: dict[str, Any],
    tool_id: str | None,
    rng: np.random.Generator,
    preferred_cycle: int | None = None,
) -> dict[str, Any] | None:
    if not tool_id:
        return None
    matches = [node for node in route_graph["nodes"] if str(node.get("tool_id")) == str(tool_id)]
    if not matches:
        return None
    if preferred_cycle is not None:
        cycle_matches = [node for node in matches if int(node.get("cycle", -1)) == int(preferred_cycle)]
        if cycle_matches:
            return dict(cycle_matches[int(rng.integers(0, len(cycle_matches)))])
    return dict(matches[int(rng.integers(0, len(matches)))])


def tool_step_index(tool_id: str | None) -> int | None:
    if not tool_id:
        return None
    try:
        step_part, _tool_part = str(tool_id).split("_T", 1)
        return int(step_part.removeprefix("S")) - 1
    except ValueError:
        return None


def preferred_cycle_for_defect(defect: dict[str, Any], rng: np.random.Generator, n_cycles: int = 3) -> int:
    tw_frac = defect.get("time_window_frac")
    if isinstance(tw_frac, list) and len(tw_frac) == 2:
        lo, hi = float(tw_frac[0]), float(tw_frac[1])
        if hi > lo:
            rel = (float(defect.get("t_norm", 0.5)) - lo) / max(hi - lo, 1e-6)
            return int(np.clip(math.floor(rel * n_cycles), 0, n_cycles - 1))
    return int(rng.integers(0, n_cycles))


def crop_panel(arr: np.ndarray, row: int, col: int, ph: int, pw: int) -> np.ndarray:
    return arr[row * ph : (row + 1) * ph, col * pw : (col + 1) * pw]


def overlay_mask_u16(img_u16: np.ndarray, mask: np.ndarray) -> Image.Image:
    base = np.clip(img_u16.astype(np.float32) / max(float(img_u16.max(initial=1)), 1.0) * 255.0, 0, 255).astype(np.uint8)
    rgb = np.repeat(base[..., None], 3, axis=2).astype(np.float32)
    for cls_id, color in MASK_COLORS.items():
        m = mask == cls_id
        if np.any(m):
            rgb[m] = 0.55 * rgb[m] + 0.45 * np.array(color, dtype=np.float32)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def save_u16_npy(arr: np.ndarray, path: Path, bit_depth: int) -> None:
    maxv = (2**int(bit_depth)) - 1
    out = np.clip(arr * maxv + 0.5, 0, maxv).astype(np.uint16)
    np.save(path, out)


def save_u16_png(arr: np.ndarray, path: Path, bit_depth: int) -> None:
    maxv = (2**int(bit_depth)) - 1
    out = np.clip(arr * maxv + 0.5, 0, maxv).astype(np.uint16)
    if maxv != 65535:
        out = (out.astype(np.float32) / maxv * 65535.0 + 0.5).astype(np.uint16)
    Image.fromarray(out, mode="I;16").save(path)


def bboxes_from_inst(mask: np.ndarray, inst: np.ndarray, defects: list[dict[str, Any]], pw: int, ph: int) -> list[dict[str, Any]]:
    by_id = {int(d["instance_id"]): d for d in defects}
    boxes = []
    for instance_id in sorted(int(x) for x in np.unique(inst) if int(x) > 0):
        d = by_id.get(instance_id)
        if not d:
            continue
        ys, xs = np.where(inst == instance_id)
        if len(xs) == 0:
            continue
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        boxes.append(
            {
                "instance_id": instance_id,
                "mask_type": d["mask_type"],
                "class_id": int(d["class_id"]),
                "extent": d["extent"],
                "extent_id": int(d["extent_id"]),
                "subtype": d["subtype"],
                "polarity": d["polarity"],
                "semu": round(float(d["semu"]), 4),
                "panel_label_semu": round(float(d.get("panel_label_semu", d["semu"])), 4),
                "panel_instance_semu": round(float(d.get("panel_instance_semu", 0.0)), 4),
                "panel_semu_peak": round(float(d.get("panel_semu_peak", 0.0)), 4),
                "panel_semu_p95_hard_mask": round(float(d.get("panel_semu_p95_hard_mask", 0.0)), 4),
                "panel_semu_p95_visible_core": round(float(d.get("panel_semu_p95_visible_core", 0.0)), 4),
                "panel_is_locally_visible": bool(d.get("panel_is_locally_visible", False)),
                "panel_is_locally_reject": bool(d.get("panel_is_locally_reject", d["is_reject"])),
                "instance_is_visible": bool(d.get("instance_is_visible", d.get("is_visible", False))),
                "instance_is_reject": bool(d.get("instance_is_reject", False)),
                "is_visible": bool(d.get("is_visible", False)),
                "is_reject": bool(d["is_reject"]),
                "process_step": d["process_step"],
                "tool_id": d["tool_id"],
                "fault_mode": d["fault_mode"],
                "time_window": d["time_window"],
                "bbox": [x0, y0, x1, y1],
                "bbox_norm": [round(x0 / pw, 6), round(y0 / ph, 6), round(x1 / pw, 6), round(y1 / ph, 6)],
                "yolo": [
                    round((x0 + x1) / 2 / pw, 6),
                    round((y0 + y1) / 2 / ph, 6),
                    round((x1 - x0) / pw, 6),
                    round((y1 - y0) / ph, 6),
                ],
                "area_px": int(len(xs)),
                "bbox_rect_area_px": int((x1 - x0) * (y1 - y0)),
            }
        )
    return boxes


def write_yolo(path: Path, boxes: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for b in boxes:
            cls = int(b["class_id"]) - 1
            cx, cy, w, h = b["yolo"]
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def font(size: int) -> ImageFont.ImageFont:
    for p in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/malgun.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def make_preview_grid(paths: list[Path], out_path: Path, cols: int = 5, thumb_w: int = 320) -> None:
    paths = [p for p in paths if p.exists()]
    if not paths:
        return
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        thumb_h = max(64, int(round(thumb_w * img.height / max(img.width, 1))))
        imgs.append(ImageOps.fit(img, (thumb_w, thumb_h), method=Image.Resampling.BICUBIC))
    thumb_h = imgs[0].height
    rows = int(math.ceil(len(imgs) / cols))
    grid = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (245, 245, 245))
    for i, img in enumerate(imgs):
        grid.paste(img, ((i % cols) * thumb_w, (i // cols) * thumb_h))
    ensure_dir(out_path.parent)
    grid.save(out_path)


def make_semu_sweep_preview(cfg: dict[str, Any], thresholds: dict[str, Any], out_dir: Path, target_size: tuple[int, int]) -> None:
    ensure_dir(out_dir)
    w, h = min(320, target_size[0]), max(120, int(min(320, target_size[0]) * target_size[1] / target_size[0]))
    classes = ["spot", "line", "region", "gradient", "band"]
    semus = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    tiles = []
    rng = np.random.default_rng(123)
    for cls_name in classes:
        subtype = cfg["subtypes"][cls_name][0]
        if cls_name == "line":
            subtype = "random_streak"
        if cls_name == "band":
            subtype = "row_banding"
        extent = "global" if cls_name in {"gradient", "band"} else "local"
        for semu in semus:
            bg = np.full((h, w), 0.55, dtype=np.float32) + rng.normal(0, 0.006, (h, w)).astype(np.float32)
            shape, _mask, _meta = make_shape(cls_name, subtype, extent, 1, 1, h, w, rng)
            alpha = alpha_from_semu(cfg, cls_name, semu, subtype)
            img = compose(bg, shape, "dark", alpha)
            arr = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            tile = Image.fromarray(arr, mode="L").convert("RGB")
            draw = ImageDraw.Draw(tile)
            draw.rectangle([0, 0, tile.width, 32], fill=(20, 20, 20))
            draw.text((8, 7), f"{cls_name} SEMU {semu:g}", fill=(255, 255, 255), font=font(16))
            tiles.append(tile)
    cols = len(semus)
    grid = Image.new("RGB", (cols * w, len(classes) * h), (245, 245, 245))
    for i, tile in enumerate(tiles):
        grid.paste(tile, ((i % cols) * w, (i // cols) * h))
    grid.save(out_dir / "semu_sweep_by_mask_type.png")


def class_distribution(rows: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    out = {
        "mask_type": Counter(),
        "extent": Counter(),
        "subtype": Counter(),
        "polarity": Counter(),
        "process_step": Counter(),
        "tool_id": Counter(),
        "fault_mode": Counter(),
        "is_reject": Counter(),
    }
    seen = set()
    for row in rows:
        for d in row["defects"]:
            key = (row["glass_id"], int(d["instance_id"]))
            if key in seen:
                continue
            seen.add(key)
            for k in out:
                out[k][str(d.get(k))] += 1
    return out


def write_counter_csv(path: Path, counter: Counter[str], key_name: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([key_name, "count"])
        for k, v in sorted(counter.items()):
            writer.writerow([k, v])


def csv_scalar(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=True)
    if value is None:
        return ""
    return value


def int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_bbox(value: Any) -> list[int] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            x0, y0, x1, y1 = [int(round(float(v))) for v in value]
        except (TypeError, ValueError):
            return None
        if x1 > x0 and y1 > y0:
            return [x0, y0, x1, y1]
    return None


def panel_glass_sizes(row: dict[str, Any]) -> dict[str, int]:
    panel_size = row.get("panel_size") or [0, 0]
    pw = int_or_zero(panel_size[0] if isinstance(panel_size, list) and len(panel_size) >= 1 else 0)
    ph = int_or_zero(panel_size[1] if isinstance(panel_size, list) and len(panel_size) >= 2 else 0)
    glass_cols = int_or_zero(row.get("glass_cols"))
    glass_rows = int_or_zero(row.get("glass_rows"))
    return {
        "panel_width_px": pw,
        "panel_height_px": ph,
        "panel_area_px": pw * ph,
        "glass_width_px": pw * glass_cols,
        "glass_height_px": ph * glass_rows,
        "glass_area_px": pw * ph * glass_cols * glass_rows,
    }


def glass_instance_size_summaries(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    acc: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        sizes = panel_glass_sizes(row)
        pw, ph = sizes["panel_width_px"], sizes["panel_height_px"]
        glass_w, glass_h, glass_area = sizes["glass_width_px"], sizes["glass_height_px"], sizes["glass_area_px"]
        panel_row = int_or_zero(row.get("panel_row"))
        panel_col = int_or_zero(row.get("panel_col"))
        x_offset = panel_col * pw
        y_offset = panel_row * ph
        panel_id = str(row.get("panel_id", ""))
        glass_id = str(row.get("glass_id", ""))
        for defect in row.get("defects", []):
            instance_id = int_or_zero(defect.get("instance_id"))
            key = (glass_id, instance_id)
            item = acc.setdefault(
                key,
                {
                    "glass_width_px": glass_w,
                    "glass_height_px": glass_h,
                    "glass_area_px": glass_area,
                    "glass_target_label_mask_area_px": 0,
                    "glass_labeled_mask_area_px": 0,
                    "glass_local_visible_mask_area_px": 0,
                    "glass_hard_mask_area_px": 0,
                    "glass_bbox_x0": None,
                    "glass_bbox_y0": None,
                    "glass_bbox_x1": None,
                    "glass_bbox_y1": None,
                    "_panel_ids": set(),
                },
            )
            item["_panel_ids"].add(panel_id)
            item["glass_target_label_mask_area_px"] += int_or_zero(defect.get("target_label_mask_area_px"))
            item["glass_labeled_mask_area_px"] += int_or_zero(defect.get("labeled_mask_area_px"))
            item["glass_local_visible_mask_area_px"] += int_or_zero(defect.get("local_visible_mask_area_px"))
            item["glass_hard_mask_area_px"] += int_or_zero(defect.get("panel_hard_mask_area_px"))
            bbox = parse_bbox(defect.get("bbox"))
            if bbox:
                x0, y0, x1, y1 = bbox
                gx0, gy0, gx1, gy1 = x0 + x_offset, y0 + y_offset, x1 + x_offset, y1 + y_offset
                item["glass_bbox_x0"] = gx0 if item["glass_bbox_x0"] is None else min(item["glass_bbox_x0"], gx0)
                item["glass_bbox_y0"] = gy0 if item["glass_bbox_y0"] is None else min(item["glass_bbox_y0"], gy0)
                item["glass_bbox_x1"] = gx1 if item["glass_bbox_x1"] is None else max(item["glass_bbox_x1"], gx1)
                item["glass_bbox_y1"] = gy1 if item["glass_bbox_y1"] is None else max(item["glass_bbox_y1"], gy1)

    out: dict[tuple[str, int], dict[str, Any]] = {}
    for key, item in acc.items():
        glass_w = int_or_zero(item.get("glass_width_px"))
        glass_h = int_or_zero(item.get("glass_height_px"))
        glass_area = int_or_zero(item.get("glass_area_px"))
        x0, y0, x1, y1 = item.get("glass_bbox_x0"), item.get("glass_bbox_y0"), item.get("glass_bbox_x1"), item.get("glass_bbox_y1")
        bbox = ""
        bbox_norm = ""
        bbox_rect_area = 0
        if None not in (x0, y0, x1, y1):
            bbox = [int(x0), int(y0), int(x1), int(y1)]
            bbox_rect_area = max(0, int(x1) - int(x0)) * max(0, int(y1) - int(y0))
            if glass_w > 0 and glass_h > 0:
                bbox_norm = [
                    round(float(x0) / glass_w, 6),
                    round(float(y0) / glass_h, 6),
                    round(float(x1) / glass_w, 6),
                    round(float(y1) / glass_h, 6),
                ]
        labeled_area = int_or_zero(item.get("glass_labeled_mask_area_px"))
        target_label_area = int_or_zero(item.get("glass_target_label_mask_area_px"))
        visible_area = int_or_zero(item.get("glass_local_visible_mask_area_px"))
        hard_area = int_or_zero(item.get("glass_hard_mask_area_px"))
        out[key] = {
            "glass_width_px": glass_w,
            "glass_height_px": glass_h,
            "glass_area_px": glass_area,
            "glass_panels_touched": len(item.get("_panel_ids", set())),
            "glass_bbox": bbox,
            "glass_bbox_norm": bbox_norm,
            "glass_bbox_rect_area_px": bbox_rect_area,
            "glass_target_label_mask_area_px": target_label_area,
            "glass_target_label_mask_area_fraction": target_label_area / glass_area if glass_area > 0 else 0.0,
            "glass_labeled_mask_area_px": labeled_area,
            "glass_labeled_mask_area_fraction": labeled_area / glass_area if glass_area > 0 else 0.0,
            "glass_local_visible_mask_area_px": visible_area,
            "glass_local_visible_mask_area_fraction": visible_area / glass_area if glass_area > 0 else 0.0,
            "glass_hard_mask_area_px": hard_area,
            "glass_hard_mask_area_fraction": hard_area / glass_area if glass_area > 0 else 0.0,
        }
    return out


def write_instance_record_csvs(reports: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "glass_id",
        "panel_id",
        "panel_index",
        "panel_row",
        "panel_col",
        "split",
        "source_id",
        "panel_width_px",
        "panel_height_px",
        "panel_area_px",
        "glass_width_px",
        "glass_height_px",
        "glass_area_px",
        "instance_id",
        "instance_key",
        "mask_type",
        "class_id",
        "extent",
        "extent_id",
        "subtype",
        "polarity",
        "semu",
        "panel_label_semu",
        "panel_instance_semu",
        "panel_semu_peak",
        "panel_semu_p95_hard_mask",
        "panel_semu_p95_visible_core",
        "sampled_instance_semu",
        "global_instance_semu",
        "effective_instance_semu",
        "alpha",
        "engineer_visible_threshold",
        "reject_visible_threshold",
        "is_visible",
        "is_reject",
        "instance_is_visible",
        "instance_is_reject",
        "panel_is_locally_visible",
        "panel_is_locally_reject",
        "segmentation_train_enabled",
        "train_mask_ignore_label",
        "label_mask_policy",
        "label_mask_tail_fraction",
        "label_mask_max_expansion_ratio",
        "label_mask_tail_radius_px",
        "label_mask_semu_threshold",
        "panel_hard_mask_area_px",
        "panel_hard_mask_area_fraction",
        "local_visible_mask_area_px",
        "local_visible_mask_area_fraction",
        "target_label_mask_area_px",
        "target_label_mask_area_fraction",
        "train_label_mask_area_px",
        "train_ignore_mask_area_px",
        "labeled_mask_area_px",
        "labeled_mask_area_fraction",
        "bbox",
        "bbox_norm",
        "yolo",
        "bbox_area_px",
        "bbox_rect_area_px",
        "glass_panels_touched",
        "glass_bbox",
        "glass_bbox_norm",
        "glass_bbox_rect_area_px",
        "glass_target_label_mask_area_px",
        "glass_target_label_mask_area_fraction",
        "glass_labeled_mask_area_px",
        "glass_labeled_mask_area_fraction",
        "glass_local_visible_mask_area_px",
        "glass_local_visible_mask_area_fraction",
        "glass_hard_mask_area_px",
        "glass_hard_mask_area_fraction",
        "process_step",
        "tool_id",
        "root_cause",
        "fault_mode",
        "time_window",
        "time_window_frac",
        "target_cycle",
        "primary_cycle",
        "primary_process_step",
        "primary_step_index",
        "primary_route_node_pos",
        "route_node_positions",
        "t_seq",
        "t_norm",
        "spatial_signature",
        "centroid_norm",
        "extent_box_px",
    ]

    panel_path = reports / "panel_instance_records.csv"
    unique_path = reports / "unique_instance_records.csv"
    glass_sizes = glass_instance_size_summaries(rows)
    seen_unique: set[tuple[str, int]] = set()
    with panel_path.open("w", newline="", encoding="utf-8") as fp, unique_path.open("w", newline="", encoding="utf-8") as fu:
        panel_writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        unique_writer = csv.DictWriter(fu, fieldnames=fields, extrasaction="ignore")
        panel_writer.writeheader()
        unique_writer.writeheader()
        for row in rows:
            base = {
                "glass_id": row.get("glass_id"),
                "panel_id": row.get("panel_id"),
                "panel_index": row.get("panel_index"),
                "panel_row": row.get("panel_row"),
                "panel_col": row.get("panel_col"),
                "split": row.get("split"),
                "source_id": row.get("source_id"),
            }
            base.update(panel_glass_sizes(row))
            for defect in row.get("defects", []):
                instance_id = int(defect.get("instance_id", -1))
                flat = {**base, **defect}
                flat["instance_id"] = instance_id
                flat["instance_key"] = f"{row.get('glass_id')}:{instance_id}"
                unique_key = (str(row.get("glass_id")), instance_id)
                flat.update(glass_sizes.get(unique_key, {}))
                flat = {key: csv_scalar(flat.get(key, "")) for key in fields}
                panel_writer.writerow(flat)
                if unique_key not in seen_unique:
                    seen_unique.add(unique_key)
                    unique_writer.writerow(flat)


def write_reports(root: Path, rows: list[dict[str, Any]]) -> None:
    reports = root / "reports"
    ensure_dir(reports)
    dist = class_distribution(rows)
    write_counter_csv(reports / "class_distribution.csv", dist["mask_type"], "mask_type")
    write_instance_record_csvs(reports, rows)
    with open(reports / "attribute_distribution.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["attribute", "value", "count"])
        for key in ["extent", "subtype", "polarity", "is_reject"]:
            for value, count in sorted(dist[key].items()):
                writer.writerow([key, value, count])
    with open(reports / "rca_distribution.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "value", "count"])
        for key in ["process_step", "tool_id", "fault_mode"]:
            for value, count in sorted(dist[key].items()):
                writer.writerow([key, value, count])
    with open(reports / "time_window_distribution.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fault_mode", "time_window", "count"])
        c = Counter()
        seen = set()
        for row in rows:
            for d in row["defects"]:
                key = (row["glass_id"], int(d["instance_id"]))
                if key in seen:
                    continue
                seen.add(key)
                c[(str(d["fault_mode"]), str(d["time_window"]))] += 1
        for (mode, window), count in sorted(c.items()):
            writer.writerow([mode, window, count])


def write_dataset_card(root: Path, cfg: dict[str, Any], stats: dict[str, Any]) -> None:
    text = f"""# MuraTrace

This dataset is a real-background hybrid benchmark for display Mura perception and process root-cause analysis (RCA).

## Visual Classes

- spot
- line
- region
- gradient
- band

Scale, subtype, polarity, SEMU, reject, process_step, tool_id, fault_mode,
time_window, and route_graph are stored as annotations, not as semantic mask classes.

## Numbering

- `glass_id`: `G000000`, `G000001`, ...
- `panel_index`: row-major index, `row * cols + col`
- `panel_id`: `G000000_P00_00`, `G000000_P00_01`, ...

## Generation Summary

- Real source backgrounds: {stats["clean_count"]}
- Augmented background variants: {stats["augmented_count"]}
- Glasses: {stats["glass_count"]}
- Panel rows x cols: {cfg["geometry"]["rows"]} x {cfg["geometry"]["cols"]}
- Saved panels: {stats["panel_count"]}
- Source-level split: train/val/test by source background id

## SEMU Thresholds

Edit `configs/muratrace_thresholds.json`.

- `engineer_visible`: SEMU threshold used to decide whether a panel-local defect instance is engineer-visible. 
                    The actual support mask policy is stored as `generation.label_mask_policy`; 
                    the final real-background profile keeps generated hard-mask support on every panel it touches. 
                    The separate `train_mask` stores sub-visible support as ignore label 255, 
                    so those pixels remain available for audit/SEMU regression but are off for segmentation loss.
                    The panel-local SEMU-thresholded core remains in `local_visible_mask_area_*` and 
                    `panel_instance_semu` as a diagnostic core statistic.
- `reject_visible`: a panel-instance gets `is_reject=true` when its panel-local regression severity `semu` 
                    reaches this threshold. Instance-level reject status is also retained as `instance_is_reject`.

Panel annotations store `semu` as the panel-local regression target:
`panel_semu_p95_hard_mask = p95(local SEMU over the generated hard-mask support)`.
This keeps sub-visible tail panels continuous instead of forcing them to zero.
`panel_is_locally_visible` and `is_reject` are then derived by comparing this panel-local `semu` with the subtype thresholds. 
The SEMU-thresholded visible core is retained separately as `panel_instance_semu` and `local_visible_mask_area_*` for diagnostics. 
The originally sampled glass-instance severity is retained as `sampled_instance_semu` and `global_instance_semu`.
 
Subtype-level thresholds override mask-type thresholds. 
This is recommended because spot, line, smooth gradient, and banding have different perceptual visibility and reject criteria.
"""
    with open(root / "dataset_card.md", "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/muratrace_config.json")
    ap.add_argument("--thresholds", default="configs/muratrace_thresholds.json")
    ap.add_argument("--bg_dir", default=None)
    ap.add_argument("--out", default="data/MuraTrace")
    ap.add_argument("--per_source", type=int, default=None)
    ap.add_argument("--target_size", type=parse_size, default=None, help="optional WIDTHxHEIGHT. Overrides resize_factor.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--num_glasses", type=int, default=0, help="optional cap after augmentation selection")
    ap.add_argument("--save_all_panels", action="store_true", help="override config and save all panels")
    ap.add_argument("--defective_panels_only", action="store_true", help="save only panels with visible labels")
    a = ap.parse_args()

    cfg = read_json(Path(a.config))
    thresholds = read_json(Path(a.thresholds))
    out_root = Path(a.out)
    dirs = prepare_output(out_root, overwrite=a.overwrite)

    bg_dir = Path(a.bg_dir or cfg["background"]["input_dir"])
    per_source = int(a.per_source if a.per_source is not None else cfg["background"]["per_source"])
    crop_edges = background_crop_edges(cfg)
    target_size = choose_target_size(bg_dir, cfg, a.target_size)
    rows, cols = int(cfg["geometry"]["rows"]), int(cfg["geometry"]["cols"])
    pw, ph = target_size
    bit_depth = int(cfg["geometry"].get("bit_depth", 12))
    label_mask_policy = str(cfg.get("generation", {}).get("label_mask_policy", "local_visible_semu"))

    bg_result = generate_background_bank(
        input_dir=bg_dir,
        out_dir=out_root,
        per_source=per_source,
        seed=a.seed,
        target_size=target_size,
        preview_per_source=3,
        crop_edges=crop_edges,
    )
    source_ids = sorted({r["source_id"] for r in bg_result["augmented"]})
    source_splits = assign_source_splits(source_ids, cfg["split"]["ratios"], seed=a.seed + 17)
    all_records = list(bg_result["augmented"])
    record_rng = np.random.default_rng(a.seed + 1009)
    record_rng.shuffle(all_records)
    if a.num_glasses and a.num_glasses > 0:
        all_records = all_records[: int(a.num_glasses)]
    records_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in bg_result["augmented"]:
        records_by_source[rec["source_id"]].append(rec)
        records_by_split[source_splits[rec["source_id"]]].append(rec)
    for sid in records_by_source:
        records_by_source[sid] = sorted(records_by_source[sid], key=lambda r: int(r["aug_index"]))
    for split_name in records_by_split:
        records_by_split[split_name] = sorted(records_by_split[split_name], key=lambda r: (str(r["source_id"]), int(r["aug_index"])))

    rng = np.random.default_rng(a.seed)
    class_id = {k: int(v) for k, v in cfg["classes"].items() if k != "background"}
    extent_id = {k: int(v) for k, v in cfg["extent_classes"].items()}
    maxv = (2**bit_depth) - 1
    n_total = len(all_records)
    save_all = bool(cfg["generation"].get("save_all_panels", True))
    if a.save_all_panels:
        save_all = True
    if a.defective_panels_only:
        save_all = False
    glass_image_limit = int(cfg["generation"].get("glass_image_limit", 0))
    use_quota_mode = subtype_quota_enabled(cfg)
    quota_state: dict[str, Counter[str]] = {"unique": Counter(), "reject": Counter()}
    quota_unique_targets, quota_reject_targets = quota_target_maps(cfg)

    manifest_rows: list[dict[str, Any]] = []
    glass_preview_paths: list[Path] = []
    class_preview_first: dict[str, Path] = {}

    manifest_path = out_root / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for gidx, bg_row in enumerate(all_records):
            glass_id = f"G{gidx:06d}"
            split = source_splits[bg_row["source_id"]]
            t_norm = float(gidx / max(n_total - 1, 1))
            active = active_faults(cfg, t_norm, n_total)
            if not active:
                active = list(cfg["fault_library"].get("persistent", []))

            n_def = sample_quota_defect_count(cfg, quota_state, rng) if use_quota_mode else sample_defect_count(cfg, rng)
            defects: list[dict[str, Any]] = []
            defect_maps = []
            selected_faults: list[dict[str, Any]] = []
            inst_id = 1
            while inst_id <= n_def:
                if use_quota_mode:
                    fault = sample_fault_for_quota(active, cfg, selected_faults, quota_state, rng, gidx, n_total)
                    if fault is None:
                        break
                else:
                    fault = sample_fault_for_defect(active, cfg, selected_faults, rng)
                selected_faults.append(dict(fault))

                mask_type = str(fault["mask_type"])
                subtype = str(fault["subtype"])
                extent = str(fault["extent"])
                tool_id = fault.get("tool_id")
                signature = cfg.get("tool_spatial_signatures", {}).get(str(tool_id), {}) if tool_id else {}
                thr = visibility_threshold(thresholds, mask_type, subtype)
                semu = semu_sample_for_quota(thr, subtype, quota_state, cfg, rng) if use_quota_mode else semu_sample(thr, rng)
                polarity = "dark" if rng.random() < float(cfg["generation"]["polarity_probability"]["dark"]) else "bright"
                shape_map, hard_mask, shape_meta = make_shape(
                    mask_type,
                    subtype,
                    extent,
                    rows,
                    cols,
                    ph,
                    pw,
                    rng,
                    signature=signature,
                )
                alpha = alpha_from_semu(cfg, mask_type, semu, subtype)
                per_semu = alpha_per_semu(cfg, mask_type, subtype)
                effective_instance_semu = float(alpha / per_semu)
                instance_local_semu = shape_map * effective_instance_semu
                instance_stats = local_semu_stats(
                    instance_local_semu,
                    hard_mask,
                    float(thr["engineer_visible"]),
                    float(thr["reject_visible"]),
                )
                is_visible = bool(instance_stats["is_visible"])
                is_reject = bool(instance_stats["is_reject"])
                root_cause = [] if not fault.get("tool_id") else [fault["tool_id"]]
                process_step = fault.get("process_step")
                d = {
                    "instance_id": inst_id,
                    "mask_type": mask_type,
                    "class_id": class_id[mask_type],
                    "extent": extent,
                    "extent_id": extent_id[extent],
                    "subtype": subtype,
                    "polarity": polarity,
                    "semu": semu,
                    "sampled_instance_semu": semu,
                    "effective_instance_semu": effective_instance_semu,
                    "instance_semu_peak": instance_stats["panel_semu_peak"],
                    "instance_semu_p95_hard_mask": instance_stats["panel_semu_p95_hard_mask"],
                    "instance_semu_p95_visible_mask": instance_stats["panel_instance_semu"],
                    "instance_visible_mask_area_fraction": instance_stats["visible_mask_area_fraction"],
                    "engineer_visible_threshold": thr["engineer_visible"],
                    "reject_visible_threshold": thr["reject_visible"],
                    "is_visible": is_visible,
                    "is_reject": is_reject,
                    "alpha": alpha,
                    "process_step": process_step,
                    "tool_id": tool_id,
                    "root_cause": root_cause,
                    "fault_mode": fault.get("fault_mode"),
                    "time_window": fault.get("time_window"),
                    "time_window_frac": fault.get("time_window_frac"),
                    "t_seq": gidx,
                    "t_norm": round(t_norm, 6),
                    "spatial_signature": signature.get("description"),
                    **shape_meta,
                }
                defects.append(d)
                defect_maps.append((shape_map, hard_mask, d))
                if use_quota_mode and is_visible:
                    quota_state["unique"][subtype] += 1
                    if is_reject:
                        quota_state["reject"][subtype] += 1
                inst_id += 1

            defect_maps.sort(key=defect_zorder_key)
            required_tool_cycles: list[tuple[str, int | None]] = []
            occupied_cycle_slots: dict[tuple[int, int], str] = {}
            for d in defects:
                if d.get("tool_id"):
                    tool_id_str = str(d["tool_id"])
                    preferred_cycle = preferred_cycle_for_defect(d, rng, n_cycles=3)
                    target_cycle = preferred_cycle
                    step_i = tool_step_index(tool_id_str)
                    if step_i is not None:
                        cycle_order = [preferred_cycle] + [c for c in range(3) if c != preferred_cycle]
                        for cycle in cycle_order:
                            slot_key = (int(cycle), int(step_i))
                            if slot_key not in occupied_cycle_slots or occupied_cycle_slots[slot_key] == tool_id_str:
                                target_cycle = int(cycle)
                                occupied_cycle_slots[slot_key] = tool_id_str
                                break
                    d["target_cycle"] = int(target_cycle)
                    required_tool_cycles.append((tool_id_str, int(d["target_cycle"])))
                else:
                    d["target_cycle"] = None
            required_tools = [str(d["tool_id"]) for d in defects if d.get("tool_id")]
            route_graph = make_route_graph(cfg, rng, required_tool_ids=required_tools, required_tool_cycles=required_tool_cycles)
            route_positions_by_tool: dict[str, list[int]] = defaultdict(list)
            for node in route_graph["nodes"]:
                route_positions_by_tool[str(node["tool_id"])].append(int(node["node_pos"]))
            for d in defects:
                d["route_node_positions"] = route_positions_by_tool.get(str(d.get("tool_id")), [])
                primary_node = choose_primary_route_node(route_graph, d.get("tool_id"), rng, d.get("target_cycle"))
                d["primary_route_node_pos"] = int(primary_node["node_pos"]) if primary_node else None
                d["primary_cycle"] = int(primary_node["cycle"]) if primary_node else None
                d["primary_process_step"] = primary_node["step"] if primary_node else d.get("process_step")
                d["primary_step_index"] = int(primary_node["step_index"]) if primary_node else None

            split_records = records_by_split.get(split) or list(bg_result["augmented"])
            panel_bg_rng = np.random.default_rng(a.seed + 5003 + gidx * 97)
            panel_count = rows * cols
            if len(split_records) >= panel_count:
                panel_bg_indices = panel_bg_rng.choice(len(split_records), size=panel_count, replace=False)
            else:
                panel_bg_indices = panel_bg_rng.choice(len(split_records), size=panel_count, replace=True)
            ref_bg = load_luma(Path(bg_row["aug_path"]), target_size=(pw, ph))
            ref_mean = float(ref_bg.mean())
            ref_std = float(ref_bg.std())
            panel_float = []
            panel_mask = []
            panel_train_mask = []
            panel_inst = []
            panel_defects = []
            panel_bg_records = []
            for r in range(rows):
                for c in range(cols):
                    panel_idx = r * cols + c
                    rec = split_records[int(panel_bg_indices[panel_idx])]
                    bg = load_luma(Path(rec["aug_path"]), target_size=(pw, ph))
                    bg = match_panel_luminance(bg, ref_mean, ref_std)
                    bg = prepare_synthesis_background(bg, cfg, panel_bg_rng)
                    mask = np.zeros((ph, pw), dtype=np.uint8)
                    train_mask = np.zeros((ph, pw), dtype=np.uint8)
                    inst = np.zeros((ph, pw), dtype=np.uint16)
                    img = bg.astype(np.float32)
                    local_defect_candidates: list[dict[str, Any]] = []
                    for shape_map, hard_mask, d in defect_maps:
                        sm = crop_panel(shape_map, r, c, ph, pw)
                        hm = crop_panel(hard_mask, r, c, ph, pw)
                        if not np.any(hm):
                            continue
                        img = compose(img, sm, d["polarity"], float(d["alpha"]))
                        local_semu = sm * float(d["effective_instance_semu"])
                        stats = local_semu_stats(
                            local_semu,
                            hm,
                            float(d["engineer_visible_threshold"]),
                            float(d["reject_visible_threshold"]),
                        )
                        visible_mask = stats["visible_mask"]
                        store_support_mask = bool(cfg.get("generation", {}).get("store_subvisible_support_masks", True))
                        if bool(d.get("is_visible", False)) or store_support_mask:
                            visible_threshold = float(d["engineer_visible_threshold"])
                            reject_threshold = float(d["reject_visible_threshold"])
                            panel_label_semu = panel_label_semu_from_stats(stats)
                            panel_is_locally_visible = bool(panel_label_semu >= visible_threshold)
                            panel_is_locally_reject = bool(panel_label_semu >= reject_threshold)
                            uses_tail_policy = label_mask_policy in {
                                "semu_tail",
                                "semu_tail_visible_instance",
                                "local_semu_tail",
                                "core_tail",
                                "core_tail_visible_instance",
                                "visible_core_tail",
                            }
                            tail_fraction = label_tail_fraction(cfg, d) if uses_tail_policy else None
                            max_expansion_ratio = label_max_expansion_ratio(cfg, d) if uses_tail_policy else None
                            tail_radius_px = label_tail_radius_px(cfg, d) if uses_tail_policy else None
                            label_semu_threshold = visible_threshold * tail_fraction if tail_fraction is not None else None
                            label_mask = label_mask_from_policy(
                                label_mask_policy,
                                hm,
                                visible_mask,
                                local_semu=local_semu,
                                visible_threshold=visible_threshold,
                                tail_fraction=tail_fraction if tail_fraction is not None else 0.70,
                                max_expansion_ratio=max_expansion_ratio,
                                tail_radius_px=tail_radius_px or 0,
                            )
                            if not np.any(label_mask):
                                continue
                            local_d = dict(d)
                            local_d.update(
                                {
                                    "semu": panel_label_semu,
                                    "panel_label_semu": panel_label_semu,
                                    "global_instance_semu": float(d["sampled_instance_semu"]),
                                    "label_mask_policy": label_mask_policy,
                                    "label_mask_tail_fraction": tail_fraction,
                                    "label_mask_max_expansion_ratio": max_expansion_ratio,
                                    "label_mask_tail_radius_px": tail_radius_px,
                                    "label_mask_semu_threshold": label_semu_threshold,
                                    "panel_semu_peak": stats["panel_semu_peak"],
                                    "panel_semu_p95_hard_mask": stats["panel_semu_p95_hard_mask"],
                                    "panel_semu_p95_visible_core": stats["panel_semu_p95_visible_core"],
                                    "panel_instance_semu": stats["panel_instance_semu"],
                                    "panel_hard_mask_area_px": stats["hard_mask_area_px"],
                                    "panel_hard_mask_area_fraction": stats["hard_mask_area_fraction"],
                                    "local_visible_mask_area_px": stats["visible_mask_area_px"],
                                    "local_visible_mask_area_fraction": stats["visible_mask_area_fraction"],
                                    "target_label_mask_area_px": int(np.count_nonzero(label_mask)),
                                    "target_label_mask_area_fraction": float(np.count_nonzero(label_mask) / label_mask.size),
                                    "is_visible": bool(d["is_visible"]),
                                    "is_reject": panel_is_locally_reject,
                                    "instance_is_visible": bool(d["is_visible"]),
                                    "instance_is_reject": bool(d["is_reject"]),
                                    "panel_is_locally_visible": panel_is_locally_visible,
                                    "panel_is_locally_reject": panel_is_locally_reject,
                                    "segmentation_train_enabled": panel_is_locally_visible,
                                    "train_mask_ignore_label": TRAIN_IGNORE_LABEL,
                                    "train_label_mask_area_px": int(np.count_nonzero(label_mask)) if panel_is_locally_visible else 0,
                                    "train_ignore_mask_area_px": 0 if panel_is_locally_visible else int(np.count_nonzero(label_mask)),
                                }
                            )
                            local_defect_candidates.append(local_d)
                            mask[label_mask] = int(d["class_id"])
                            inst[label_mask] = int(d["instance_id"])
                            if panel_is_locally_visible:
                                train_mask[label_mask] = int(d["class_id"])
                            else:
                                train_mask[label_mask] = TRAIN_IGNORE_LABEL
                    panel_float.append(img)
                    panel_mask.append(mask)
                    panel_train_mask.append(train_mask)
                    panel_inst.append(inst)
                    panel_defects.append(local_defect_candidates)
                    panel_bg_records.append(rec)

            route_path = dirs["glass_route_graphs"] / f"{glass_id}.json"
            with open(route_path, "w", encoding="utf-8") as f:
                json.dump(route_graph, f, indent=2, ensure_ascii=True)

            if glass_image_limit < 0 or gidx < glass_image_limit:
                glass_img = np.zeros((rows * ph, cols * pw), dtype=np.float32)
                glass_mask = np.zeros((rows * ph, cols * pw), dtype=np.uint8)
                for r in range(rows):
                    for c in range(cols):
                        idx = r * cols + c
                        glass_img[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw] = panel_float[idx]
                        glass_mask[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw] = panel_mask[idx]
                glass_png = dirs["glass_images"] / f"{glass_id}.png"
                save_u16_png(glass_img, glass_png, bit_depth)
                overlay = overlay_mask_u16(np.clip(glass_img * maxv + 0.5, 0, maxv).astype(np.uint16), glass_mask)
                overlay_path = dirs["glass_overlays"] / f"{glass_id}_mask_overlay.png"
                overlay.save(overlay_path)
                if gidx < 20:
                    prev_path = dirs["preview_glass"] / f"{glass_id}_overlay.png"
                    overlay.resize((min(1200, overlay.width), int(overlay.height * min(1200, overlay.width) / overlay.width))).save(prev_path)
                    glass_preview_paths.append(prev_path)

            for r in range(rows):
                for c in range(cols):
                    panel_idx = r * cols + c
                    panel_id = f"{glass_id}_P{r:02d}_{c:02d}"
                    img = panel_float[panel_idx]
                    mask = panel_mask[panel_idx]
                    train_mask = panel_train_mask[panel_idx]
                    inst = panel_inst[panel_idx]
                    rec = panel_bg_records[panel_idx]
                    visible_ids = set(int(x) for x in np.unique(inst) if int(x) > 0)
                    local_defects = []
                    for d in panel_defects[panel_idx]:
                        instance_id = int(d["instance_id"])
                        if instance_id not in visible_ids:
                            continue
                        labeled = inst == instance_id
                        if not np.any(labeled):
                            continue
                        local_d = dict(d)
                        local_d["labeled_mask_area_px"] = int(np.count_nonzero(labeled))
                        local_d["labeled_mask_area_fraction"] = float(np.count_nonzero(labeled) / labeled.size)
                        local_defects.append(local_d)
                    if not save_all and not local_defects:
                        continue

                    img_path = dirs["panel_images"] / f"{panel_id}_img.npy"
                    mask_path = dirs["panel_masks"] / f"{panel_id}_mask.npy"
                    train_mask_path = dirs["panel_train_masks"] / f"{panel_id}_train_mask.npy"
                    inst_path = dirs["panel_inst"] / f"{panel_id}_inst.npy"
                    save_u16_npy(img, img_path, bit_depth)
                    np.save(mask_path, mask.astype(np.uint8))
                    np.save(train_mask_path, train_mask.astype(np.uint8))
                    np.save(inst_path, inst.astype(np.uint16))
                    boxes = bboxes_from_inst(mask, inst, local_defects, pw, ph)
                    train_boxes = [b for b in boxes if bool(b.get("panel_is_locally_visible", False))]
                    box_by_instance = {int(b["instance_id"]): b for b in boxes}
                    for d in local_defects:
                        b = box_by_instance.get(int(d["instance_id"]))
                        if b:
                            d["bbox"] = b.get("bbox")
                            d["bbox_norm"] = b.get("bbox_norm")
                            d["yolo"] = b.get("yolo")
                            d["bbox_area_px"] = b.get("area_px")
                            d["bbox_rect_area_px"] = b.get("bbox_rect_area_px")
                    box_path = dirs["panel_boxes"] / f"{panel_id}_boxes.json"
                    with open(box_path, "w", encoding="utf-8") as f:
                        json.dump({"panel_id": panel_id, "panel_size": [pw, ph], "boxes": boxes}, f, indent=2, ensure_ascii=True)
                    train_box_path = dirs["panel_train_boxes"] / f"{panel_id}_train_boxes.json"
                    with open(train_box_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "panel_id": panel_id,
                                "panel_size": [pw, ph],
                                "boxes": train_boxes,
                                "source_boxes": str(box_path.relative_to(out_root)),
                                "filter": "panel_is_locally_visible == true",
                            },
                            f,
                            indent=2,
                            ensure_ascii=True,
                        )
                    yolo_path = dirs["panel_yolo"] / f"{panel_id}.txt"
                    write_yolo(yolo_path, boxes)
                    train_yolo_path = dirs["panel_train_yolo"] / f"{panel_id}.txt"
                    write_yolo(train_yolo_path, train_boxes)

                    for d in local_defects:
                        if d["mask_type"] not in class_preview_first:
                            prev = overlay_mask_u16(np.clip(img * maxv + 0.5, 0, maxv).astype(np.uint16), mask)
                            p = dirs["preview_class"] / f"{d['mask_type']}_{d['subtype']}.png"
                            prev.save(p)
                            class_preview_first[d["mask_type"]] = p

                    row = {
                        "panel_id": panel_id,
                        "glass_id": glass_id,
                        "panel_index": panel_idx,
                        "panel_row": r,
                        "panel_col": c,
                        "image": str(img_path.relative_to(out_root)),
                        "mask": str(mask_path.relative_to(out_root)),
                        "train_mask": str(train_mask_path.relative_to(out_root)),
                        "inst": str(inst_path.relative_to(out_root)),
                        "boxes": str(box_path.relative_to(out_root)),
                        "train_boxes": str(train_box_path.relative_to(out_root)),
                        "yolo": str(yolo_path.relative_to(out_root)),
                        "train_yolo": str(train_yolo_path.relative_to(out_root)),
                        "split": split,
                        "source_id": rec["source_id"],
                        "source_file": rec["source_file"],
                        "panel_background_aug_id": rec["aug_id"],
                        "glass_seed_source_id": bg_row["source_id"],
                        "glass_seed_background_aug_id": bg_row["aug_id"],
                        "route_graph": str(route_path.relative_to(out_root)),
                        "t_seq": gidx,
                        "t_norm": round(t_norm, 6),
                        "panel_has_defect": bool(local_defects),
                        "panel_has_train_defect": bool(np.any((train_mask > 0) & (train_mask < TRAIN_IGNORE_LABEL))),
                        "panel_has_ignored_defect": bool(np.any(train_mask == TRAIN_IGNORE_LABEL)),
                        "defects": local_defects,
                        "boxes_count": len(boxes),
                        "train_boxes_count": len(train_boxes),
                        "glass_rows": rows,
                        "glass_cols": cols,
                        "panel_size": [pw, ph],
                    }
                    mf.write(json.dumps(row, ensure_ascii=True) + "\n")
                    manifest_rows.append(row)

    write_reports(out_root, manifest_rows)
    make_preview_grid(glass_preview_paths, dirs["previews"] / "glass_examples_grid.png", cols=2, thumb_w=700)
    make_semu_sweep_preview(cfg, thresholds, dirs["preview_semu"], target_size)

    final_cfg = {
        **cfg,
        "runtime": {
            "seed": a.seed,
            "background_input_dir": str(bg_dir),
            "background_crop_edges": crop_edges,
            "output_dir": str(out_root),
            "target_size": [pw, ph],
            "source_splits": source_splits,
            "selected_glasses": n_total,
            "save_all_panels": save_all,
            "subtype_quota_mode_enabled": use_quota_mode,
            "subtype_quota_targets": {
                "unique": quota_unique_targets,
                "reject": quota_reject_targets,
            },
            "subtype_quota_final": {
                "unique": dict(quota_state["unique"]),
                "reject": dict(quota_state["reject"]),
            },
        },
        "threshold_file": str(Path(a.thresholds)),
        "background_bank": {
            "clean_count": len(bg_result["clean"]),
            "augmented_count": len(bg_result["augmented"]),
        },
    }
    with open(out_root / "config.json", "w", encoding="utf-8") as f:
        json.dump(final_cfg, f, indent=2, ensure_ascii=True)
    stats = {
        "clean_count": len(bg_result["clean"]),
        "augmented_count": len(bg_result["augmented"]),
        "glass_count": n_total,
        "panel_count": len(manifest_rows),
    }
    write_dataset_card(out_root, cfg, stats)
    print(f"wrote {n_total} glasses and {len(manifest_rows)} panel rows to {out_root}")
    print(f"target_size={pw}x{ph}, source_splits={source_splits}")


if __name__ == "__main__":
    main()
