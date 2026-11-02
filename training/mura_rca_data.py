from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


MASK_TYPES = ["spot", "line", "region", "gradient", "band"]
SEG_IGNORE_INDEX = 255
EXTENTS = ["local", "regional", "global"]
SUBTYPES = [
    "particle_spot",
    "random_streak",
    "stitch_line",
    "puddle_blob",
    "thermal_blob",
    "anneal_radial_gradient",
    "row_banding",
    "column_banding",
]
POLARITIES = ["dark", "bright"]
PROCESS_STEPS = ["none", "Clean", "Coat", "Deposit", "Litho", "Etch", "Strip", "Anneal"]
FAULT_MODES = ["none", "persistent", "excursion", "background"]


def build_tool_ids() -> list[str]:
    tools = ["none"]
    for step_idx, count in enumerate([2, 3, 3, 2, 3, 2, 2], start=1):
        for tool_idx in range(1, count + 1):
            tools.append(f"S{step_idx}_T{tool_idx}")
    return tools


TOOL_IDS = build_tool_ids()

MASK_TYPE_TO_IDX = {v: i for i, v in enumerate(MASK_TYPES)}
EXTENT_TO_IDX = {v: i for i, v in enumerate(EXTENTS)}
SUBTYPE_TO_IDX = {v: i for i, v in enumerate(SUBTYPES)}
POLARITY_TO_IDX = {v: i for i, v in enumerate(POLARITIES)}
PROCESS_TO_IDX = {v: i for i, v in enumerate(PROCESS_STEPS)}
TOOL_TO_IDX = {v: i for i, v in enumerate(TOOL_IDS)}
FAULT_MODE_TO_IDX = {v: i for i, v in enumerate(FAULT_MODES)}


def stable_hash_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def rel_path(root: Path, value: str) -> Path:
    return root / value.replace("\\", "/")


def row_path_value(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    raise KeyError(f"manifest row is missing all path keys: {keys}")


def load_manifest(root: Path) -> list[dict[str, Any]]:
    manifest = root / "manifest.jsonl"
    rows = []
    with open(manifest, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"manifest JSON error at line {line_no}: {exc}") from exc
    return rows


def assigned_split(row: dict[str, Any], split_mode: str, train_ratio: float = 0.6, val_ratio: float = 0.2) -> str:
    if split_mode == "manifest":
        return str(row["split"])
    if split_mode == "random_panel":
        key = str(row["panel_id"])
    elif split_mode == "random_glass":
        key = str(row["glass_id"])
    else:
        raise ValueError(f"unknown split_mode: {split_mode}")
    u = (stable_hash_int(key) % 1_000_000) / 1_000_000.0
    if u < train_ratio:
        return "train"
    if u < train_ratio + val_ratio:
        return "val"
    return "test"


def normalize_image(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    hi = float(arr.max()) if arr.size else 1.0
    if hi > 255.0:
        arr = arr / 4095.0 if hi <= 4095.0 else arr / 65535.0
    elif hi > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def resize_tensor_image(x: torch.Tensor, size: int | None) -> torch.Tensor:
    if size is None:
        return x
    return torch.nn.functional.interpolate(
        x.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def resize_tensor_mask(mask: torch.Tensor, size: int | None) -> torch.Tensor:
    if size is None:
        return mask
    out = torch.nn.functional.interpolate(
        mask[None, None].float(),
        size=(size, size),
        mode="nearest",
    )[0, 0]
    return out.long()


class PanelSegDataset(Dataset):
    """Panel-level semantic segmentation dataset.

    Images are single-channel panel `.npy` files. Masks use ids:
    0 background, 1 spot, 2 line, 3 region, 4 gradient, 5 band.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int | None = 384,
        split_mode: str = "manifest",
        defective_only: bool = False,
        mask_key: str = "train_mask",
        limit: int | None = None,
        source_ids: "set[str] | list[str] | None" = None,
        subsample: int | None = None,
        subsample_seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.mask_key = str(mask_key)
        rows = load_manifest(self.root)
        keep_sources = {str(s) for s in source_ids} if source_ids else None
        filtered = []
        for row in rows:
            if assigned_split(row, split_mode) != split:
                continue
            if defective_only and not bool(row.get("panel_has_train_defect", row.get("panel_has_defect", False))):
                continue
            # camera-ready background ablation: restrict a split to a subset of real background sources
            if keep_sources is not None and str(row.get("source_id")) not in keep_sources:
                continue
            filtered.append(row)
        # camera-ready background ablation: panel-count-matched control (random subset, fixed seed)
        if subsample and int(subsample) < len(filtered):
            rng = np.random.default_rng(int(subsample_seed))
            idx = sorted(rng.choice(len(filtered), int(subsample), replace=False).tolist())
            filtered = [filtered[i] for i in idx]
        if limit:
            filtered = filtered[: int(limit)]
        self.rows = filtered

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image = normalize_image(np.load(rel_path(self.root, row["image"])))
        mask_path = row_path_value(row, self.mask_key, "mask")
        mask = np.load(rel_path(self.root, mask_path)).astype(np.int64)
        x = torch.from_numpy(image)[None]
        y = torch.from_numpy(mask).long()
        x = resize_tensor_image(x, self.image_size)
        y = resize_tensor_mask(y, self.image_size)
        return {
            "image": x,
            "mask": y,
            "panel_id": row["panel_id"],
            "glass_id": row["glass_id"],
        }


@dataclass
class DefectRecord:
    key: tuple[str, int]
    row: dict[str, Any]
    box: dict[str, Any]
    defect: dict[str, Any]
    area: int


def _find_defect(row: dict[str, Any], instance_id: int, box: dict[str, Any]) -> dict[str, Any]:
    for defect in row.get("defects", []):
        if int(defect.get("instance_id", -1)) == instance_id:
            return defect
    return dict(box)


def build_defect_records(root: Path, split: str, split_mode: str = "manifest") -> list[DefectRecord]:
    """Return one representative panel crop per unique glass instance."""
    best: dict[tuple[str, int], DefectRecord] = {}
    for row in load_manifest(root):
        if assigned_split(row, split_mode) != split:
            continue
        box_path = rel_path(root, row["boxes"])
        with open(box_path, encoding="utf-8") as f:
            box_obj = json.load(f)
        for box in box_obj.get("boxes", []):
            instance_id = int(box["instance_id"])
            key = (str(row["glass_id"]), instance_id)
            x0, y0, x1, y1 = [int(v) for v in box["bbox"]]
            area = max(0, x1 - x0) * max(0, y1 - y0)
            defect = _find_defect(row, instance_id, box)
            rec = DefectRecord(key=key, row=row, box=box, defect=defect, area=area)
            if key not in best or area > best[key].area:
                best[key] = rec
    return list(best.values())


def load_route(root: Path, row: dict[str, Any], route_shuffle: bool = False, seed_key: str = "") -> dict[str, torch.Tensor]:
    with open(rel_path(root, row["route_graph"]), encoding="utf-8") as f:
        graph = json.load(f)
    nodes = list(graph["nodes"])
    if route_shuffle:
        rng = np.random.default_rng(stable_hash_int(seed_key or row["glass_id"]))
        order = rng.permutation(len(nodes)).tolist()
        nodes = [nodes[i] for i in order]
    tool = [TOOL_TO_IDX.get(str(n.get("tool_id", "none")), 0) for n in nodes]
    step = [PROCESS_TO_IDX.get(str(n.get("step", "none")), 0) for n in nodes]
    cycle = [int(n.get("cycle", 0)) for n in nodes]
    pos = [int(n.get("node_pos", i)) for i, n in enumerate(nodes)]
    return {
        "route_tool": torch.tensor(tool, dtype=torch.long),
        "route_step": torch.tensor(step, dtype=torch.long),
        "route_cycle": torch.tensor(cycle, dtype=torch.long),
        "route_pos": torch.tensor(pos, dtype=torch.long),
    }


def crop_defect(image: np.ndarray, bbox: list[int], crop_scale: float = 1.8) -> np.ndarray:
    h, w = image.shape
    x0, y0, x1, y1 = [int(v) for v in bbox]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = max(bw, bh) * float(crop_scale)
    side = max(side, 48.0)
    nx0 = int(max(0, math.floor(cx - side / 2)))
    ny0 = int(max(0, math.floor(cy - side / 2)))
    nx1 = int(min(w, math.ceil(cx + side / 2)))
    ny1 = int(min(h, math.ceil(cy + side / 2)))
    if nx1 <= nx0 or ny1 <= ny0:
        return image
    return image[ny0:ny1, nx0:nx1]


def time_bin_from_defect(defect: dict[str, Any], n_bins: int) -> int:
    tw = defect.get("time_window")
    if not tw:
        return int(n_bins)
    lo, hi = int(tw[0]), int(tw[1])
    mid = 0.5 * (lo + hi)
    total = max(float(hi), 1.0)
    # Use the stored t_norm if available; for excursion windows, the midpoint is
    # a stable label for abnormal time localization.
    frac = float(defect.get("t_norm", mid / total))
    return int(np.clip(math.floor(frac * n_bins), 0, n_bins - 1))


class DefectRCADataset(Dataset):
    """Defect-token dataset for multimodal RCA.

    One item is one unique defect instance represented by the largest panel crop
    where that instance appears.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 224,
        split_mode: str = "manifest",
        route_shuffle: bool = False,
        n_time_bins: int = 10,
        limit: int | None = None,
        route_swap: bool = False,
        route_swap_seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.image_size = int(image_size)
        self.route_shuffle = bool(route_shuffle)
        self.n_time_bins = int(n_time_bins)
        records = build_defect_records(self.root, split=split, split_mode=split_mode)
        records.sort(key=lambda r: (r.key[0], r.key[1]))
        if limit:
            records = records[: int(limit)]
        self.records = records
        # camera-ready control: `route_swap` feeds every defect the route graph of a *different*
        # glass (a fixed derangement of the split's glasses), so the route no longer carries the
        # causal tool of the observed defect while keeping its marginal statistics intact. Unlike
        # `route_shuffle` (node-order permutation), this is not absorbed by permutation-invariant
        # mean-pooling / cross-attention over the nodes.
        self.route_swap = bool(route_swap)
        self.route_swap_map: dict[str, dict[str, Any]] = {}
        if self.route_swap:
            glass_rows: dict[str, dict[str, Any]] = {}
            for rec in records:
                glass_rows.setdefault(str(rec.row["glass_id"]), rec.row)
            glasses = sorted(glass_rows)
            if len(glasses) > 1:
                order = np.random.default_rng(int(route_swap_seed)).permutation(len(glasses)).tolist()
                for i, gi in enumerate(order):
                    donor = glasses[order[(i + 1) % len(order)]]
                    self.route_swap_map[glasses[gi]] = {"glass_id": donor, "route_graph": glass_rows[donor]["route_graph"]}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        row = rec.row
        box = rec.box
        defect = rec.defect
        image = normalize_image(np.load(rel_path(self.root, row["image"])))
        crop = crop_defect(image, box["bbox"])
        crop_u8 = np.clip(crop * 255.0, 0, 255).astype(np.uint8)
        pil = Image.fromarray(crop_u8, mode="L").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0)[None]

        pw, ph = [float(v) for v in row.get("panel_size", [image.shape[1], image.shape[0]])]
        x0, y0, x1, y1 = [float(v) for v in box["bbox"]]
        pos = torch.tensor(
            [
                float(row["panel_row"]) / max(float(row["glass_rows"] - 1), 1.0),
                float(row["panel_col"]) / max(float(row["glass_cols"] - 1), 1.0),
                ((x0 + x1) * 0.5) / max(pw, 1.0),
                ((y0 + y1) * 0.5) / max(ph, 1.0),
                (x1 - x0) / max(pw, 1.0),
                (y1 - y0) / max(ph, 1.0),
                float(row.get("t_norm", 0.0)),
            ],
            dtype=torch.float32,
        )
        route_row = row
        if getattr(self, "route_swap", False):
            route_row = self.route_swap_map.get(str(row["glass_id"]), row)
        route = load_route(self.root, route_row, self.route_shuffle, seed_key=f"{row['glass_id']}:{rec.key[1]}")
        labels = {
            "mask_type": torch.tensor(MASK_TYPE_TO_IDX[str(box["mask_type"])], dtype=torch.long),
            "extent": torch.tensor(EXTENT_TO_IDX[str(box["extent"])], dtype=torch.long),
            "subtype": torch.tensor(SUBTYPE_TO_IDX[str(box["subtype"])], dtype=torch.long),
            "polarity": torch.tensor(POLARITY_TO_IDX[str(box["polarity"])], dtype=torch.long),
            "semu": torch.tensor(float(box["semu"]), dtype=torch.float32),
            "engineer_visible_threshold": torch.tensor(
                float(box.get("engineer_visible_threshold", defect.get("engineer_visible_threshold", 0.0))),
                dtype=torch.float32,
            ),
            "reject_visible_threshold": torch.tensor(
                float(box.get("reject_visible_threshold", defect.get("reject_visible_threshold", 0.0))),
                dtype=torch.float32,
            ),
            "is_reject": torch.tensor(float(bool(box["is_reject"])), dtype=torch.float32),
            "process_step": torch.tensor(PROCESS_TO_IDX.get(str(box.get("process_step", "none")), 0), dtype=torch.long),
            "tool_id": torch.tensor(TOOL_TO_IDX.get(str(box.get("tool_id", "none")), 0), dtype=torch.long),
            "fault_mode": torch.tensor(FAULT_MODE_TO_IDX.get(str(box.get("fault_mode", "none")), 0), dtype=torch.long),
            "time_bin": torch.tensor(time_bin_from_defect(defect, self.n_time_bins), dtype=torch.long),
        }
        return {
            "image": x,
            "position": pos,
            **route,
            **labels,
            "glass_id": row["glass_id"],
            "panel_id": row["panel_id"],
            "instance_id": int(rec.key[1]),
        }


def collate_dict(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [b[key] for b in batch]
        if torch.is_tensor(values[0]):
            out[key] = torch.stack(values)
        else:
            out[key] = values
    return out


def summarize_defect_dataset(root: str | Path) -> dict[str, dict[str, int]]:
    root = Path(root)
    records_by_split = {split: build_defect_records(root, split=split) for split in ["train", "val", "test"]}
    summary: dict[str, dict[str, int]] = {}
    for split, records in records_by_split.items():
        counts = defaultdict(int)
        for rec in records:
            counts[f"subtype/{rec.box['subtype']}"] += 1
            counts[f"mask_type/{rec.box['mask_type']}"] += 1
            counts[f"reject/{rec.box['is_reject']}"] += 1
        summary[split] = dict(sorted(counts.items()))
    return summary
