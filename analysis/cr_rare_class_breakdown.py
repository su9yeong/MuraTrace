# -*- coding: utf-8 -*-
"""Camera-ready analysis (reviewer sDje): does class imbalance hurt the rare
classes (gradient, band) in Task 2 (process-trace) and Task 3 (visibility) the
way it does in segmentation?

(a) Process-trace attribution: per visual class / per subtype Hit@1 and Hit@3
    for each head, from the saved test predictions of the default and the
    ambiguous-cause configurations.
(b) Visibility regression: per class / per subtype MAE, R^2, Pearson r and the
    reject-F1 obtained by thresholding the regressed SEMU at the subtype's
    reject threshold, from the dedicated regressor's saved test predictions.

Outputs -> analysis_camera_ready/
    rca_breakdown.csv, rca_breakdown.md, semu_breakdown.csv, semu_breakdown.md,
    semu_test_instances.csv (per-instance true/pred/class/subtype)

Run with the training virtualenv (needs training/mura_rca_data importable to
unpickle the cached DefectRecord list):
    virtualenv\\Scripts\\python.exe scripts\\cr_rare_class_breakdown.py
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
import mura_rca_data  # noqa: F401  (module name needed to unpickle DefectRecord)

OUT = ROOT / "analysis_camera_ready"
OUT.mkdir(exist_ok=True)

CLASSES = ["spot", "line", "region", "gradient", "band"]
SUBTYPES = ["particle_spot", "random_streak", "stitch_line", "puddle_blob",
            "thermal_blob", "anneal_radial_gradient", "row_banding", "column_banding"]
HEADS = ["tool_id", "process_step", "fault_mode", "time_bin"]
HEAD_SHORT = {"tool_id": "tool", "process_step": "step", "fault_mode": "fault", "time_bin": "time"}

CR_SEEDS = (7, 17, 27)
RCA_RUNS = {
    "default/image+position": ROOT / "analysis_wacv_realbg_visible/rca_test_eval/rca_image_position/test_predictions.jsonl",
    "default/cross-fusion": ROOT / "analysis_wacv_realbg_visible/rca_test_eval/rca_visionroute/test_predictions.jsonl",
    "ambiguous/image+position": ROOT / "analysis_wacv_ambiguous/rca_test_eval/rca_image_position/test_predictions.jsonl",
    "ambiguous/late-fusion": ROOT / "analysis_wacv_ambiguous/rca_test_eval/rca_late_fusion/test_predictions.jsonl",
    "ambiguous/cross-fusion": ROOT / "analysis_wacv_ambiguous/rca_test_eval/rca_visionroute/test_predictions.jsonl",
    # camera-ready consistent-protocol re-runs (ConvNeXt crop encoder, batch 32); a list = averaged over seeds
    "primary-cr/image+position": [ROOT / f"analysis_camera_ready/rca_ambiguous/rca_image_position_s{s}/test_predictions.jsonl" for s in CR_SEEDS],
    "primary-cr/late-fusion": [ROOT / f"analysis_camera_ready/rca_ambiguous/rca_late_fusion_s{s}/test_predictions.jsonl" for s in CR_SEEDS],
    "primary-cr/cross-fusion": [ROOT / f"analysis_camera_ready/rca_ambiguous/rca_visionroute_s{s}/test_predictions.jsonl" for s in CR_SEEDS],
    "primary-cr/cross-fusion-swap": [ROOT / f"analysis_camera_ready/rca_ambiguous/rca_visionroute_swap_s{s}/test_predictions.jsonl" for s in CR_SEEDS],
}
METRIC_KEYS = [f"{HEAD_SHORT[h]}_hit1" for h in HEADS] + [f"{HEAD_SHORT[h]}_hit3" for h in HEADS] + ["mean_hit1", "mean_hit3"]


def hit(rec: dict, head: str, k: int) -> int:
    lab = rec["labels"][head]["id"]
    if k == 1:
        return int(rec["predictions"][head]["id"] == lab)
    return int(lab in [t["id"] for t in rec["top3"][head][:k]])


def fmt(x: float) -> str:
    return f"{x:.3f}"


def cell(r: dict, k: str) -> str:
    return fmt(r[k]) + (f"+-{r[k + '_std']:.3f}" if r.get("n_seeds", 1) > 1 else "")


def rca_breakdown() -> list[dict]:
    rows: list[dict] = []
    for name, path in RCA_RUNS.items():
        paths = [p for p in (path if isinstance(path, list) else [path]) if p.exists()]
        if not paths:
            print(f"  [skip] {name}: {path} missing")
            continue
        per_seed: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for p in paths:
            recs = [json.loads(l) for l in open(p, encoding="utf-8")]
            groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
            for r in recs:
                groups[("all", "all")].append(r)
                groups[("class", r["labels"]["mask_type"]["name"])].append(r)
                groups[("subtype", r["labels"]["subtype"]["name"])].append(r)
            for (kind, g), rs in groups.items():
                row = {"n": len(rs)}
                for h in HEADS:
                    row[f"{HEAD_SHORT[h]}_hit1"] = float(np.mean([hit(r, h, 1) for r in rs]))
                    row[f"{HEAD_SHORT[h]}_hit3"] = float(np.mean([hit(r, h, 3) for r in rs]))
                row["mean_hit1"] = float(np.mean([row[f"{HEAD_SHORT[h]}_hit1"] for h in HEADS]))
                row["mean_hit3"] = float(np.mean([row[f"{HEAD_SHORT[h]}_hit3"] for h in HEADS]))
                per_seed[(kind, g)].append(row)
        # average over seeds (population std, as in cr_collect_results.py)
        for (kind, g), srows in per_seed.items():
            row = {"run": name, "kind": kind, "group": g, "n": srows[0]["n"], "n_seeds": len(srows)}
            for k in METRIC_KEYS:
                v = [sr[k] for sr in srows]
                row[k] = float(np.mean(v))
                row[f"{k}_std"] = float(np.std(v))
            rows.append(row)
    cols = ["run", "kind", "group", "n", "n_seeds"] + METRIC_KEYS + [f"{k}_std" for k in METRIC_KEYS]
    with open(OUT / "rca_breakdown.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    # markdown: one table per run, class-level then subtype-level
    md = ["# Process-trace attribution: per-class / per-subtype breakdown (test)\n",
          "primary-cr rows: mean (+-population std) over seeds 7/17/27 of the camera-ready re-runs.\n"]
    for name in RCA_RUNS:
        sub = [r for r in rows if r["run"] == name]
        if not sub:
            continue
        md.append(f"\n## {name}\n")
        md.append("| group | n | mean H@1 | tool H@1 | step H@1 | fault H@1 | time H@1 | tool H@3 | mean H@3 |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        order = [("all", "all")] + [("class", c) for c in CLASSES] + [("subtype", s) for s in SUBTYPES]
        for kind, g in order:
            for r in sub:
                if r["kind"] == kind and r["group"] == g:
                    md.append(f"| {g} | {r['n']} | {cell(r, 'mean_hit1')} | {cell(r, 'tool_hit1')} | {cell(r, 'step_hit1')} | "
                              f"{cell(r, 'fault_hit1')} | {cell(r, 'time_hit1')} | {cell(r, 'tool_hit3')} | {cell(r, 'mean_hit3')} |")
    (OUT / "rca_breakdown.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return rows


def r2(y: np.ndarray, p: np.ndarray) -> float:
    sst = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - p) ** 2).sum()) / sst if sst > 1e-12 else float("nan")


def f1(tp: int, fp: int, fn: int) -> float:
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else float("nan")


def semu_breakdown() -> list[dict]:
    cache = ROOT / "runs_semu_reg/_cache/records_test.pkl"
    npz_path = ROOT / "runs_semu_reg/semu_regression_preds.npz"
    recs = pickle.load(open(cache, "rb"))
    recs.sort(key=lambda r: (r.key[0], r.key[1]))
    npz = np.load(npz_path)
    tags = sorted({k.rsplit("_", 1)[0] for k in npz.files})
    n = len(recs)
    for t in tags:
        assert len(npz[f"{t}_true"]) == n, (t, len(npz[f"{t}_true"]), n)
    semu_box = np.array([float(r.box["semu"]) for r in recs])
    # sanity: the cached order must reproduce the saved ground truth
    for t in tags:
        assert np.allclose(npz[f"{t}_true"], semu_box, atol=1e-4), f"order mismatch for {t}"
    cls = np.array([r.box["mask_type"] for r in recs])
    sub = np.array([r.box["subtype"] for r in recs])
    rej = np.array([bool(r.box["is_reject"]) for r in recs])
    rthr = np.array([float(r.box.get("reject_visible_threshold", r.defect.get("reject_visible_threshold", np.nan))) for r in recs])

    # per-instance export
    with open(OUT / "semu_test_instances.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["glass_id", "instance_id", "class", "subtype", "semu_true", "is_reject", "reject_threshold"] + [f"pred_{t}" for t in tags])
        for i, r in enumerate(recs):
            w.writerow([r.key[0], r.key[1], cls[i], sub[i], f"{semu_box[i]:.4f}", int(rej[i]), rthr[i]] + [f"{npz[f'{t}_pred'][i]:.4f}" for t in tags])

    rows: list[dict] = []
    groups = [("all", "all", np.ones(n, bool))] + [("class", c, cls == c) for c in CLASSES] + [("subtype", s, sub == s) for s in SUBTYPES]
    for kind, g, m in groups:
        if m.sum() == 0:
            continue
        y = semu_box[m]
        row = {"kind": kind, "group": g, "n": int(m.sum()), "semu_mean": float(y.mean()), "semu_std": float(y.std()),
               "reject_frac": float(rej[m].mean())}
        for t in tags:
            p = npz[f"{t}_pred"][m]
            row[f"{t}_mae"] = float(np.abs(y - p).mean())
            row[f"{t}_r2"] = r2(y, p)
            row[f"{t}_pearson"] = float(np.corrcoef(y, p)[0, 1]) if y.std() > 1e-9 and p.std() > 1e-9 else float("nan")
            pred_rej = p >= rthr[m]
            true_rej = rej[m]
            tp = int((pred_rej & true_rej).sum()); fp = int((pred_rej & ~true_rej).sum()); fn = int((~pred_rej & true_rej).sum())
            row[f"{t}_reject_f1"] = f1(tp, fp, fn)
        rows.append(row)
    cols = list(rows[0].keys())
    with open(OUT / "semu_breakdown.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    md = ["# Visibility regression (dedicated regressor, test): per-class / per-subtype breakdown\n",
          "reject-F1 = threshold the regressed SEMU at the subtype reject threshold.\n"]
    for t in tags:
        md.append(f"\n## {t}\n")
        md.append("| group | n | SEMU mean | SEMU std | reject frac | MAE | R^2 | Pearson r | reject F1 |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            md.append(f"| {r['group']} | {r['n']} | {fmt(r['semu_mean'])} | {fmt(r['semu_std'])} | {fmt(r['reject_frac'])} | "
                      f"{fmt(r[f'{t}_mae'])} | {fmt(r[f'{t}_r2'])} | {fmt(r[f'{t}_pearson'])} | {fmt(r[f'{t}_reject_f1'])} |")
    (OUT / "semu_breakdown.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    print("process-trace breakdown ...", flush=True)
    rca_rows = rca_breakdown()
    if "--rca_only" in sys.argv:  # camera-ready: the visibility breakdown below reads the submitted regressor's preds
        print((OUT / "rca_breakdown.md").read_text(encoding="utf-8"))
        print(f"wrote {len(rca_rows)} rca rows -> {OUT}")
        return
    print("visibility breakdown ...", flush=True)
    semu_rows = semu_breakdown()
    print((OUT / "rca_breakdown.md").read_text(encoding="utf-8"))
    print((OUT / "semu_breakdown.md").read_text(encoding="utf-8"))
    print(f"wrote {len(rca_rows)} rca rows, {len(semu_rows)} semu rows -> {OUT}")


if __name__ == "__main__":
    main()
