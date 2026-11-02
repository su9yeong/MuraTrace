# -*- coding: utf-8 -*-
"""Collect every camera-ready experiment output that exists so far into one
Markdown summary with LaTeX-ready rows, so the \\TODO placeholders in
tex_camera_ready can be filled without hand-copying numbers.

    python scripts/cr_collect_results.py        # -> analysis_camera_ready/RESULTS_SUMMARY.md
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ANA = ROOT / "analysis_camera_ready"
HEADS = ["process_step", "tool_id", "fault_mode", "time_bin"]
STRICT = {"bg14", "bg15", "bg25"}
REPEAT = {"bg07", "bg18", "bg23"}


def f3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def rca_row(summary_path: Path) -> dict:
    s = json.load(open(summary_path, encoding="utf-8"))
    m = s["metrics"]
    h1 = [m[f"{h}_hit1"] for h in HEADS]; h3 = [m[f"{h}_hit3"] for h in HEADS]
    return {"epoch": s["checkpoint_epoch"], "mean1": float(np.mean(h1)), "mean3": float(np.mean(h3)),
            "step": h1[0], "tool": h1[1], "fault": h1[2], "time": h1[3], "tool3": h3[1]}


def main() -> None:
    md = ["# Camera-ready results summary (auto-collected)\n"]

    # ------------------------------------------------------------ segmentation per background
    md.append("## Segmentation: per test background (mIoU_def over classes present) and strict/repeat subsets\n")
    md.append("| model | all | " + " | ".join(sorted(STRICT | REPEAT)) + " | strict | repeat | spread (min-max) |")
    md.append("|---|---:|" + "---:|" * 6 + "---:|---:|---|")
    for f in sorted(glob.glob(str(ANA / "seg_per_source_*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        ps = d["per_source"]
        caps = sorted(ps)
        confs = {k: np.array(v["confusion"]) for k, v in ps.items() if "confusion" in v}

        def miou(keys):
            c = sum(confs[k] for k in keys if k in confs)
            tp = np.diag(c).astype(float); den = c.sum(0) + c.sum(1) - tp
            iou = np.where(den > 0, tp / np.maximum(den, 1), np.nan)
            return float(np.nanmean(iou[1:]))
        vals = [ps[c]["miou_defect_present"] for c in caps]
        name = Path(f).stem.replace("seg_per_source_", "")
        st = miou(STRICT) if confs else float("nan"); rp = miou(REPEAT) if confs else float("nan")
        md.append(f"| {name} | {f3(d['all']['miou_defect_present'])} | " + " | ".join(f3(v) for v in vals)
                  + f" | {f3(st)} | {f3(rp)} | {f3(min(vals))}-{f3(max(vals))} |")
        md.append(f"|  ↳ LaTeX | " + " & ".join(f3(v) for v in vals) + f" & {f3(st)} & {f3(rp)} \\\\ |" + " |" * 7)

    # ------------------------------------------------------------ background ablation
    md.append("\n## Background-count ablation (DeepLabV3+ ConvNeXt-S, 24k steps) test mIoU_def\n")
    md.append("| run | mIoU_def | spot | line | region | gradient | band | best step |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    abl = {}
    for f in sorted(glob.glob(str(ANA / "bg_ablation" / "*" / "test_summary.json"))):
        d = json.load(open(f, encoding="utf-8")); m = d["metrics"]; name = Path(f).parent.name
        abl[name] = m
        bp = ROOT / "runs_cr_bgablation" / name / "best_point.json"
        step = json.load(open(bp)).get("steps_done", "?") if bp.exists() else "?"
        md.append(f"| {name} | {f3(m['miou_defect'])} | {f3(m['iou_1'])} | {f3(m['iou_2'])} | {f3(m['iou_3'])} | {f3(m['iou_4'])} | {f3(m['iou_5'])} | {step} |")
    if abl:
        row_k = " & ".join(f3(abl.get(k, {}).get("miou_defect")) for k in ("k03_bg3", "k06_bg6", "k12_bg12", "k18_full"))
        row_m = " & ".join(f3(abl.get(k, {}).get("miou_defect")) for k in ("k03_match18", "k06_match18", "k12_match18")) + " & --"
        md.append(f"\nLaTeX rows for tab:bgabl:\n    $k$ backgrounds & {row_k} \\\\\n    18 backgrounds, same panels & {row_m} \\\\")

    # ------------------------------------------------------------ ambiguous RCA (camera-ready protocol)
    md.append("\n## Process-trace, primary protocol (ambiguous-cause; ConvNeXt crop encoder, batch 32)\n")
    md.append("| run | seed | epoch | mean@1 | mean@3 | step | tool | fault | time | tool@3 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    by_mode: dict[str, list[dict]] = {}
    for f in sorted(glob.glob(str(ANA / "rca_ambiguous" / "*" / "test_summary.json"))):
        name = Path(f).parent.name  # rca_<mode>_s<seed>
        mode, seed = name[4:].rsplit("_s", 1)
        r = rca_row(Path(f)); r["seed"] = int(seed)
        by_mode.setdefault(mode, []).append(r)
        md.append(f"| {mode} | {seed} | {r['epoch']} | {f3(r['mean1'])} | {f3(r['mean3'])} | {f3(r['step'])} | {f3(r['tool'])} | {f3(r['fault'])} | {f3(r['time'])} | {f3(r['tool3'])} |")
    if by_mode:
        md.append("\nLaTeX rows (seed 7 value; mean±std over seeds where >1):")
        for mode, rs in by_mode.items():
            def agg(k):
                v = [r[k] for r in rs]
                return f"{np.mean(v):.3f}" + (f"$\\pm${np.std(v):.3f}" if len(v) > 1 else "")
            md.append(f"    {mode} ({len(rs)} seeds): & {agg('mean1')} & {agg('mean3')} & {agg('step')} & {agg('tool')} & {agg('fault')} & {agg('time')} \\\\")

    # ------------------------------------------------------------ default RCA controls
    md.append("\n## Process-trace, default configuration controls (small CNN, batch 64)\n")
    for f in sorted(glob.glob(str(ANA / "rca_default" / "*" / "test_summary.json"))):
        r = rca_row(Path(f)); name = Path(f).parent.name
        md.append(f"- {name}: mean@1 {f3(r['mean1'])} mean@3 {f3(r['mean3'])} step {f3(r['step'])} tool {f3(r['tool'])} fault {f3(r['fault'])} time {f3(r['time'])} (epoch {r['epoch']})")

    # ------------------------------------------------------------ visibility regression
    # camera-ready protocol = 100 epochs, constant LR, fp32 2x64 micro-batches -> runs_semu_reg_cr100* (pooled);
    # runs_semu_reg_cr/ holds the superseded 20-epoch cosine runs and is deliberately NOT read here.
    runs = []
    for rp in sorted(glob.glob(str(ROOT / "runs_semu_reg_cr100*" / "semu_regression_result.json"))):
        runs += json.load(open(rp, encoding="utf-8"))["per_run"]
    if runs:
        md.append("\n## Visibility regression (camera-ready protocol: 100 ep, const LR; pooled over runs_semu_reg_cr100*; mean ± population std over seeds)\n")
        md.append("| variant | seeds | R2 | MAE |"); md.append("|---|---:|---:|---:|")
        by_tag: dict[str, list[dict]] = {}
        for r in runs:
            by_tag.setdefault(r["tag"], []).append(r)
        for tag, rs in by_tag.items():
            r2 = [r["test_r2"] for r in rs]; mae = [r["test_mae"] for r in rs]
            md.append(f"| {tag} | {len(rs)} | {np.mean(r2):.3f} ± {np.std(r2):.3f} | {np.mean(mae):.3f} ± {np.std(mae):.3f} |")
        md.append("\nLaTeX rows: " + " ".join(f"{tag}: ${np.mean([r['test_r2'] for r in rs]):.3f}\\pm{np.std([r['test_r2'] for r in rs]):.3f}$ & ${np.mean([r['test_mae'] for r in rs]):.3f}\\pm{np.std([r['test_mae'] for r in rs]):.3f}$ \\\\" for tag, rs in by_tag.items()))
        md.append("\nper run: " + "; ".join(f"{r['tag']} s{r['seed']}: R2 {r['test_r2']:.3f} (val {r['best_val_r2']:.3f})" for r in runs))

    # ------------------------------------------------------------ sim-to-real
    s2r = sorted(glob.glob(str(ANA / "sim2real_*.json")))
    if s2r:
        md.append("\n## Zero-shot real-data check (Qiu et al. weak-Mura)\n")
        for f in s2r:
            d = json.load(open(f, encoding="utf-8")); o = d.get("overall", {}); s = d.get("settings", {})
            md.append(f"- {Path(f).stem}: tiles {d.get('n_tiles')} boxes {o.get('boxes')} recall(any) {o.get('recall_any', o.get('recall', float('nan'))):.3f} "
                      f"chance(any) {o.get('chance_any', float('nan')):.3f} recall(spot) {o.get('recall_spot', float('nan')):.3f} chance(spot) {o.get('chance_spot', float('nan')):.3f} "
                      f"FA/def-tile {o.get('fa_per_defective_tile', float('nan')):.2f} clean-alarm {o.get('clean_tile_alarm_rate', float('nan')):.3f}; "
                      f"scale={s.get('scale')} denoise={s.get('denoise_sigma')}")

    (ANA / "RESULTS_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
