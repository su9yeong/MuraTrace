"""Analyze the expert perceptual survey (run AFTER collecting real responses).

Reads realism_responses_template.csv and threshold_responses_template.csv in this
folder, skips placeholder rows (rater_id starting with 'EXAMPLE'), and prints the
numbers needed for the paper: realism O-rate and mean score, inter-rater
agreement, and expert vs. dataset SEMU thresholds.

  python expert_study/analyze_survey.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
THRESH_JSON = ROOT / "configs" / "semu_visibility_thresholds_realbg_visible.json"


def read(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not str(r.get("rater_id", "")).startswith("EXAMPLE")]
    # drop fully-blank rows (unfilled form rows have empty rater_id and answers)
    return [r for r in rows if str(r.get("rater_id", "")).strip()]


def read_forms(pattern):
    """Read all per-rater form files matching forms/<pattern>, skipping *_blank.csv."""
    out = []
    for p in sorted((HERE / "forms").glob(pattern)):
        if p.name.endswith("_blank.csv"):
            continue
        out += read(p)
    return out


def dataset_thresholds():
    if not THRESH_JSON.exists():
        return {}
    obj = json.load(open(THRESH_JSON, encoding="utf-8"))
    # tolerate a couple of schema shapes
    out = {}
    src = obj.get("thresholds", obj)
    for k, v in src.items():
        if isinstance(v, dict):
            out[k] = (v.get("engineer_visible"), v.get("reject_visible"))
    return out


def main():
    realism = read(HERE / "realism_responses_template.csv") + read_forms("partA_*.csv")
    thresh = read(HERE / "threshold_responses_template.csv") + read_forms("partB_*.csv")
    if not realism and not thresh:
        print("No real responses found yet. Fill the CSV templates (remove EXAMPLE rows).")
        return

    print("=== Realism (Part A) ===")
    raters = sorted({r["rater_id"] for r in realism})
    print(f"raters: {len(raters)}")
    if realism:
        o = sum(1 for r in realism if r["seen_in_field"].strip().upper() == "O")
        print(f"seen-in-field O-rate: {100*o/len(realism):.1f}%  (n={len(realism)})")
        print(f"mean realism (1-5):   {st.mean(float(r['realism_1to5']) for r in realism):.2f}")
        bycls = defaultdict(list)
        for r in realism:
            bycls[r["class"]].append((r["seen_in_field"].strip().upper() == "O", float(r["realism_1to5"])))
        print("  per class (O-rate / mean realism):")
        for c, v in sorted(bycls.items()):
            print(f"    {c:9s} {100*sum(a for a,_ in v)/len(v):5.1f}%  {st.mean(b for _,b in v):.2f}")
        # simple inter-rater agreement on Q1: per sample, fraction agreeing with the majority
        bysample = defaultdict(list)
        for r in realism:
            bysample[r["sample_id"]].append(r["seen_in_field"].strip().upper() == "O")
        agrees = [max(s.count(True), s.count(False)) / len(s) for s in bysample.values() if s]
        if agrees:
            print(f"  Q1 inter-rater agreement (majority): {100*st.mean(agrees):.1f}%")

    print("\n=== SEMU thresholds (Part B): expert vs. dataset ===")
    dt = dataset_thresholds()
    bysub = defaultdict(lambda: {"perc": [], "rej": []})
    for r in thresh:
        bysub[r["subtype"]]["perc"].append(float(r["perception_semu"]))
        bysub[r["subtype"]]["rej"].append(float(r["reject_semu"]))
    print(f"{'subtype':24s} {'exp_perc':>8} {'ds_eng':>7} {'exp_rej':>8} {'ds_rej':>7}")
    for sub, v in sorted(bysub.items()):
        ep = st.mean(v["perc"]); er = st.mean(v["rej"])
        de, dr = dt.get(sub, (None, None))
        print(f"{sub:24s} {ep:8.2f} {str(de):>7} {er:8.2f} {str(dr):>7}")
    if thresh:
        print("\n(Use mean expert perception/reject SEMU and their closeness to the dataset "
              "engineer_visible/reject_visible thresholds to fill the paper.)")


if __name__ == "__main__":
    main()
