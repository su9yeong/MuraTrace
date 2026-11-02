"""Build the ambiguous-cause RCA variant of the benchmark.

The causal tool is randomized among the interchangeable tools of the SAME process step and recorded ONLY in the route graph. 
The defect appearance still gives the subtype (-> step), but WHICH tool (equipment) at that step is causal is 
no longer determined by appearance or position -- only the route graph encodes it. So:
  image_position  : localizes the causal node but the equipment there is random -> tool fails
  image_route     : knows the route tools but cannot localize the node          -> tool fails
  late/visionroute: position localizes the node AND route reads the tool        -> tool recovered
The route graph is therefore necessary (no longer redundant) to recover the tool.

Images/masks/train_boxes are reused by reference (relative path back to the source root); 
only route graphs, the RCA `boxes`, and the manifest are rewritten.

  python generation/supplement_ambiguous.py
-> data/MuraTrace_ambiguous/
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "MuraTrace"
DST = ROOT / "data" / "MuraTrace_ambiguous"
BACK = "../MuraTrace/"          # DST and SRC are siblings
TOOLS_PER_STEP = [2, 3, 3, 2, 3, 2, 2]             # must match the route-graph tool layout
SEED = 207
# fields that point at large unchanged assets -> reference the source copy
REF_FIELDS = ["image", "mask", "train_mask", "inst", "train_boxes", "yolo", "train_yolo"]


def srel(p):
    return SRC / str(p).replace("\\", "/")


def step_tools(step_index: int):
    n = step_index + 1
    return [f"S{n}_T{t}" for t in range(1, TOOLS_PER_STEP[step_index] + 1)]


def main():
    rng = random.Random(SEED)
    (DST / "route_graphs").mkdir(parents=True, exist_ok=True)
    (DST / "boxes").mkdir(parents=True, exist_ok=True)
    print("reading manifest ...", flush=True)
    rows = [json.loads(l) for l in open(SRC / "manifest.jsonl", encoding="utf-8")]
    print(f"  {len(rows)} panel rows; randomizing route graphs ...", flush=True)

    # pass 1: randomize each glass's route-graph node tools (equipment within step)
    glass_node_tool: dict[str, dict[int, str]] = {}
    for r in rows:
        g = r["glass_id"]
        if g in glass_node_tool or not r.get("route_graph"):
            continue
        rg = json.load(open(srel(r["route_graph"]), encoding="utf-8"))
        for nd in rg["nodes"]:
            si = int(nd.get("step_index", int(nd["node_pos"]) % 7))
            cands = step_tools(si)
            nd["tool_id"] = cands[rng.randrange(len(cands))]
        name = Path(r["route_graph"]).name
        json.dump(rg, open(DST / "route_graphs" / name, "w", encoding="utf-8"))
        glass_node_tool[g] = {int(nd["node_pos"]): nd["tool_id"] for nd in rg["nodes"]}

    print(f"  {len(glass_node_tool)} route graphs done; relabeling boxes/manifest "
          f"(~12K small files, ~1-3 min) ...", flush=True)
    # pass 2: relabel boxes + manifest defects to the route's (randomized) causal tool
    n_box, n_def = 0, 0
    out = open(DST / "manifest.jsonl", "w", encoding="utf-8")
    for i, r in enumerate(rows):
        if i and i % 5000 == 0:
            print(f"    {i}/{len(rows)} rows", flush=True)
        node_tool = glass_node_tool.get(r["glass_id"], {})
        inst_node = {int(d["instance_id"]): d.get("primary_route_node_pos")
                     for d in r.get("defects", []) if d.get("instance_id") is not None}

        if r.get("boxes"):
            bj = json.load(open(srel(r["boxes"]), encoding="utf-8"))
            for b in bj.get("boxes", []):
                npos = inst_node.get(int(b["instance_id"]))
                if npos is not None and npos in node_tool:
                    b["tool_id"] = node_tool[npos]; n_box += 1
            name = Path(r["boxes"]).name
            json.dump(bj, open(DST / "boxes" / name, "w", encoding="utf-8"))
            r["boxes"] = f"boxes/{name}"

        for d in r.get("defects", []):
            npos = d.get("primary_route_node_pos")
            if npos is not None and npos in node_tool:
                d["tool_id"] = node_tool[npos]; n_def += 1

        if r.get("route_graph"):
            r["route_graph"] = f"route_graphs/{Path(r['route_graph']).name}"
        for k in REF_FIELDS:
            if r.get(k):
                r[k] = BACK + str(r[k]).replace("\\", "/")
        out.write(json.dumps(r) + "\n")
    out.close()

    for fn in ["config.json", "dataset_card.md"]:
        if (SRC / fn).exists():
            shutil.copy(SRC / fn, DST / fn)
    print(f"wrote {DST}  (relabeled {n_box} boxes, {n_def} manifest defects, "
          f"{len(glass_node_tool)} route graphs)")


if __name__ == "__main__":
    main()
