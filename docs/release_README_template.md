# MuraTrace v{VERSION} — data package

Real-background synthetic benchmark for multi-scale display Mura inspection with paired
perceptual and process-trace annotations (WACV 2027). Built {DATE}.

* {N_GLASSES} synthetic glasses (8×7 panels each), {N_PANELS} panel images, {N_INSTANCES} visible defect instances
  in five classes (spot, line, region, gradient, band) and eight subtypes
* every instance: dense mask, bounding box, subtype, polarity, SEMU visibility, two reject labels
  (author thresholds `is_reject`; expert-consensus thresholds `is_reject_expert` = per-subtype rater
  medians applied to `panel_instance_semu`, 1,715 → 2,188 unique reject instances), process step, tool,
  fault mode, time window and route-graph node
* every glass: a 21-node route graph (default configuration) and, in `ambiguous/`, the
  ambiguous-cause configuration in which the causal tool is written only into the route graph

Code, baselines and documentation: https://github.com/su9yeong/MuraTrace — archive DOI: 10.5281/zenodo.23889760

## Layout

```
manifest.jsonl              one row per panel: ids, grid position, split, source background, paths, t_norm, defects[]
splits/background_level.json   the paper's protocol (18/6/6 real backgrounds -> train/val/test)
splits/glass_disjoint.json  glass-disjoint protocol for process-trace experiments
configs/                    generation config + SEMU thresholds (author / expert consensus) + config as run
panels/images/*.png         uint16 luminance, 727x359 (12-bit range)
panels/masks/*.png          uint8 semantic mask: 0 background, 1 spot, 2 line, 3 region, 4 gradient, 5 band
panels/train_masks/*.png    training mask (engineer-visible policy; see dataset_card.md)
panels/inst/*.png           uint16 instance ids
panels/boxes/*.json         per-instance annotations (see below); train_boxes = training subset
panels/yolo/*.txt           YOLO export of the boxes (class cx cy w h, normalized)
glass/route_graphs/*.json   default configuration route graphs
ambiguous/                  ambiguous-cause configuration: manifest, boxes, route graphs (panels are shared)
backgrounds/                30 clean backgrounds + 540 augmented variants (if included)
expert_study/               perceptual-study form and anonymized threshold responses
DATASHEET.md, dataset_card.md, LICENSE, SHA256SUMS, release_summary.json
```

Per-instance fields (boxes/*.json → `boxes[]`, and manifest `defects[]`): `instance_id`, `mask_type`,
`class_id`, `subtype`, `extent`, `polarity`, `bbox`, `semu` (panel-instance SEMU, 95th percentile over
visible pixels), `is_visible`, `is_reject`, `is_visible_expert`, `is_reject_expert`, `process_step`,
`tool_id`, `fault_mode`, `time_window`, `root_cause`, `target_cycle`, `route_node_positions`.

## Loading

```python
import json, numpy as np, cv2
rows = [json.loads(l) for l in open("manifest.jsonl")]
r = rows[0]
img  = cv2.imread(r["image"], cv2.IMREAD_UNCHANGED)   # uint16 (359, 727)
mask = cv2.imread(r["mask"],  cv2.IMREAD_UNCHANGED)   # uint8
inst = cv2.imread(r["inst"],  cv2.IMREAD_UNCHANGED)   # uint16
boxes = json.load(open(r["boxes"]))["boxes"]
```
Normalize images as in the baselines: `img.astype(np.float32) / 4095.0`.

## Verify

```
python tools/verify_release.py --root . --sha      # from the code repository
```

## License and citation

Data: CC BY 4.0 (see LICENSE). If you use MuraTrace, cite:

```bibtex
@inproceedings{lee2027muratrace,
  title     = {MuraTrace: A Real-Background Synthetic Benchmark for Multi-Scale Display Mura Inspection},
  author    = {Lee, Sugyeong and Lee, Dong-Hee},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## Contact

Contact: Sugyeong Lee, sugyeong@g.skku.edu or sugyeong.el.lee@gmail.com.

## Known limitations

All 30 real backgrounds come from one inspection line and share its low-frequency illumination
field; defects are synthetic (multiplicative luminance fields with SEMU-controlled visibility); the
step/tool mapping of the process-trace labels is a documented modeling choice. See DATASHEET.md.
