# MuraTrace

**A Real-Background Synthetic Benchmark for Multi-Scale Display Mura Inspection** — WACV 2027 (Datasets track)

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23889760-blue)](https://doi.org/10.5281/zenodo.23889760)
[![Data license: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)
[![Code license: MIT](https://img.shields.io/badge/code-MIT-lightgrey)](LICENSE)

MuraTrace composites physically motivated Mura defects (multiplicative luminance fields with
SEMU-controlled visibility) onto real defect-free OLED-mobile panel backgrounds and pairs every defect
with dense perceptual annotations **and** process-trace ground truth (route graph, causal tool, fault
mode, time window). 540 glasses × 56 panels = 30,240 panel images, 5,736 visible defect instances,
five classes (spot, line, region, gradient, band), eight subtypes, three tasks:

| task | input → output | metric |
|---|---|---|
| 1 scale-wise segmentation | panel → 6-class mask | mIoU over the defect classes |
| 2 process-trace attribution (default and ambiguous-cause configurations) | defect crop + panel position/time + glass route graph → step / tool / fault / time bin | Hit@1 / Hit@3 |
| 3 visibility prediction | defect crop (+ position) → panel-instance SEMU | R², MAE |

## Get the data

Archive (Zenodo, DOI): https://doi.org/10.5281/zenodo.23889760

```bash
python tools/download.py --zenodo 23889760 --out data/MuraTrace
python tools/verify_release.py --root data/MuraTrace --sha
```

Package layout, per-instance fields and a loading snippet: `docs/release_README_template.md`
(shipped as README.md inside the package); schema details: `docs/schema.md`; datasheet: `docs/DATASHEET.md`.

## Install

```bash
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt                          # training / evaluation (PyTorch 2.11, timm, segmentation_models_pytorch)
pip install -r requirements_gen.txt                      # generation only (numpy, Pillow)
```

## Reproduce

Generation (from the 30 real backgrounds, `generation/` — see `docs/schema.md` and the supplementary material, Sec. S1):

```bash
python generation/main_augmentbg.py --input backgrounds/clean --out bgbank
python generation/main_defectgen.py --bg_dir bgbank/backgrounds_aug --out data/MuraTrace_regen
python generation/supplement_updateth.py --data_root data/MuraTrace_regen --thresholds configs/muratrace_thresholds.json
python generation/supplement_ambiguous.py          # ambiguous-cause label configuration
```

Baselines (`training/`; every script prints its options with `--help`):

```bash
# Task 1 — segmentation (e.g. UPerNet Swin-S); per-background breakdown
python training/train_segmentation_advanced.py --help
python training/eval_seg_per_source.py --checkpoint <ckpt> --data_root data/MuraTrace --split test --out results/per_source.json
# Task 2 — process-trace attribution (ambiguous-cause labels)
python training/train_rca_advanced.py --help           # --split_mode manifest | random_glass | random_panel
python training/evaluate_rca_advanced.py --help
# Task 3 — visibility regression (paper protocol: 100 epochs, constant LR, 3 seeds)
python training/train_semu_regression.py --help
# zero-shot check on the public OLED weak-Mura set of Qiu et al. 2024 (download separately, CC BY 4.0)
python training/eval_real_weakmura.py --help
```

Analysis scripts of the paper (`analysis/`): background texture statistics and figure, rare-class
breakdowns, result collection; `results/` holds the collected numbers of every reported run.

## Results (test split, background-level protocol)

Task 1, mIoU over the five defect classes (per-class in the paper, Table 3):

| model | mIoU_def | spot | line | region | gradient | band |
|---|---|---|---|---|---|---|
| UPerNet Swin-S | **0.871** | 0.876 | 0.889 | **0.805** | **0.841** | 0.944 |
| UPerNet ConvNeXt-S | 0.865 | **0.886** | **0.890** | 0.780 | 0.819 | 0.950 |
| DeepLabV3+ ConvNeXt-S | 0.839 | 0.866 | 0.847 | 0.741 | 0.794 | 0.949 |
| SegFormer MiT-b2 | 0.807 | 0.853 | 0.842 | 0.687 | 0.704 | **0.951** |
| UPerNet HRNet-W32 | 0.646 | 0.872 | 0.889 | 0.423 | 0.245 | 0.804 |
| DeepLabV3+ ResNet-34 | 0.655 | 0.778 | 0.829 | 0.415 | 0.405 | 0.848 |

Task 2, ambiguous-cause configuration, ImageNet ConvNeXt-S crop encoder, mean of seeds 7/17/27:

| model | mean Hit@1 | step | tool | fault | time |
|---|---|---|---|---|---|
| image+position | 0.813 | 0.984 | 0.435 | 0.945 | 0.887 |
| late-fusion | 0.809 | 0.978 | 0.613 | 0.921 | 0.723 |
| cross-fusion | **0.855** | 0.976 | **0.726** | 0.929 | 0.790 |
| cross-fusion, route of another glass (control) | 0.774 | 0.971 | 0.398 | 0.927 | 0.799 |

Task 3, SEMU regression (100 epochs, constant LR, 3 seeds): image+position R² 0.64 ± 0.03 (MAE 0.30),
image-only 0.61 ± 0.02, image+position+route 0.63 ± 0.02.

Baseline checkpoints (weights only, 19 files, 2.9 GB): https://github.com/su9yeong/MuraTrace/releases (release v1.0; SHA256 list in `docs/checkpoints.md`).

## Contact

Contact: Sugyeong Lee, sugyeong@g.skku.edu or sugyeong.el.lee@gmail.com.

## Repository layout

```
generation/   dataset generation pipeline (background augmentation, defect compositing, threshold update, ambiguous labels)
configs/      generation config, SEMU thresholds (author + expert consensus)
training/     training and evaluation code of the three tasks, sim-to-real evaluation
analysis/     analysis scripts of the paper
tools/        make_release.py, verify_release.py, download.py, strip_checkpoints.py
results/      collected numbers of every reported run
sample_data/  a small generated sample for smoke tests
expert_study/ perceptual-study form and anonymized threshold responses
docs/         datasheet, schema, package README template
```

## Citation

```bibtex
@inproceedings{lee2027muratrace,
  title     = {MuraTrace: A Real-Background Synthetic Benchmark for Multi-Scale Display Mura Inspection},
  author    = {Lee, Sugyeong and Lee, Dong-Hee},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

Code: MIT. Data: CC BY 4.0. The weak-Mura images used for the zero-shot check belong to Qiu et al. 2024
(figshare 24532633, CC BY 4.0) and are not redistributed here.
