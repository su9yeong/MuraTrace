# Zenodo record — field values (copy and paste)

Reserved DOI: 10.5281/zenodo.23889760 (record 23889760)

| Field | Value |
|---|---|
| Resource type | Dataset |
| Title | MuraTrace: A Real-Background Synthetic Benchmark for Multi-Scale Display Mura Inspection (data package v1.0) |
| Publication date | date of publishing the record |
| Creators | Lee, Sugyeong — Sungkyunkwan University; Samsung Display Co., Ltd. (add ORCID if available) / Lee, Dong-Hee — Sungkyunkwan University |
| Description | see below |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Keywords | display inspection; Mura; OLED; synthetic benchmark; semantic segmentation; root-cause analysis; visibility prediction; SEMU; industrial anomaly detection |
| Language | English |
| Version | 1.0 |
| Related works | https://github.com/su9yeong/MuraTrace — relation "Is supplemented by" (Software); the paper DOI — relation "Is described by" (Publication), add when the proceedings DOI exists |
| Visibility | Public |
| Files | MuraTrace_v1.0_*.zip parts, PARTS.txt, ZIP_SHA256SUMS, README.md, DATASHEET.md, LICENSE |

## Description (HTML is allowed in the Zenodo editor)

<p><b>MuraTrace</b> is a real-background synthetic benchmark for multi-scale display Mura inspection
with paired perceptual and process-trace annotations (IEEE/CVF WACV 2027, Datasets track).
Physically motivated Mura defects (multiplicative luminance fields with SEMU-controlled visibility) are
composited onto 30 defect-free real OLED-mobile panel backgrounds. The package contains 540 synthetic
glasses (8&times;7 panels each), 30,240 panel images (16-bit PNG, 727&times;359), and 5,736 unique visible
defect instances in five classes (spot, line, region, gradient, band) and eight subtypes. Every instance
carries a dense semantic mask, an instance map, a bounding box, subtype, polarity, panel-instance SEMU
visibility, two reject labels (author thresholds and expert-consensus thresholds), and process-trace
metadata (process step, tool, fault mode, time window, route-graph node). Every glass carries a 21-node
route graph in two configurations (default and ambiguous-cause).</p>

<p>The package ships the background-level split used in the paper (18/6/6 real backgrounds for
train/val/test), a glass-disjoint split protocol, the generation configuration and SEMU thresholds, the
perceptual-study instrument with anonymized responses, and SHA256 checksums. These files are the exact
data on which all reported numbers were computed.</p>

<p>Code (generation, training, evaluation, analysis) and baseline checkpoints:
<a href="https://github.com/su9yeong/MuraTrace">https://github.com/su9yeong/MuraTrace</a>.
Unpack every zip part into the same folder and verify with
<code>python tools/verify_release.py --root &lt;folder&gt; --sha</code>.</p>

<p>Citation: S. Lee and D.-H. Lee, "MuraTrace: A Real-Background Synthetic Benchmark for Multi-Scale
Display Mura Inspection," IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2027.</p>

## Order of operations

1. Upload all files (zip parts first, then PARTS.txt, ZIP_SHA256SUMS, README.md, DATASHEET.md, LICENSE).
2. Fill the fields above; Save draft.
3. Re-check: the reserved DOI is shown, the file count matches PARTS.txt, the license is CC BY 4.0.
4. Publish. After publishing, files cannot be changed (only a new version can be added), so publish only after
   the zip verification has passed.
