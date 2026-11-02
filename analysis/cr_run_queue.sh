#!/usr/bin/env bash
# ============================================================================
#  MuraTrace camera-ready experiment queue (single GPU, sequential, resumable)
#
#  Everything writes to NEW directories (runs_cr_*, runs_semu_reg_cr,
#  analysis_camera_ready/*); the WACV submission runs are never touched.
#  Re-running the script skips stages whose test outputs already exist.
#
#  Stages (in order of value):
#   1. Task-3 visibility regression: +route-graph variant, 3 seeds, per-class breakdown
#   2. per-held-out-background breakdown of the top segmentation models (unseen textures)
#   3. ambiguous-cause process-trace block re-run at one consistent protocol
#      (ConvNeXt crop encoder, batch 32 to fit the 8 GB GPU, seed 7) incl. the
#      proper route-SWAP control
#   4. background-count ablation (3/6/12/18 train backgrounds, fixed 24k-step budget,
#      panel-count-matched controls), DeepLabV3+ ConvNeXt-S
#   5. seeds 17/27 for the key ambiguous rows (error bars)
#   6. default-configuration route-swap control pair (smallcnn encoder as in the paper)
#
#  Usage:  bash scripts/cr_run_queue.sh > logs/cr_queue.log 2>&1 &
# ============================================================================
set -u
cd /c/dmb/MuraTrace/dataset_gen || exit 1
PY="${PY:-python}"
DATA_DEF=data/mura_rca_wacv_realbg_visible
DATA_AMB=data/mura_rca_wacv_ambiguous
ANA=analysis_camera_ready
mkdir -p logs runs_cr_ambiguous runs_cr_bgablation runs_cr_default "$ANA"

stamp() { date +%Y-%m-%d_%H:%M:%S; }
run() { echo; echo "[$(stamp)] RUN: $*"; "$@"; local rc=$?; echo "[$(stamp)] exit=$rc"; return $rc; }

# ---------------------------------------------------------------- stage 0: zero-shot real-data check (Qiu et al. 2024 weak-Mura tiles)
S2R_DATA=external/qiu2024_weakmura/extracted/datasets
s2r() {  # name ckpt [extra args]
  local name=$1 ckpt=$2; shift 2
  local out=$ANA/sim2real_${name}.json
  if [ -f "$out" ]; then echo "[skip] sim2real $name"; return 0; fi
  [ -d "$S2R_DATA" ] || { echo "[skip] sim2real $name: data missing"; return 0; }
  run $PY training/eval_real_weakmura.py --data_root $S2R_DATA --checkpoint "$ckpt" --out "$out" \
      --examples_dir "$ANA/sim2real_examples_${name}" --n_examples 24 --group 16 --batch 16 "$@"
}
SWIN=runs_wacv_realbg_visible/seg_adv_swin_small/best.pt
CNX=runs_wacv_realbg_visible/seg_adv_upernet_convnext_small/best.pt
s2r swin_fit         $SWIN --scale 0.863
s2r convnext_fit     $CNX  --scale 0.863
s2r swin_fit_dn15    $SWIN --scale 0.863 --denoise_sigma 1.5
s2r convnext_fit_dn15 $CNX --scale 0.863 --denoise_sigma 1.5
s2r swin_s050        $SWIN --scale 0.5
s2r swin_s100        $SWIN --scale 1.0

# ---------------------------------------------------------------- stage 1 (moved to the end: see stage 7)

# ---------------------------------------------------------------- stage 2
per_source() {  # name ckpt
  local name=$1 ckpt=$2
  if [ -f "$ANA/seg_per_source_${name}.json" ]; then echo "[skip] per-source $name"; return 0; fi
  run $PY training/eval_seg_per_source.py --checkpoint "$ckpt" --data_root $DATA_DEF --split test \
      --out "$ANA/seg_per_source_${name}.json" --batch_size 8
}
per_source swin_small            runs_wacv_realbg_visible/seg_adv_swin_small/best.pt
per_source upernet_convnext_small runs_wacv_realbg_visible/seg_adv_upernet_convnext_small/best.pt
per_source deeplab_convnext_small runs_wacv_realbg_visible/seg_adv_deeplab_convnext_small/best.pt

# ---------------------------------------------------------------- stage 3 / 5
rca_amb() {  # name mode seed [extra args]
  local name=$1 mode=$2 seed=$3; shift 3
  local out=runs_cr_ambiguous/rca_${name}_s${seed}
  local ev=$ANA/rca_ambiguous/rca_${name}_s${seed}
  if [ -f "$ev/test_summary.json" ]; then echo "[skip] ambiguous $name s$seed"; return 0; fi
  SEED=$seed run $PY training/run_seeded.py train_rca_advanced --data_root $DATA_AMB --out_dir "$out" \
      --mode "$mode" --task joint --image_encoder convnext --image_pretrained \
      --epochs 50 --batch_size 32 --image_size 224 --amp "$@" \
  && run $PY training/evaluate_rca_advanced.py --checkpoint "$out/best.pt" --data_root $DATA_AMB --split test --out_dir "$ev" --amp
}
rca_amb image_position      image_position 7
rca_amb visionroute         visionroute    7
rca_amb visionroute_swap    visionroute    7 --route_swap
rca_amb late_fusion         late_fusion    7
rca_amb image_only          image_only     7
rca_amb route_only          route_only     7
rca_amb image_route         image_route    7
rca_amb rcfilm              rcfilm         7

# ---------------------------------------------------------------- stage 4
seg_abl() {  # name [extra args]
  local name=$1; shift
  local out=runs_cr_bgablation/$name
  local ev=$ANA/bg_ablation/$name
  if [ -f "$ev/test_summary.json" ]; then echo "[skip] bg-ablation $name"; return 0; fi
  run $PY training/train_segmentation_advanced.py --data_root $DATA_DEF --out_dir "$out" \
      --arch deeplabv3plus --encoder_name tu-convnext_small --pretrained --in_channels 3 \
      --epochs 50 --batch_size 4 --image_size 512 --lr 2e-4 --class_weights auto --class_weight_cap 4 --amp \
      --seed 7 --max_steps 24000 --val_every_steps 3000 "$@" \
  && run $PY training/evaluate_segmentation_advanced.py --checkpoint "$out/best.pt" --data_root $DATA_DEF \
      --split test --mask_key train_mask --out_dir "$ev" --amp
}
# nested random subsets of the 18 training backgrounds (python random.Random(0).shuffle):
#   k=3  : bg00,bg17,bg20                                  (2936 panels)
#   k=6  : + bg03,bg27,bg28                                (5911 panels)
#   k=12 : + bg06,bg10,bg11,bg19,bg26,bg29                 (12102 panels)
#   k=18 : all                                             (18200 panels)
# matched controls: all 18 backgrounds, randomly subsampled to the same panel count.
seg_abl k03_bg3       --train_sources bg00,bg17,bg20
seg_abl k03_match18   --train_subsample 2936 --subsample_seed 0
seg_abl k06_bg6       --train_sources bg00,bg03,bg17,bg20,bg27,bg28
seg_abl k06_match18   --train_subsample 5911 --subsample_seed 0
seg_abl k18_full
seg_abl k12_bg12      --train_sources bg00,bg03,bg06,bg10,bg11,bg17,bg19,bg20,bg26,bg27,bg28,bg29
seg_abl k12_match18   --train_subsample 12102 --subsample_seed 0

# ---------------------------------------------------------------- stage 7: Task-3 visibility regression
# (last, because the reproduction of the submitted protocol on this GPU is still being checked)
if [ ! -f runs_semu_reg_cr/semu_regression_result.json ]; then
  run $PY training/train_semu_regression.py --seeds 7,17,27 \
      --variants image_only,image_position,image_position_route,image_route \
      --out runs_semu_reg_cr --fig_dir tex_camera_ready/figures
  cp -f runs_semu_reg_cr/semu_breakdown.md "$ANA/semu_breakdown_cr.md" 2>/dev/null
  cp -f runs_semu_reg_cr/semu_breakdown.csv "$ANA/semu_breakdown_cr.csv" 2>/dev/null
  cp -f runs_semu_reg_cr/semu_regression_result.json "$ANA/semu_regression_result_cr.json" 2>/dev/null
else
  echo "[skip] stage 7 (semu regression) done"
fi

# ---------------------------------------------------------------- stage 6
rca_def() {  # name mode seed [extra]
  local name=$1 mode=$2 seed=$3; shift 3
  local out=runs_cr_default/rca_${name}_s${seed}
  local ev=$ANA/rca_default/rca_${name}_s${seed}
  if [ -f "$ev/test_summary.json" ]; then echo "[skip] default $name s$seed"; return 0; fi
  SEED=$seed run $PY training/run_seeded.py train_rca_advanced --data_root $DATA_DEF --out_dir "$out" \
      --mode "$mode" --task joint --image_encoder smallcnn --no_image_pretrained \
      --epochs 50 --batch_size 64 --image_size 224 --amp "$@" \
  && run $PY training/evaluate_rca_advanced.py --checkpoint "$out/best.pt" --data_root $DATA_DEF --split test --out_dir "$ev" --amp
}
rca_def visionroute      visionroute 7
rca_def visionroute_swap visionroute 7 --route_swap

# ---------------------------------------------------------------- stage 5
for s in 17 27; do
  rca_amb image_position   image_position $s
  rca_amb visionroute      visionroute    $s
  rca_amb visionroute_swap visionroute    $s --route_swap
  rca_amb late_fusion      late_fusion    $s
done

# ---------------------------------------------------------------- stage 8: per-background breakdown of the remaining seg models
per_source segformer_mit_b2        runs_wacv_realbg_visible/seg_adv_segformer_mit_b2/best.pt
per_source hrnet_w32               runs_wacv_realbg_visible/seg_adv_hrnet_w32/best.pt
per_source deeplab_r34_pretrained  runs_wacv_realbg_visible/seg_adv_deeplab_r34_pretrained/best.pt
per_source deeplab_r34_mae         runs_wacv_realbg_visible/seg_adv_deeplab_r34_mae/best.pt
per_source deeplabv3plus_auto_w4   runs_wacv_realbg_visible/seg_deeplabv3plus_auto_w4/best.pt
per_source deeplabv3plus_unweighted runs_wacv_realbg_visible/seg_deeplabv3plus_nofocal_unweighted/best.pt

# ---------------------------------------------------------------- stage 9: A/B check of the submitted Task-3 script (slow: spills GPU memory)
if ! grep -q "TEST R2" runs_semu_reg_cr/ab_orig_image_only.log 2>/dev/null; then
  mkdir -p runs_semu_reg_cr
  run $PY training/train_semu_regression_orig.py --variants image_only,image_position > runs_semu_reg_cr/ab_orig_image_only.log 2>&1
  grep "TEST R2" runs_semu_reg_cr/ab_orig_image_only.log
fi

echo
echo "[$(stamp)] QUEUE COMPLETE"
