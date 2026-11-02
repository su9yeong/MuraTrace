# Baseline checkpoints (weights only)

Released as assets of GitHub release v1.0 of https://github.com/su9yeong/MuraTrace.
Load with the evaluation scripts in `training/` (segmentation: `evaluate_segmentation_checkpoint.py`,
attribution: `evaluate_rca_checkpoint.py`); the Task 3 regressors are not released as checkpoints
(their predictions are in `results/`).

| file | task | model (paper) | size (MB) | SHA256 |
|---|---|---|---:|---|
| `rca_image_only_s7.pt` | Task 2, ambiguous-cause | ConvNeXt-S, image-only, seed 7 | 200.6 | `4da7b40c2f8340a91ca8d9634992030c278f269a310cab75e1d963b89a92a95e` |
| `rca_image_position.pt` | Task 2, default configuration | small CNN, image+position (Table 4 top, mean Hit@1 0.941) | 15.8 | `23d0d5427bde899ab19f807d1b405f6e89ccb32c7ef03b1c1d62d74bc056d20e` |
| `rca_image_position_s7.pt` | Task 2, ambiguous-cause | ConvNeXt-S, image+position, seed 7 | 200.6 | `ec5ad38f33a856b6cd1afc180cde0a5d6d128bf1680fecf1001e90fed90cab1f` |
| `rca_image_route_s7.pt` | Task 2, ambiguous-cause | ConvNeXt-S, image+route, seed 7 | 200.6 | `b6298fd26b705737931ed4eb4a66b5a1d778d9253c5f9e68c4bd178f3a9c21e5` |
| `rca_late_fusion_s7.pt` | Task 2, ambiguous-cause | ConvNeXt-S, late-fusion, seed 7 | 200.6 | `dbdea585f18710925ea8a99376f8fdf51c47929675cf65e9bd31c457decd79da` |
| `rca_rcfilm_s7.pt` | Task 2, ambiguous-cause | ConvNeXt-S, FiLM-fusion, seed 7 | 200.6 | `c5e331a3265a8a93500d7be11a600d5f3c831fd6aaab200217097ca4701e8d16` |
| `rca_route_only_s7.pt` | Task 2, ambiguous-cause | ConvNeXt-S, route-only, seed 7 | 200.6 | `d0adf3238a1c727abe50a8d6d1ad7229aba8b194c4ba6b55380c1a743674cf02` |
| `rca_visionroute.pt` | Task 2, default configuration | small CNN, cross-fusion (0.900) | 15.8 | `ab9e8b78d7790f43a335160e5222b8173caeb2e90014b791314bbd8a06cbd1c3` |
| `rca_visionroute_s7.pt` | Task 2, ambiguous-cause | ConvNeXt-S, cross-fusion, seed 7 (Table 4 bottom: three-seed mean 0.855) | 200.6 | `b5baff1cf0536ef7607ec6371288d26fb112f77f5956d4e701051153ee4f83c7` |
| `rca_visionroute_swap_s7.pt` | Task 2, ambiguous-cause | ConvNeXt-S, cross-fusion with the route graph of another glass (swap control), seed 7 | 200.6 | `7a357c2516f1e2408b23d78fdbcf083ce56020e100fd7f423a45e75ece610185` |
| `seg_adv_deeplab_convnext_small.pt` | Task 1 | DeepLabV3+ ConvNeXt-S (0.839) | 194.5 | `9ff52fc39cb82f38f9ed4f704e44233c2d4f9e93619e56e8b1ed3d8e8a7152c7` |
| `seg_adv_deeplab_r34_mae.pt` | Task 1 | DeepLabV3+ ResNet-34, SimMIM init (0.633) | 85.8 | `062bff05a2c7badc21a58355c6ce60c8d4545a9593f77bd9b409643193ef9487` |
| `seg_adv_deeplab_r34_pretrained.pt` | Task 1 | DeepLabV3+ ResNet-34, ImageNet init (0.645) | 85.8 | `54e721bb8166562bf849578d77096195fe0c73ec3ef22d802508bd803a96f6b9` |
| `seg_adv_hrnet_w32.pt` | Task 1 | UPerNet HRNet-W32 (0.646) | 157.5 | `26d9f183d8b61aaafb8cf60339cbfd3dad10c12d94a3734e4dc7128560e44423` |
| `seg_adv_segformer_mit_b2.pt` | Task 1 | SegFormer MiT-b2 (0.807) | 94.4 | `d0d41c94d54a9032a196152b9f08295c620dc81bce06af470d81787a6040c0aa` |
| `seg_adv_swin_small.pt` | Task 1 | UPerNet Swin-S (Table 3, mIoU_def 0.871) | 221.7 | `588404258ff55e5e4fd8113f6807b68bb2f415f0ef132e3c5c0bc6e4ddad00fc` |
| `seg_adv_upernet_convnext_small.pt` | Task 1 | UPerNet ConvNeXt-S (0.865) | 224.0 | `7082d161bab8cdac1a18f9f2e6a3526d30866e353e881c68b2a8769ba32b8d94` |
| `seg_deeplabv3plus_auto_w4.pt` | Task 1 | DeepLabV3+ ResNet-34, class-weighted (0.655) | 85.7 | `f003c1883d56c3466c0d55e5837c7c4a7f52e383918247820b6c48c63eebcb62` |
| `seg_deeplabv3plus_nofocal_unweighted.pt` | Task 1 | DeepLabV3+ ResNet-34, unweighted (0.633) | 85.8 | `f30ba1c45b88b005c0c0707b73c516f604ee1aef12771fab31192e1f34623c4f` |
