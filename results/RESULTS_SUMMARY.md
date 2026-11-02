# Camera-ready results summary (auto-collected)

## Segmentation: per test background (mIoU_def over classes present) and strict/repeat subsets

| model | all | bg07 | bg14 | bg15 | bg18 | bg23 | bg25 | strict | repeat | spread (min-max) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| deeplab_convnext_small | 0.837 | 0.866 | 0.810 | 0.850 | 0.834 | 0.877 | 0.809 | 0.821 | 0.855 | 0.809-0.877 |
|  ↳ LaTeX | 0.866 & 0.810 & 0.850 & 0.834 & 0.877 & 0.809 & 0.821 & 0.855 \\ | | | | | | | |
| deeplab_r34_mae | 0.632 | 0.617 | 0.639 | 0.618 | 0.641 | 0.676 | 0.605 | 0.620 | 0.649 | 0.605-0.676 |
|  ↳ LaTeX | 0.617 & 0.639 & 0.618 & 0.641 & 0.676 & 0.605 & 0.620 & 0.649 \\ | | | | | | | |
| deeplab_r34_pretrained | 0.643 | 0.625 | 0.602 | 0.681 | 0.683 | 0.664 | 0.609 | 0.627 | 0.661 | 0.602-0.683 |
|  ↳ LaTeX | 0.625 & 0.602 & 0.681 & 0.683 & 0.664 & 0.609 & 0.627 & 0.661 \\ | | | | | | | |
| hrnet_w32 | 0.662 | 0.672 | 0.627 | 0.669 | 0.674 | 0.715 | 0.629 | 0.639 | 0.689 | 0.627-0.715 |
|  ↳ LaTeX | 0.672 & 0.627 & 0.669 & 0.674 & 0.715 & 0.629 & 0.639 & 0.689 \\ | | | | | | | |
| segformer_mit_b2 | 0.808 | 0.806 | 0.760 | 0.828 | 0.853 | 0.846 | 0.777 | 0.784 | 0.836 | 0.760-0.853 |
|  ↳ LaTeX | 0.806 & 0.760 & 0.828 & 0.853 & 0.846 & 0.777 & 0.784 & 0.836 \\ | | | | | | | |
| swin_small | 0.868 | 0.890 | 0.905 | 0.824 | 0.872 | 0.907 | 0.822 | 0.852 | 0.887 | 0.822-0.907 |
|  ↳ LaTeX | 0.890 & 0.905 & 0.824 & 0.872 & 0.907 & 0.822 & 0.852 & 0.887 \\ | | | | | | | |
| upernet_convnext_small | 0.865 | 0.885 | 0.872 | 0.825 | 0.876 | 0.888 | 0.850 | 0.851 | 0.879 | 0.825-0.888 |
|  ↳ LaTeX | 0.885 & 0.872 & 0.825 & 0.876 & 0.888 & 0.850 & 0.851 & 0.879 \\ | | | | | | | |

## Background-count ablation (DeepLabV3+ ConvNeXt-S, 24k steps) test mIoU_def

| run | mIoU_def | spot | line | region | gradient | band | best step |
|---|---:|---:|---:|---:|---:|---:|---:|
| k03_bg3 | 0.722 | 0.846 | 0.673 | 0.545 | 0.609 | 0.935 | 23488 |
| k03_match18 | 0.726 | 0.835 | 0.825 | 0.522 | 0.517 | 0.931 | 24000 |
| k06_bg6 | 0.731 | 0.847 | 0.817 | 0.503 | 0.552 | 0.938 | 14770 |
| k06_match18 | 0.764 | 0.816 | 0.807 | 0.627 | 0.648 | 0.922 | 14770 |
| k12_bg12 | 0.789 | 0.861 | 0.814 | 0.653 | 0.674 | 0.943 | 21175 |
| k12_match18 | 0.779 | 0.852 | 0.821 | 0.688 | 0.599 | 0.937 | 21175 |
| k18_full | 0.787 | 0.849 | 0.826 | 0.651 | 0.675 | 0.934 | 18200 |

LaTeX rows for tab:bgabl:
    $k$ backgrounds & 0.722 & 0.731 & 0.789 & 0.787 \\
    18 backgrounds, same panels & 0.726 & 0.764 & 0.779 & -- \\

## Process-trace, ambiguous-cause configuration (ambiguous-cause; ConvNeXt crop encoder, batch 32)

| run | seed | epoch | mean@1 | mean@3 | step | tool | fault | time | tool@3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| image_only | 7 | 4 | 0.415 | 0.770 | 0.451 | 0.210 | 0.461 | 0.539 | 0.558 |
| image_position | 17 | 38 | 0.813 | 0.995 | 0.990 | 0.423 | 0.944 | 0.892 | 0.982 |
| image_position | 27 | 41 | 0.815 | 0.995 | 0.986 | 0.432 | 0.952 | 0.889 | 0.981 |
| image_position | 7 | 41 | 0.810 | 0.994 | 0.976 | 0.448 | 0.937 | 0.880 | 0.977 |
| image_route | 7 | 44 | 0.415 | 0.744 | 0.392 | 0.274 | 0.545 | 0.450 | 0.552 |
| late_fusion | 17 | 42 | 0.815 | 0.972 | 0.982 | 0.609 | 0.926 | 0.742 | 0.978 |
| late_fusion | 27 | 42 | 0.807 | 0.966 | 0.977 | 0.609 | 0.914 | 0.727 | 0.952 |
| late_fusion | 7 | 41 | 0.805 | 0.973 | 0.974 | 0.622 | 0.924 | 0.699 | 0.961 |
| rcfilm | 7 | 30 | 0.479 | 0.809 | 0.500 | 0.300 | 0.669 | 0.446 | 0.654 |
| route_only | 7 | 45 | 0.399 | 0.763 | 0.397 | 0.283 | 0.509 | 0.406 | 0.569 |
| visionroute | 17 | 48 | 0.863 | 0.993 | 0.980 | 0.743 | 0.932 | 0.797 | 0.986 |
| visionroute | 27 | 39 | 0.850 | 0.996 | 0.978 | 0.714 | 0.930 | 0.778 | 0.989 |
| visionroute | 7 | 30 | 0.853 | 0.995 | 0.970 | 0.722 | 0.925 | 0.795 | 0.987 |
| visionroute_swap | 17 | 25 | 0.766 | 0.989 | 0.978 | 0.389 | 0.928 | 0.767 | 0.962 |
| visionroute_swap | 27 | 22 | 0.787 | 0.993 | 0.976 | 0.413 | 0.930 | 0.828 | 0.973 |
| visionroute_swap | 7 | 14 | 0.770 | 0.988 | 0.960 | 0.392 | 0.924 | 0.803 | 0.961 |

LaTeX rows (seed 7 value; mean±std over seeds where >1):
    image_only (1 seeds): & 0.415 & 0.770 & 0.451 & 0.210 & 0.461 & 0.539 \\
    image_position (3 seeds): & 0.813$\pm$0.002 & 0.995$\pm$0.000 & 0.984$\pm$0.006 & 0.435$\pm$0.010 & 0.945$\pm$0.006 & 0.887$\pm$0.005 \\
    image_route (1 seeds): & 0.415 & 0.744 & 0.392 & 0.274 & 0.545 & 0.450 \\
    late_fusion (3 seeds): & 0.809$\pm$0.004 & 0.970$\pm$0.003 & 0.978$\pm$0.003 & 0.613$\pm$0.006 & 0.921$\pm$0.005 & 0.723$\pm$0.018 \\
    rcfilm (1 seeds): & 0.479 & 0.809 & 0.500 & 0.300 & 0.669 & 0.446 \\
    route_only (1 seeds): & 0.399 & 0.763 & 0.397 & 0.283 & 0.509 & 0.406 \\
    visionroute (3 seeds): & 0.855$\pm$0.006 & 0.995$\pm$0.001 & 0.976$\pm$0.004 & 0.726$\pm$0.012 & 0.929$\pm$0.003 & 0.790$\pm$0.008 \\
    visionroute_swap (3 seeds): & 0.774$\pm$0.009 & 0.990$\pm$0.002 & 0.971$\pm$0.008 & 0.398$\pm$0.010 & 0.927$\pm$0.003 & 0.799$\pm$0.025 \\

## Process-trace, default configuration controls (small CNN, batch 64)

- rca_visionroute_s7: mean@1 0.902 mean@3 0.990 step 0.977 tool 0.931 fault 0.935 time 0.765 (epoch 48)
- rca_visionroute_swap_s7: mean@1 0.898 mean@3 0.995 step 0.974 tool 0.927 fault 0.925 time 0.764 (epoch 40)

## Visibility regression (camera-ready protocol: 100 ep, const LR; pooled over runs_semu_reg_cr100*; mean ± population std over seeds)

| variant | seeds | R2 | MAE |
|---|---:|---:|---:|
| image_only | 3 | 0.607 ± 0.017 | 0.303 ± 0.003 |
| image_position | 3 | 0.637 ± 0.027 | 0.301 ± 0.008 |
| image_position_route | 3 | 0.626 ± 0.023 | 0.310 ± 0.007 |
| image_route | 3 | 0.515 ± 0.046 | 0.323 ± 0.007 |

LaTeX rows: image_only: $0.607\pm0.017$ & $0.303\pm0.003$ \\ image_position: $0.637\pm0.027$ & $0.301\pm0.008$ \\ image_position_route: $0.626\pm0.023$ & $0.310\pm0.007$ \\ image_route: $0.515\pm0.046$ & $0.323\pm0.007$ \\

per run: image_only s7: R2 0.595 (val 0.568); image_only s17: R2 0.631 (val 0.591); image_only s27: R2 0.596 (val 0.579); image_position s7: R2 0.675 (val 0.608); image_position s17: R2 0.614 (val 0.599); image_position s27: R2 0.621 (val 0.622); image_position_route s7: R2 0.648 (val 0.641); image_position_route s17: R2 0.638 (val 0.635); image_position_route s27: R2 0.594 (val 0.614); image_route s17: R2 0.521 (val 0.574); image_route s27: R2 0.568 (val 0.586); image_route s7: R2 0.456 (val 0.578)

## Zero-shot real-data check (Qiu et al. weak-Mura)

- sim2real_convnext_fit: tiles 10020 boxes 5790 recall(any) 0.642 chance(any) 0.152 recall(spot) 0.532 chance(spot) 0.001 FA/def-tile 3.72 clean-alarm 0.973; scale=0.863 denoise=0.0
- sim2real_convnext_fit_dn15: tiles 10020 boxes 5790 recall(any) 0.958 chance(any) 0.377 recall(spot) 0.899 chance(spot) 0.061 FA/def-tile 12.54 clean-alarm 0.993; scale=0.863 denoise=1.5
- sim2real_swin_fit: tiles 10020 boxes 5790 recall(any) 0.882 chance(any) 0.224 recall(spot) 0.827 chance(spot) 0.004 FA/def-tile 3.32 clean-alarm 0.914; scale=0.863 denoise=0.0
- sim2real_swin_fit_dn15: tiles 10020 boxes 5790 recall(any) 0.941 chance(any) 0.232 recall(spot) 0.920 chance(spot) 0.028 FA/def-tile 6.78 clean-alarm 0.991; scale=0.863 denoise=1.5
- sim2real_swin_s050: tiles 10020 boxes 5790 recall(any) 0.731 chance(any) 0.268 recall(spot) 0.571 chance(spot) 0.001 FA/def-tile 2.24 clean-alarm 0.924; scale=0.5 denoise=0.0
- sim2real_swin_s100: tiles 10020 boxes 5790 recall(any) 0.841 chance(any) 0.187 recall(spot) 0.802 chance(spot) 0.003 FA/def-tile 3.37 clean-alarm 0.830; scale=1.0 denoise=0.0
