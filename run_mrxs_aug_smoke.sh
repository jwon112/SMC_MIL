#!/usr/bin/env bash
set -euo pipefail

cd /home/jupyter/image_team/projects/SMC_MIL

args=(
  --source mrxs
  --source-dataset mrxs13
  --dataset-root /home/jupyter/data/image_team/mrxs13_inbox
  --feature-manifest
  /home/jupyter/data/image_team/mrxs13_inbox/_clam/mrxs_feature_manifest_l0.csv
  --baseline-dir data/features/uni_v2/l0_0p25mpp_40x
  --stain-csv dataset_csv/stain_aug_labels.csv
  --task-csv cohorts/he_manual_acr_v1/cohort.csv
  --split-csv cohorts/he_manual_acr_v1/splits_0.csv
  --output-root data/features/uni_v2_aug/l0_0p25mpp_40x/fold_0
  --stain HE
  --label-source manual_only
  --policy geometry
  --num-views 1
  --allow-mrxs-baseline-h5-coord-mismatch
  --check-original-patches 16
  --smoke-patches 64
  --preview-count 8
  --batch-size 8
  --device cuda
  --amp
)

exec python augment_uni2_wsi.py "${args[@]}"
