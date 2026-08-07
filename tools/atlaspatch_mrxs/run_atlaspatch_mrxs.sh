#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_ROOT:?Set DATASET_ROOT to the uploaded MRXS dataset root.}"
: "${ATLASPATCH_CHECKPOINT:?Set ATLASPATCH_CHECKPOINT to model.pth.}"
: "${ATLASPATCH_CONFIG:?Set ATLASPATCH_CONFIG to sam2.1_hiera_t.yaml.}"

python tools/atlaspatch_mrxs/segment_mrxs_atlaspatch.py \
  --dataset-root "$DATASET_ROOT" \
  --checkpoint "$ATLASPATCH_CHECKPOINT" \
  --config "$ATLASPATCH_CONFIG" \
  --device "${DEVICE:-cuda}" \
  --failure-log "${FAILURE_LOG:-$DATASET_ROOT/logs/atlaspatch_mrxs_failures.csv}" \
  "$@"
