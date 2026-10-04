#!/usr/bin/env bash
# One sequential process: prepare features, then all requested task/seed runs.
# Both UNI2 extraction and CLAM training use physical GPU 0 only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=0
MODE="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi
case "$MODE" in
  audit|prepare|train)
    python -u run_smc_gold_he_aug.py "$MODE" "$@" --gpu 0
    ;;
  all)
    python -u run_smc_gold_he_aug.py prepare "$@" --gpu 0
    python -u run_smc_gold_he_aug.py train "$@" --gpu 0
    ;;
  *)
    echo "Usage: bash run_smc_gold_he_aug_gpu0.sh [audit|prepare|train|all] [runner options]" >&2
    exit 2
    ;;
esac
