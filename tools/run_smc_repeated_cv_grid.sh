#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU=""
WORKER=""
FEATURE_ROOT="$ROOT_DIR/data/features/uni_v2"
SEEDS="11,21,31,41"
FOLDS=5
MAX_EPOCHS=200
EARLY_STOP_PATIENCE=20
EARLY_STOP_MIN_EPOCH=50

usage() {
  echo "Usage: bash $0 --gpu GPU_ID --worker a|b [--seeds 11,21,31,41] [--feature-root PATH] [--folds 5] [--max-epochs 200] [--early-stop-patience 20] [--early-stop-min-epoch 50]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --worker) WORKER="$2"; shift 2 ;;
    --feature-root) FEATURE_ROOT="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
    --early-stop-patience) EARLY_STOP_PATIENCE="$2"; shift 2 ;;
    --early-stop-min-epoch) EARLY_STOP_MIN_EPOCH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$GPU" ]] || { usage >&2; exit 2; }
[[ "$WORKER" == "a" || "$WORKER" == "b" ]] || { usage >&2; exit 2; }

IFS=',' read -r -a SEED_VALUES <<< "$SEEDS"
cd "$ROOT_DIR"

run_grid() {
  local seed="$1"
  local task_worker="$2"
  local scale_worker="$3"
  bash tools/run_smc_cv_grid.sh \
    --gpu "$GPU" \
    --worker "$task_worker" \
    --scale-worker "$scale_worker" \
    --feature-root "$FEATURE_ROOT" \
    --folds "$FOLDS" \
    --seed "$seed" \
    --max-epochs "$MAX_EPOCHS" \
    --early-stop-patience "$EARLY_STOP_PATIENCE" \
    --early-stop-min-epoch "$EARLY_STOP_MIN_EPOCH"
}

for SEED in "${SEED_VALUES[@]}"; do
  echo "[REPEAT] seed=$SEED worker=$WORKER"
  if [[ "$WORKER" == "a" ]]; then
    run_grid "$SEED" acr_high 40x
    run_grid "$SEED" amr 40x
  else
    run_grid "$SEED" significant 40x
    run_grid "$SEED" primary 20x
    run_grid "$SEED" primary 10x
    run_grid "$SEED" primary 5x
  fi
done

echo "[DONE] repeated CV seeds=$SEEDS worker=$WORKER"
