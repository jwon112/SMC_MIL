#!/usr/bin/env bash
set -euo pipefail

GPU=""
WORKER="all"
FEATURE_ROOT="data/features/uni_v2"
MANIFEST_ROOT=""
RESULTS_ROOT="results/smc_event_stain_gold5x5"
FOLDS=5
MAX_EPOCHS=50
PATIENCE=10
MIN_EPOCHS=10
SEEDS=(1 11 21 31 41)
MODES=(aware aware_nomask agnostic)
SCALES=(40x 20x 10x 5x)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --worker) WORKER="$2"; shift 2 ;;
    --feature-root) FEATURE_ROOT="$2"; shift 2 ;;
    --manifest-root) MANIFEST_ROOT="$2"; shift 2 ;;
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --min-epochs) MIN_EPOCHS="$2"; shift 2 ;;
    --seeds)
      shift; SEEDS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done
      ;;
    --modes)
      shift; MODES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do MODES+=("$1"); shift; done
      ;;
    --scales)
      shift; SCALES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do SCALES+=("$1"); shift; done
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$MANIFEST_ROOT" ]] || { echo "--manifest-root is required" >&2; exit 2; }
[[ -z "$GPU" ]] || export CUDA_VISIBLE_DEVICES="$GPU"

case "$WORKER" in
  acr) TASKS=(acr_high) ;;
  amr) TASKS=(amr_positive) ;;
  significant) TASKS=(significant_rejection) ;;
  amr_significant) TASKS=(amr_positive significant_rejection) ;;
  all) TASKS=(acr_high amr_positive significant_rejection) ;;
  *) echo "--worker must be acr, amr, significant, amr_significant, or all" >&2; exit 2 ;;
esac

scale_dir() {
  case "$1" in
    40x) echo "l0_0p25mpp_40x" ;;
    20x) echo "l1_0p50mpp_20x" ;;
    10x) echo "l2_1p00mpp_10x" ;;
    5x) echo "l3_2p00mpp_5x" ;;
    *) echo "Unsupported scale: $1" >&2; return 2 ;;
  esac
}

for MODE in "${MODES[@]}"; do
  [[ "$MODE" == "aware" || "$MODE" == "aware_nomask" || "$MODE" == "agnostic" ]] || { echo "Unsupported mode: $MODE" >&2; exit 2; }
  for SCALE in "${SCALES[@]}"; do
    SCALE_DIR="$(scale_dir "$SCALE")"
    FEATURE_DIR="$FEATURE_ROOT/$SCALE_DIR"
    [[ -d "$FEATURE_DIR/pt_files" ]] || { echo "Missing feature directory: $FEATURE_DIR/pt_files" >&2; exit 1; }

    for TASK in "${TASKS[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        SUFFIX=""
        TRAIN_MODE="$MODE"
        MODE_ARGS=()
        if [[ "$MODE" == "aware_nomask" ]]; then
          SUFFIX="_aware_nomask"
          TRAIN_MODE="aware"
          MODE_ARGS=(--no-presence-mask)
        elif [[ "$MODE" == "agnostic" ]]; then
          SUFFIX="_agnostic"
        fi
        OUT="$RESULTS_ROOT/${TASK}_${SCALE_DIR}${SUFFIX}_seed${SEED}"
        CHECKPOINTS=0
        [[ ! -d "$OUT" ]] || CHECKPOINTS="$(find "$OUT" -maxdepth 1 -type f -name 's_*_checkpoint.pt' | wc -l)"
        if [[ -f "$OUT/oof_predictions.csv" && "$CHECKPOINTS" -eq "$FOLDS" ]]; then
          echo "[SKIP] complete task=$TASK scale=$SCALE mode=$MODE seed=$SEED"
          continue
        fi

        echo "[RUN] task=$TASK scale=$SCALE mode=$MODE seed=$SEED"
        python train_smc_event_stain_mil.py \
          --event-csv "$MANIFEST_ROOT/$TASK/events.csv" \
          --event-slides-csv "$MANIFEST_ROOT/$TASK/event_slides.csv" \
          --split-dir "$MANIFEST_ROOT/$TASK/splits/seed${SEED}" \
          --feature-dir "$FEATURE_DIR" \
          --results-dir "$OUT" \
          --folds "$FOLDS" \
          --seed "$SEED" \
          --stain-mode "$TRAIN_MODE" \
          "${MODE_ARGS[@]}" \
          --max-epochs "$MAX_EPOCHS" \
          --patience "$PATIENCE" \
          --min-epochs "$MIN_EPOCHS" \
          --device cuda
      done
    done
  done
done
