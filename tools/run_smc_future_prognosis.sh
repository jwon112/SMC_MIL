#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU=""
TASK_ROOT=""
FEATURE_ROOT="$ROOT_DIR/data/features/uni_v2"
MAX_EPOCHS=30
FOLDS=3
SCALE="l0_0p25mpp_40x"

usage() {
  echo "Usage: bash $0 --gpu ID --task-root PATH [--feature-root PATH] [--folds 3] [--max-epochs 30] [--scale all|SCALE]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --task-root) TASK_ROOT="$2"; shift 2 ;;
    --feature-root) FEATURE_ROOT="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
    --scale) SCALE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$GPU" && -n "$TASK_ROOT" ]] || { usage >&2; exit 2; }
[[ "$FOLDS" == 3 || "$FOLDS" == 4 || "$FOLDS" == 5 ]] || { echo "--folds must be 3, 4, or 5" >&2; exit 2; }
[[ "$MAX_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "--max-epochs must be positive" >&2; exit 2; }

CSV="$TASK_ROOT/dataset_csv/smc_future_significant_rejection.csv"
SPLITS="$TASK_ROOT/splits/smc_future_significant_rejection_gold${FOLDS}"
[[ -f "$CSV" ]] || { echo "Missing dataset CSV: $CSV" >&2; exit 1; }
[[ -d "$SPLITS" ]] || { echo "Missing splits: $SPLITS" >&2; exit 1; }

ALL_SCALES=(
  "l0_0p25mpp_40x"
  "l1_0p50mpp_20x"
  "l2_1p00mpp_10x"
  "l3_2p00mpp_5x"
)
if [[ "$SCALE" == "all" ]]; then
  SCALES=("${ALL_SCALES[@]}")
elif [[ " ${ALL_SCALES[*]} " == *" $SCALE "* ]]; then
  SCALES=("$SCALE")
else
  echo "Unknown --scale: $SCALE" >&2
  exit 2
fi

cd "$ROOT_DIR"
mkdir -p results/logs

for CURRENT_SCALE in "${SCALES[@]}"; do
  FEATURE_DIR="$FEATURE_ROOT/$CURRENT_SCALE"
  [[ -d "$FEATURE_DIR/pt_files" ]] || { echo "Missing features: $FEATURE_DIR/pt_files" >&2; exit 1; }
  EXP_CODE="smc_future_significant_${CURRENT_SCALE}_uni2_clamsb_gold${FOLDS}cv_fixed${MAX_EPOCHS}"
  RESULT_DIR="results/${EXP_CODE}_s1"
  if [[ -f "$RESULT_DIR/summary.csv" ]]; then
    echo "[SKIP] $RESULT_DIR already has summary.csv"
    continue
  fi
  echo "[RUN] $CURRENT_SCALE on GPU $GPU"

  CUDA_VISIBLE_DEVICES="$GPU" python main.py \
    --data_root_dir "$FEATURE_DIR" \
    --task task_smc_future_significant_rejection \
    --csv_path "$CSV" \
    --split_dir "$SPLITS" \
    --k "$FOLDS" \
    --exp_code "$EXP_CODE" \
    --model_type clam_sb \
    --model_size small \
    --max_epochs "$MAX_EPOCHS" \
    --no_val \
    --lr-scheduler cosine \
    --min-lr 0 \
    --drop_out 0.25 \
    --lr 2e-4 \
    --reg 1e-5 \
    --bag_loss ce \
    --inst_loss svm \
    --bag_weight 0.7 \
    --B 8 \
    --weighted_sample \
    --embed_dim 1536 \
    > "results/logs/${EXP_CODE}.log" 2>&1

  echo "[DONE] $RESULT_DIR"
done
