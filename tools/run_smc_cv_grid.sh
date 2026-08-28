#!/usr/bin/env bash
# Run one sequential half of the SMC patient-grouped 3-fold CV grid on one physical GPU.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FEATURE_ROOT="$ROOT_DIR/data/features/uni_v2"
GPU=""
WORKER=""
FULL_TRAIN_CV=false
MAX_EPOCHS=200
WEAK_TRAIN_ROOT=""

usage() {
  cat <<'EOF'
Usage: bash tools/run_smc_cv_grid.sh --gpu GPU_ID --worker acr|amr [--feature-root PATH] [--weak-train-root PATH] [--full-train-cv --max-epochs N]

Workers:
  acr  Runs the two ACR tasks at L0, L1, L2, and L3.
  amr  Runs the AMR and composite rejection tasks at L0, L1, L2, and L3.

Modes:
  --full-train-cv  Train on all two outer-training folds without validation or
                   early stopping. Uses cosine LR decay and *_fulltrain splits.
  --max-epochs N   Training epochs (default: 200; use 100 with --full-train-cv).
  --weak-train-root PATH
                   Root created by build_smc_weak_unique_training.py. Keeps each
                   standard CV validation fold unchanged and augments train only.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --worker) WORKER="$2"; shift 2 ;;
    --feature-root) FEATURE_ROOT="$2"; shift 2 ;;
    --weak-train-root) WEAK_TRAIN_ROOT="$2"; shift 2 ;;
    --full-train-cv) FULL_TRAIN_CV=true; shift ;;
    --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$GPU" || ( "$WORKER" != "acr" && "$WORKER" != "amr" ) ]]; then
  usage >&2
  exit 2
fi

[[ "$MAX_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "--max-epochs must be a positive integer" >&2; exit 2; }

case "$WORKER" in
  acr)
    TASK_SPECS=(
      "task_smc_acr_binary_0r_vs_1r2r3r|smc_cv_acr_0r_vs_1r2r3r_standard3|acr_0r_vs_rest"
      "task_smc_acr_binary_0r1r_vs_2r3r|smc_cv_acr_0r1r_vs_2r3r_standard3|acr_high_grade"
    )
    ;;
  amr)
    TASK_SPECS=(
      "task_smc_amr_binary_pamr0_vs_positive|smc_cv_amr_pamr0_vs_positive_standard3|amr_positive"
      "task_smc_any_rejection_binary|smc_cv_any_rejection_standard3|any_rejection"
    )
    ;;
esac

SCALES=(
  "l0_0p25mpp_40x"
  "l1_0p50mpp_20x"
  "l2_1p00mpp_10x"
  "l3_2p00mpp_5x"
)

cd "$ROOT_DIR"
mkdir -p results/logs

for scale in "${SCALES[@]}"; do
  feature_dir="$FEATURE_ROOT/$scale"
  [[ -d "$feature_dir/pt_files" ]] || { echo "Missing feature directory: $feature_dir/pt_files" >&2; exit 1; }

  for spec in "${TASK_SPECS[@]}"; do
    IFS='|' read -r task split_dir short_name <<< "$spec"
    mode_args=(--early_stopping --cv-validation)
    exp_mode="_cv3val"
    if [[ "$FULL_TRAIN_CV" == true ]]; then
      split_dir="${split_dir%_standard3}_fulltrain"
      mode_args=(--no_val --lr-scheduler cosine --min-lr 0)
      exp_mode="_fulltrain${MAX_EPOCHS}cosine"
    fi

    csv_args=()
    if [[ -n "$WEAK_TRAIN_ROOT" ]]; then
      [[ "$FULL_TRAIN_CV" == false ]] || { echo "--weak-train-root cannot be combined with --full-train-cv" >&2; exit 2; }
      split_dir="${split_dir}_weak_unique_0to3"
      weak_csv="$WEAK_TRAIN_ROOT/dataset_csv/${task}_weak_unique_0to3.csv"
      [[ -f "$weak_csv" ]] || { echo "Missing weak-label CSV: $weak_csv" >&2; exit 1; }
      csv_args=(--csv_path "$weak_csv")
      exp_mode="${exp_mode}_weakunique3"
    fi
    if [[ -n "$WEAK_TRAIN_ROOT" ]]; then
      split_path="$WEAK_TRAIN_ROOT/splits/$split_dir"
      split_dir="$split_path"
    else
      split_path="splits/$split_dir"
    fi
    [[ -d "$split_path" ]] || { echo "Missing CV splits: $split_path" >&2; exit 1; }

    exp_code="smc_${short_name}_${scale}_uni2_clamsb${exp_mode}"
    log_path="results/logs/${exp_code}.log"
    result_dir="results/${exp_code}_s1"
    if [[ -f "$result_dir/summary.csv" ]]; then
      echo "[SKIP] $exp_code already has summary.csv"
      continue
    fi
    echo "[RUN] GPU $GPU | $task | $scale"

    CUDA_VISIBLE_DEVICES="$GPU" python main.py \
      --data_root_dir "$feature_dir" \
      --task "$task" \
      --split_dir "$split_dir" \
      "${csv_args[@]}" \
      --k 3 \
      --exp_code "$exp_code" \
      --model_type clam_sb \
      --model_size small \
      --max_epochs "$MAX_EPOCHS" \
      --drop_out 0.25 \
      "${mode_args[@]}" \
      --lr 2e-4 \
      --reg 1e-5 \
      --bag_loss ce \
      --inst_loss svm \
      --bag_weight 0.7 \
      --B 8 \
      --weighted_sample \
      --embed_dim 1536 \
      > "$log_path" 2>&1

    echo "[DONE] $exp_code"
  done
done
