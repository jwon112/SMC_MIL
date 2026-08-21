#!/usr/bin/env bash
# Run one sequential half of the SMC nested-CV grid on one physical GPU.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FEATURE_ROOT="$ROOT_DIR/data/features/uni_v2"
GPU=""
WORKER=""

usage() {
  cat <<'EOF'
Usage: bash tools/run_smc_cv_grid.sh --gpu GPU_ID --worker acr|amr [--feature-root PATH]

Workers:
  acr  Runs the two ACR tasks at L0, L1, L2, and L3.
  amr  Runs the AMR and composite rejection tasks at L0, L1, L2, and L3.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --worker) WORKER="$2"; shift 2 ;;
    --feature-root) FEATURE_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$GPU" || ( "$WORKER" != "acr" && "$WORKER" != "amr" ) ]]; then
  usage >&2
  exit 2
fi

case "$WORKER" in
  acr)
    TASK_SPECS=(
      "task_smc_acr_binary_0r_vs_1r2r3r|smc_cv_acr_0r_vs_1r2r3r|acr_0r_vs_rest"
      "task_smc_acr_binary_0r1r_vs_2r3r|smc_cv_acr_0r1r_vs_2r3r|acr_high_grade"
    )
    ;;
  amr)
    TASK_SPECS=(
      "task_smc_amr_binary_pamr0_vs_positive|smc_cv_amr_pamr0_vs_positive|amr_positive"
      "task_smc_any_rejection_binary|smc_cv_any_rejection|any_rejection"
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
    [[ -d "splits/$split_dir" ]] || { echo "Missing nested CV splits: splits/$split_dir" >&2; exit 1; }

    exp_code="smc_${short_name}_${scale}_uni2_clamsb_nested3"
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
      --k 3 \
      --exp_code "$exp_code" \
      --model_type clam_sb \
      --model_size small \
      --max_epochs 200 \
      --drop_out 0.25 \
      --early_stopping \
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
