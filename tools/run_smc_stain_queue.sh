#!/usr/bin/env bash
# Queue stain-restricted SMC CV after currently running grid workers finish.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FEATURE_ROOT="$ROOT_DIR/data/features/uni_v2"
STAIN_ROOT=""
ACR_GPU=""
AMR_GPU=""
MAX_EPOCHS=30
WAIT_PIDS=()
COHORTS=(mixed_known he_only non_he ihc_only)

usage() {
  cat <<'EOF'
Usage: bash tools/run_smc_stain_queue.sh --stain-root PATH --acr-gpu ID --amr-gpu ID [options]

Queues, in order, mixed_known, he_only, non_he, and ihc_only.  Each cohort
runs ACR 0R vs rest on the ACR GPU and AMR plus any-rejection on the AMR GPU.
High-grade ACR is intentionally excluded because the stain-restricted
validation folds have too few positive bags for a stable estimate.

Options:
  --wait-pid PID   Wait for a currently running worker PID; repeat as needed.
  --feature-root P Feature root (default: data/features/uni_v2).
  --max-epochs N   Maximum epochs per fold (default: 30).
  --cohorts NAMES  One or more cohorts; default: mixed_known he_only non_he ihc_only.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stain-root) STAIN_ROOT="$2"; shift 2 ;;
    --acr-gpu) ACR_GPU="$2"; shift 2 ;;
    --amr-gpu) AMR_GPU="$2"; shift 2 ;;
    --wait-pid) WAIT_PIDS+=("$2"); shift 2 ;;
    --feature-root) FEATURE_ROOT="$2"; shift 2 ;;
    --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
    --cohorts) shift; COHORTS=("$@") ; break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$STAIN_ROOT" && -n "$ACR_GPU" && -n "$AMR_GPU" ]] || { usage >&2; exit 2; }
[[ "$MAX_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "--max-epochs must be a positive integer" >&2; exit 2; }
for cohort in "${COHORTS[@]}"; do
  case "$cohort" in mixed_known|he_only|non_he|ihc_only) ;; *) echo "Unknown cohort: $cohort" >&2; exit 2 ;; esac
done

cd "$ROOT_DIR"
mkdir -p results/logs

for pid in "${WAIT_PIDS[@]}"; do
  while kill -0 "$pid" 2>/dev/null; do
    echo "[WAIT] worker PID $pid is still running"
    sleep 120
  done
  echo "[READY] worker PID $pid has exited"
done

for cohort in "${COHORTS[@]}"; do
  echo "[QUEUE] cohort=$cohort"
  bash tools/run_smc_cv_grid.sh \
    --gpu "$ACR_GPU" --worker acr_low \
    --feature-root "$FEATURE_ROOT" \
    --stain-root "$STAIN_ROOT" --stain-cohort "$cohort" \
    --max-epochs "$MAX_EPOCHS" \
    > "results/logs/grid_gpu${ACR_GPU}_stain_${cohort}.log" 2>&1 &
  acr_pid=$!

  bash tools/run_smc_cv_grid.sh \
    --gpu "$AMR_GPU" --worker amr \
    --feature-root "$FEATURE_ROOT" \
    --stain-root "$STAIN_ROOT" --stain-cohort "$cohort" \
    --max-epochs "$MAX_EPOCHS" \
    > "results/logs/grid_gpu${AMR_GPU}_stain_${cohort}.log" 2>&1 &
  amr_pid=$!

  wait "$acr_pid"
  wait "$amr_pid"
  echo "[DONE] cohort=$cohort"
done
