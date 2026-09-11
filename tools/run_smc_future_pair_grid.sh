#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_ROOT=""
FEATURE_ROOT="$ROOT_DIR/data/features/uni_v2"
GOLD_CSV="$ROOT_DIR/dataset_csv/smc_significant_rejection_binary.csv"
FOLDS=3
WORKER="all"
BOOTSTRAP=2000

usage() {
  echo "Usage: bash $0 --task-root PATH --gold-csv CSV [--feature-root PATH] [--folds 3] [--worker all|a|b] [--bootstrap 2000]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-root) TASK_ROOT="$2"; shift 2 ;;
    --feature-root) FEATURE_ROOT="$2"; shift 2 ;;
    --gold-csv) GOLD_CSV="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --worker) WORKER="$2"; shift 2 ;;
    --bootstrap) BOOTSTRAP="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$TASK_ROOT" ]] || { usage >&2; exit 2; }
[[ "$WORKER" == "all" || "$WORKER" == "a" || "$WORKER" == "b" ]] || {
  echo "--worker must be all, a, or b" >&2
  exit 2
}

PAIR_MANIFEST="$TASK_ROOT/future_pair_manifest.csv"
PATIENT_FOLDS="$TASK_ROOT/splits/future_pair_patient_folds_${FOLDS}.csv"
[[ -f "$PAIR_MANIFEST" ]] || { echo "Missing pair manifest: $PAIR_MANIFEST" >&2; exit 1; }
[[ -f "$PATIENT_FOLDS" ]] || { echo "Missing patient folds: $PATIENT_FOLDS" >&2; exit 1; }
[[ -f "$GOLD_CSV" ]] || { echo "Missing gold CSV: $GOLD_CSV" >&2; exit 1; }

ALL_SCALES=(
  "l0_0p25mpp_40x"
  "l1_0p50mpp_20x"
  "l2_1p00mpp_10x"
  "l3_2p00mpp_5x"
)
case "$WORKER" in
  all) SCALES=("${ALL_SCALES[@]}") ;;
  a) SCALES=("${ALL_SCALES[0]}" "${ALL_SCALES[2]}") ;;
  b) SCALES=("${ALL_SCALES[1]}" "${ALL_SCALES[3]}") ;;
esac

cd "$ROOT_DIR"
mkdir -p results/logs
for SCALE in "${SCALES[@]}"; do
  FEATURE_DIR="$FEATURE_ROOT/$SCALE"
  OUTPUT_DIR="results/smc_future_pair_${SCALE}_uni2_meanpool_gold${FOLDS}cv_s1"
  if [[ -f "$OUTPUT_DIR/pooled_metrics.csv" ]]; then
    echo "[SKIP] $OUTPUT_DIR already completed"
    continue
  fi
  echo "[RUN] $SCALE"
  python tools/run_smc_future_pair_baseline.py \
    --pair-manifest "$PAIR_MANIFEST" \
    --patient-folds "$PATIENT_FOLDS" \
    --gold-csv "$GOLD_CSV" \
    --feature-dir "$FEATURE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --folds "$FOLDS" \
    --bootstrap "$BOOTSTRAP"
  echo "[DONE] $SCALE"
done
