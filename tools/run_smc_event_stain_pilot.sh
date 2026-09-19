#!/usr/bin/env bash
set -euo pipefail
GPU=""; WORKER="all"; FEATURE_ROOT="data/features/uni_v2"; MANIFEST_ROOT=""; RESULTS_ROOT="results/smc_event_stain_pilot"; SEED=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2;; --worker) WORKER="$2"; shift 2;; --feature-root) FEATURE_ROOT="$2"; shift 2;; --manifest-root) MANIFEST_ROOT="$2"; shift 2;; --results-root) RESULTS_ROOT="$2"; shift 2;; --seed) SEED="$2"; shift 2;; *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
[[ -n "$MANIFEST_ROOT" ]] || { echo "--manifest-root is required" >&2; exit 2; }; [[ -z "$GPU" ]] || export CUDA_VISIBLE_DEVICES="$GPU"
case "$WORKER" in acr) TASKS=(acr_high);; amr) TASKS=(amr_positive);; significant) TASKS=(significant_rejection);; all) TASKS=(acr_high amr_positive significant_rejection);; *) echo "worker: acr, amr, significant, all" >&2; exit 2;; esac
for TASK in "${TASKS[@]}"; do
  OUT="$RESULTS_ROOT/${TASK}_l0_0p25mpp_seed${SEED}"; echo "[RUN] task=$TASK seed=$SEED"
  python train_smc_event_stain_mil.py --event-csv "$MANIFEST_ROOT/$TASK/events.csv" --event-slides-csv "$MANIFEST_ROOT/$TASK/event_slides.csv" --split-dir "$MANIFEST_ROOT/$TASK/splits" --feature-dir "$FEATURE_ROOT/0p25mpp" --results-dir "$OUT" --folds 5 --seed "$SEED" --device cuda
done
