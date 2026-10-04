#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
bash tools/run_smc_event_ablation_grid.sh \
  --worker acr \
  --feature-root /home/jupyter/image_team/projects/SMC_MIL/data/features/uni_v2 \
  --manifest-root /home/jupyter/data/image_team/labels/derived/event_stain_mil_gold_provisional_20260930 \
  --results-root results/smc_event_presence_control_20261004_exclude25 \
  --paired-fold-seeding \
  --modes aware aware_zero_mask \
  --scales 40x \
  --seeds 1 11 21 31 41 \
  "$@"
