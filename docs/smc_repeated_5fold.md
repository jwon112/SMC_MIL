# Repeated 5-fold gold-only CV

This experiment repeats the patient-grouped 5-fold baseline for the three
primary tasks:

- high-grade ACR (`ACR >= 2R`)
- AMR positive (`pAMR >= 1`)
- significant rejection (`ACR >= 2R OR pAMR >= 1`)

Seed 1 is the completed baseline. Seeds 11, 21, 31, and 41 use both a new
patient split and the same value as the CLAM initialization seed. Every task,
scale, and model under one seed uses the same task-specific patient split.

The additional runs retain the seed 1 training protocol so that all five
repeats are comparable: 200 maximum epochs, early-stopping patience 20, and
minimum stopping epoch 50.

## Prepare splits

```bash
cd /home/jupyter/image_team/projects/SMC_MIL
python tools/prepare_smc_repeated_cv_splits.py \
  --folds 5 \
  --seeds 11 21 31 41
```

The existing seed 1 split directories remain unchanged. Additional splits use
the suffix `_standard5_seedN`.

## Run on two GPUs

```bash
FEATURE_ROOT=/home/jupyter/image_team/projects/SMC_MIL/data/features/uni_v2
mkdir -p results/logs

nohup bash tools/run_smc_repeated_cv_grid.sh \
  --gpu 1 \
  --worker a \
  --feature-root "$FEATURE_ROOT" \
  --seeds 11,21,31,41 \
  > results/logs/repeated5_gpu1_scales_a.log 2>&1 &

nohup bash tools/run_smc_repeated_cv_grid.sh \
  --gpu 3 \
  --worker b \
  --feature-root "$FEATURE_ROOT" \
  --seeds 11,21,31,41 \
  > results/logs/repeated5_gpu3_scales_b.log 2>&1 &
```

Worker `a` runs 40x high-grade ACR and AMR. Worker `b` runs 40x significant
rejection and all three tasks at 20x, 10x, and 5x. This offsets the much larger
40x bags. A completed `summary.csv` is skipped on restart.

## Check progress

```bash
pgrep -af 'run_smc_repeated_cv_grid.sh|run_smc_cv_grid.sh|main.py'
tail -n 40 results/logs/repeated5_gpu1_scales_a.log
tail -n 40 results/logs/repeated5_gpu3_scales_b.log

find results -maxdepth 2 -type f \
  \( -path '*/smc_acr_high_grade_*_cv5val_s*/summary.csv' \
     -o -path '*/smc_amr_positive_*_cv5val_s*/summary.csv' \
     -o -path '*/smc_significant_rejection_*_cv5val_s*/summary.csv' \) \
  | wc -l
```

There should be 60 completed experiment directories after including seed 1:
three tasks times four scales times five seeds. Each experiment contains five
fold checkpoints.

## Summarize

```bash
python tools/summarize_smc_repeated_cv.py \
  --results-root results \
  --output-dir results/smc_repeated_cv_summary \
  --folds 5 \
  --seeds 1 11 21 31 41 \
  --threshold 0.5
```

Outputs:

- `per_seed_oof_metrics.csv`: one pooled OOF result per task, scale, and seed
- `repeated_cv_summary.csv`: mean, standard deviation, minimum, and maximum
  across seeds
- `paired_scale_differences.csv`: paired scale differences and win counts on
  the same seeds

The threshold-dependent metrics in this initial summary use 0.5. Threshold
selection must be performed within training data before treating sensitivity
or specificity as a clinical operating point.

## GSE290577 external validation

Evaluate every seed-level five-fold checkpoint ensemble separately. Worker `a`
evaluates 40x high-grade ACR and AMR; worker `b` evaluates all remaining
experiments. Use separate output roots because both processes write an
experiment inventory and combined CSV.

```bash
GSE_WORK_ROOT=/home/jupyter/data/image_team/GSE290577_work
GSE_FEATURE_ROOT=/home/jupyter/image_team/projects/SMC_MIL/data/features/uni_v2/GSE290577
GSE_OUTPUT_ROOT=results/gse290577_external/gold5_repeated

nohup env CUDA_VISIBLE_DEVICES=1 python -u tools/gse290577/evaluate_all_internal_experiments.py \
  --results-root results \
  --work-root "$GSE_WORK_ROOT" \
  --feature-root "$GSE_FEATURE_ROOT" \
  --output-root "$GSE_OUTPUT_ROOT/worker_a" \
  --include-variant cv5val \
  --include-task acr_high \
  --include-task amr_positive \
  --include-task significant_rejection \
  --include-seed 1 --include-seed 11 --include-seed 21 --include-seed 31 --include-seed 41 \
  --worker a \
  --device cuda \
  --bootstrap 2000 \
  > results/logs/gse_gold5_repeated_worker_a.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python -u tools/gse290577/evaluate_all_internal_experiments.py \
  --results-root results \
  --work-root "$GSE_WORK_ROOT" \
  --feature-root "$GSE_FEATURE_ROOT" \
  --output-root "$GSE_OUTPUT_ROOT/worker_b" \
  --include-variant cv5val \
  --include-task acr_high \
  --include-task amr_positive \
  --include-task significant_rejection \
  --include-seed 1 --include-seed 11 --include-seed 21 --include-seed 31 --include-seed 41 \
  --worker b \
  --device cuda \
  --bootstrap 2000 \
  > results/logs/gse_gold5_repeated_worker_b.log 2>&1 &
```

Each experiment evaluates its five fold checkpoints as an ensemble. The result
therefore contains seed-level external performance, not a single 25-model
ensemble. Keep GSE results separate from internal model selection.
