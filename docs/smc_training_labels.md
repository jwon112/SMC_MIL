# SMC CLAM Training Labels

Build the CLAM training CSVs from the external Sheet1 pathology-label workbook.
The workbook and the quality-exclusion list remain in the data area, not Git.

```bash
python build_smc_training_labels.py \
  --label-xlsx /home/jupyter/data/image_team/labels/raw/WSI_LABEL_ID_MATCH_20260814.xlsx \
  --dicom-root /home/jupyter/data/image_team/exp3_inbox \
  --mrxs-root /home/jupyter/data/image_team/mrxs13_inbox \
  --quality-exclusions /home/jupyter/data/image_team/labels/derived/slide_quality_exclusions.csv
```

This writes five CSVs to `dataset_csv/`:

- `smc_acr_binary_0r_vs_1r2r3r.csv`: `0R` versus `1R/2R/3R`.
- `smc_acr_binary_0r1r_vs_2r3r.csv`: `0R/1R` versus `2R/3R`.
- `smc_amr_binary_pamr0_vs_positive.csv`: `pAMR0` versus `pAMR1`, `pAMR1(I+)`, or `pAMR2`.
- `smc_any_rejection_binary.csv`: positive when ACR is at least `1R` or AMR is positive.
- `smc_significant_rejection_binary.csv`: positive when ACR is at least `2R` or AMR is positive.

Every row represents one feature bag. `case_id` is a stable pseudonym derived
from the source patient ID; it is shared across all slide bags belonging to the
same patient and must be used for group-wise splitting. `slide_id` is the exact
feature filename stem written by the extraction scripts.

## Imbalance

Do not duplicate positive rows in the CSV: that would also duplicate validation
or test cases if applied before splitting. Instead create patient-grouped splits
first and use CLAM's existing training-only weighted sampler:

```bash
python main.py ... --weighted_sample --embed_dim 1536
```

For the AMR task, the rare positive class still has very few patients. Report
patient-grouped cross-validation results as exploratory and retain each fold's
class counts alongside AUROC, balanced accuracy, sensitivity, and specificity.

## Patient-grouped standard 3-fold CV

Use `create_smc_cv_splits.py`, not the older `create_splits_seq.py`. Each fold
uses two patient folds for training and the remaining patient fold for
validation. The validation fold is used for early stopping and is the reported
cross-validation result; there is no separate test split or inner split.

```bash
# Run once per task. The patient-level case_id prevents slide/event leakage.
python create_smc_cv_splits.py \
  --task task_smc_acr_binary_0r_vs_1r2r3r

python create_smc_cv_splits.py \
  --task task_smc_acr_binary_0r1r_vs_2r3r

python create_smc_cv_splits.py \
  --task task_smc_amr_binary_pamr0_vs_positive

python create_smc_cv_splits.py \
  --task task_smc_any_rejection_binary

python create_smc_cv_splits.py \
  --task task_smc_significant_rejection_binary
```

Point `--data_root_dir` to the selected scale directory containing the feature
bags for every row in the chosen CSV. Verify that the shared feature directory
contains both the exp3 and MRXS bags before training.

## Four-scale Grid

After creating the CV splits, run exactly one training process per GPU.
The two workers partition the four tasks across GPUs and each process runs its
eight assigned task-scale combinations sequentially.

```bash
FEATURE_ROOT=/home/jupyter/image_team/projects/SMC_MIL/data/features/uni_v2

nohup bash tools/run_smc_cv_grid.sh \
  --gpu 1 --worker acr --feature-root "$FEATURE_ROOT" \
  > results/logs/grid_gpu1.log 2>&1 &

nohup bash tools/run_smc_cv_grid.sh \
  --gpu 3 --worker amr --feature-root "$FEATURE_ROOT" \
  > results/logs/grid_gpu3.log 2>&1 &
```

This yields 16 task-scale experiments in total. Each experiment runs all three
CV folds sequentially. Do not start more than one of these training workers
per GPU. The runner skips any experiment that already has its `summary.csv`,
so rerunning the same worker resumes incomplete task-scale combinations.
