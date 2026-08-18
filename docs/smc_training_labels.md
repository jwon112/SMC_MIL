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

This writes three CSVs to `dataset_csv/`:

- `smc_acr_4class_0r_1r_2r_3r.csv`: labels `0R`, `1R`, `2R`, `3R` as classes 0--3. The current matched data has no `3R` bag, so this file is an auditable label export, not a runnable four-class experiment.
- `smc_acr_binary_0r_vs_1r2r3r.csv`: `0R` versus `1R/2R/3R`.
- `smc_amr_binary_pamr0_vs_positive.csv`: `pAMR0` versus `pAMR1`, `pAMR1(I+)`, or `pAMR2`.

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

## Patient-grouped splits and training

The two binary tasks are registered in `create_splits_seq.py` and `main.py`.
Use three folds; the four-class ACR CSV cannot be split or trained until at
least one `3R` event is available.

```bash
# Run once per task. The patient-level case_id prevents slide/event leakage.
python create_splits_seq.py \
  --task task_smc_acr_binary_0r_vs_1r2r3r \
  --k 3 --val_frac 0.15 --test_frac 0.15

python create_splits_seq.py \
  --task task_smc_amr_binary_pamr0_vs_positive \
  --k 3 --val_frac 0.15 --test_frac 0.15
```

Point `--data_root_dir` to a directory containing the feature bags for every
row in the chosen CSV. With the current separate DICOM and MRXS feature roots,
create a combined feature view first; do not silently train after one source's
feature files have been filtered out.
