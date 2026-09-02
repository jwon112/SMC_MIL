# SMC_MIL: Cardiac Transplant Rejection WSI Pipeline

This repository contains the code used to prepare whole-slide images (WSIs),
extract UNI2-h features, and train CLAM models for cardiac allograft rejection
research. It is a focused extension of the original
[CLAM](https://github.com/mahmoodlab/CLAM) codebase.

The WSI files, model checkpoints, feature bags, review images, clinical label
workbooks, and experiment results are deliberately kept outside Git.

## What Is Included

- AtlasPatch tissue masking and manual-mask aware DICOM feature extraction.
- MRXS masking, coordinate generation, and feature extraction.
- UNI v1 / UNI2-h encoder support (`uni_v2` produces 1536-dimensional bags).
- L0-L3 coordinate generation for physical-resolution ablations.
- Patient-grouped standard 3-fold cross-validation for ACR, AMR, and any
  rejection tasks.
- Train-only weak-label augmentation and stain-restricted cohort builders.
- WSI quality/stain curation manifests and Jupyter review notebooks.

The canonical label CSVs that define the gold cohort are versioned in
`dataset_csv/`. Generated split folders, feature files, logs, and result
directories are ignored because they are reproducible and/or large.

## Repository Layout

```text
dataset_csv/                 Gold-cohort CLAM label CSVs
dataset_modules/             CLAM dataset implementations
docs/                        Operational notes for each pipeline stage
models/                      Encoder builders and CLAM architectures
tools/atlaspatch_mrxs/       AtlasPatch workflow for MRXS slides
tools/curation/              Quality/stain inventory and review utilities
tools/run_smc_cv_grid.sh     Four-scale patient-CV training runner
extract_features_dicom.py    DICOM feature extraction
extract_features_mrxs.py     MRXS feature extraction
main.py                      CLAM training entry point
```

## Data Conventions

The server data roots are not part of this repository. The current workflows
expect paths analogous to:

```text
/home/jupyter/data/image_team/
  exp3_inbox/                DICOM WSI dataset
  mrxs13_inbox/              Additional MRXS WSI dataset
  exp3_features/uni_v2/      Feature bags, organized by resolution
  labels/raw/                Source clinical workbooks
  labels/derived/            Derived linkage and curation artifacts
```

Each feature directory contains CLAM-compatible pairs:

```text
<feature-root>/<scale>/
  h5_files/<slide_id>.h5
  pt_files/<slide_id>.pt
```

Feature filenames contain a sanitized combination of the source case path and
slide path, so duplicate slide basenames remain distinct.

## Installation

Create the project environment using the local server setup, then install the
dependencies required by the selected encoder. UNI2-h is gated on Hugging
Face: request access to `MahmoodLab/UNI2-h` and authenticate on the server
before running the extractor.

```bash
hf auth login
python -c "import pydicom, torch; print(pydicom.__version__, torch.cuda.is_available())"
```

For the DICOM / MRXS operational dependencies and commands, see the documents
linked below rather than treating this README as a full environment lockfile.

## Core Workflow

### 1. Build feature manifests

Use the masking result selected for each slide (`patch_coords_manual.h5` when
available, otherwise the original AtlasPatch coordinate file).

```bash
DATASET_ROOT=/home/jupyter/data/image_team/exp3_inbox

python build_dicom_feature_manifest.py \
  --dataset-root "$DATASET_ROOT"
```

Equivalent MRXS commands are documented in
[`docs/mrxs_feature_pipeline.md`](docs/mrxs_feature_pipeline.md).

### 2. Extract UNI2-h features

Run a small smoke test first. Do not use a patch-limited smoke-test directory
for model training.

```bash
python extract_features_dicom.py \
  --dataset-root "$DATASET_ROOT" \
  --manifest "$DATASET_ROOT/_clam/dicom_feature_manifest.csv" \
  --feat-dir /home/jupyter/data/image_team/exp3_features/smoke \
  --model-name uni_v2 \
  --target-patch-size 224 \
  --batch-size 32 \
  --device cuda --amp \
  --max-patches-per-slide 128 \
  --limit 1
```

For final extraction, omit `--limit` and `--max-patches-per-slide`. Use
separate shard processes for separate GPUs. The full multi-resolution process
is described in [`docs/dicom_feature_pipeline.md`](docs/dicom_feature_pipeline.md).

### 3. Build labels and patient-grouped splits

The gold label builder emits four binary tasks:

- ACR: `0R` vs `1R/2R/3R`
- ACR high grade: `0R/1R` vs `2R/3R`
- AMR: `pAMR0` vs positive
- Any rejection: ACR-positive or AMR-positive

```bash
python build_smc_training_labels.py \
  --label-xlsx /path/to/WSI_LABEL_ID_MATCH.xlsx \
  --dicom-root /path/to/exp3_inbox \
  --mrxs-root /path/to/mrxs13_inbox

python create_smc_cv_splits.py \
  --task task_smc_acr_binary_0r_vs_1r2r3r
```

The split script performs standard patient-grouped 3-fold CV: two patient
folds train and the remaining fold validates. The validation fold is used for
early stopping and reporting; it is not an independent held-out test set.

### 4. Run the four-scale grid

```bash
FEATURE_ROOT=/home/jupyter/image_team/projects/SMC_MIL/data/features/uni_v2
mkdir -p results/logs

nohup bash tools/run_smc_cv_grid.sh \
  --gpu 1 --worker acr --feature-root "$FEATURE_ROOT" \
  > results/logs/grid_gpu1.log 2>&1 &

nohup bash tools/run_smc_cv_grid.sh \
  --gpu 3 --worker amr --feature-root "$FEATURE_ROOT" \
  > results/logs/grid_gpu3.log 2>&1 &
```

Results are written under `results/` and intentionally excluded from Git.
Compare completed experiments with `compare_experiments.py`.

## Supporting Workflows

- [`docs/dicom_feature_pipeline.md`](docs/dicom_feature_pipeline.md): DICOM,
  manual masks, UNI2-h extraction, and multiscale coordinates.
- [`docs/mrxs_atlaspatch.md`](docs/mrxs_atlaspatch.md): AtlasPatch handling for
  MRXS slides.
- [`docs/mrxs_feature_pipeline.md`](docs/mrxs_feature_pipeline.md): MRXS
  feature manifests and extraction.
- [`docs/smc_training_labels.md`](docs/smc_training_labels.md): label tasks,
  imbalance handling, and patient-level CV.
- [`docs/wsi_curation.md`](docs/wsi_curation.md): quality/stain review and
  curation workflow.

## Git Synchronization on the Server

```bash
cd /home/jupyter/image_team/projects/SMC_MIL
git fetch --prune origin
git status --short
git pull --ff-only
```

Keep raw data, model files, features, clinical workbooks, and generated
results outside the repository. Commit only source code, compact canonical
task CSVs, scripts, and documentation needed to reproduce the workflow.
