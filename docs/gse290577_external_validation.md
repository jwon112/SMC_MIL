# GSE290577 external validation

This pipeline keeps the public cohort separate from internal training data. It supports:

- 15 biopsy H&E WSI and 45 matched CD3/CD8/CD68 IHC WSI.
- 195 Xenium tissue cores from 62 patients, represented as ROIs in two OME-TIFF files.
- Physical-resolution patches at 0.25, 0.50, 1.00, and 2.00 um/px.
- Separate WSI-level and core-level metrics. Core confidence intervals are bootstrapped by patient.

The native OpenSlide level number is not used as a magnification label. GSE290577 SVS files have irregular native pyramids, so patches are sampled by MPP and resized to the UNI v2 input size.

## 1. Extract the archive

Inspect the archive root before extracting:

```bash
tar -tf /home/jupyter/data/image_team/GSE290577_dataset.tar | head -n 20
mkdir -p /home/jupyter/data/image_team/GSE290577
tar -xf /home/jupyter/data/image_team/GSE290577_dataset.tar \
  -C /home/jupyter/data/image_team/GSE290577
```

The commands below assume that this creates:

```text
/home/jupyter/data/image_team/GSE290577/dataset
```

## 2. Build WSI and core inventories

```bash
cd /home/jupyter/image_team/projects/SMC_MIL

DATASET_ROOT=/home/jupyter/data/image_team/GSE290577/dataset
WORK_ROOT=/home/jupyter/data/image_team/GSE290577_work

python tools/gse290577/prepare_gse290577.py \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$WORK_ROOT/manifests"

python tools/gse290577/build_gse290577_core_manifest.py \
  --dataset-root "$DATASET_ROOT" \
  --rds "$DATASET_ROOT/analysis_objects/GSE290577_heart_spatial.rds" \
  --output-dir "$WORK_ROOT/manifests"
```

The core step needs base R but does not need Seurat packages. If R is unavailable on the server, run `export_core_bounds.R` elsewhere and pass its CSV with `--bounds-csv`.

Visually verify the affine core alignment before segmentation:

```bash
python tools/gse290577/export_inventory_previews.py \
  --dataset-root "$DATASET_ROOT" \
  --inventory "$WORK_ROOT/manifests/gse290577_core_inventory.csv" \
  --output-dir "$WORK_ROOT/qc/core_alignment"
```

Open `qc/core_alignment/index.html` in JupyterLab. Do not continue with core encoding if ROIs are systematically displaced.

## 3. Create AtlasPatch masks

Run WSI and core inventories separately. The original public files are read-only; all outputs go under `WORK_ROOT`.

```bash
CHECKPOINT=/home/jupyter/data/image_team/exp3_inbox/models/AtlasPatch/model.pth
CONFIG=/home/jupyter/data/image_team/exp3_inbox/models/AtlasPatch/sam2.1_hiera_t.yaml

for COHORT in wsi core; do
  python tools/gse290577/segment_openslide_atlaspatch.py \
    --dataset-root "$DATASET_ROOT" \
    --inventory "$WORK_ROOT/manifests/gse290577_${COHORT}_inventory.csv" \
    --processing-root "$WORK_ROOT" \
    --checkpoint "$CHECKPOINT" --config "$CONFIG" --device cuda \
    --skip-existing
done
```

Review `tissue_overlay.png` outputs, especially all 195 core ROIs. A manual correction may replace `tissue_mask.png` with `tissue_mask_manual.png`; coordinate generation prefers the manual file without overwriting either source.

## 4. Build MPP-specific coordinate manifests

```bash
for MPP in 0.25 0.50 1.00 2.00; do
  TAG=$(printf '%.2f' "$MPP" | sed 's/0*$//;s/\.$//;s/\./p/')
  for COHORT in wsi core; do
    python tools/gse290577/build_openslide_patch_coords.py \
      --dataset-root "$DATASET_ROOT" \
      --inventory "$WORK_ROOT/manifests/gse290577_${COHORT}_inventory.csv" \
      --processing-root "$WORK_ROOT" \
      --target-mpp "$MPP" --patch-size 256 --stride 256 \
      --output-manifest "$WORK_ROOT/manifests/gse290577_${COHORT}_${TAG}mpp.csv"
  done
done
```

Use `gse290577_wsi_he.csv` as the label/QC subset for the primary 15-WSI H&E evaluation. Coordinates and feature bags may still be generated once for all 60 WSI so that IHC-only models can be evaluated later.

## 5. Encode UNI v2 features

Smoke-test one cohort before launching both GPU shards:

```bash
MPP=0.25
TAG=0p25
FEATURE_DIR=/home/jupyter/data/image_team/features/uni_v2/GSE290577/${TAG}mpp/wsi

python extract_features_openslide.py \
  --dataset-root "$DATASET_ROOT" --processing-root "$WORK_ROOT" \
  --manifest "$WORK_ROOT/manifests/gse290577_wsi_${TAG}mpp.csv" \
  --feat-dir "$FEATURE_DIR" --model-name uni_v2 --target-patch-size 224 \
  --batch-size 32 --device cuda --amp --limit 1 --max-patches-per-slide 128
```

Remove the smoke-test feature directory or rerun with `--overwrite`, then run full deterministic shards:

```bash
mkdir -p "$FEATURE_DIR/logs"
nohup env CUDA_VISIBLE_DEVICES=1 python extract_features_openslide.py \
  --dataset-root "$DATASET_ROOT" --processing-root "$WORK_ROOT" \
  --manifest "$WORK_ROOT/manifests/gse290577_wsi_${TAG}mpp.csv" \
  --feat-dir "$FEATURE_DIR" --model-name uni_v2 --target-patch-size 224 \
  --batch-size 32 --device cuda --amp --num-shards 2 --shard-index 0 \
  > "$FEATURE_DIR/logs/encode_gpu1.log" 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 python extract_features_openslide.py \
  --dataset-root "$DATASET_ROOT" --processing-root "$WORK_ROOT" \
  --manifest "$WORK_ROOT/manifests/gse290577_wsi_${TAG}mpp.csv" \
  --feat-dir "$FEATURE_DIR" --model-name uni_v2 --target-patch-size 224 \
  --batch-size 32 --device cuda --amp --num-shards 2 --shard-index 1 \
  > "$FEATURE_DIR/logs/encode_gpu3.log" 2>&1 &
```

Repeat for `core` and the other MPP values. Verify every feature directory with `verify_feature_bags.py`; the OpenSlide H5 files store `patch_level=0` because coordinates are always in source level-0 space.

## 6. External evaluation

Evaluate one internal experiment at a time. The three fold checkpoints are ensembled by averaging positive-class probabilities. The external cohort must never be used to choose the epoch or probability threshold.

```bash
python evaluate_external_clam.py \
  --checkpoint-dir results/<internal_experiment>_s1 \
  --task acr_high \
  --cohort wsi_he "$WORK_ROOT/manifests/gse290577_wsi_he.csv" \
    /home/jupyter/data/image_team/features/uni_v2/GSE290577/0p25mpp/wsi \
  --cohort core "$WORK_ROOT/manifests/gse290577_core_inventory.csv" \
    /home/jupyter/data/image_team/features/uni_v2/GSE290577/0p25mpp/core \
  --output-dir results/gse290577_external/<internal_experiment>/acr_high_0p25mpp \
  --embed-dim 1536 --threshold 0.5 --bootstrap 2000 --device cuda
```

Available task names are `acr_any`, `acr_high`, `amr_positive`, and `any_rejection`. ACR-only WSI cases have no AMR grade and are automatically omitted from AMR evaluation. Report WSI and core results separately; the core `n=195` does not represent 195 independent patients.
