# DICOM Feature Pipeline

This pipeline replaces CLAM's OpenSlide/SVS patch-extraction stage for the
AtlasPatch DICOM dataset. It never uses `patch_coords_post.h5`.

## Server Git Check

Run this in the server clone before and after pulling a new commit:

```bash
cd /path/to/SMC_MIL/image_team/CLAM-master
git fetch --prune origin
git status --short
git rev-list --left-right --count HEAD...origin/main
git pull --ff-only
```

The first count is local-only commits and the second is commits missing from
the server. A clean, synchronized checkout reports `0 0` before the pull.

## Build the Encoding Manifest

Run on the server after the manual mask review is complete:

```bash
python build_dicom_feature_manifest.py \
  --dataset-root /home/jupyter/data/image_team/exp3_inbox
```

It writes `_clam/dicom_feature_manifest.csv` under the dataset root. Each row
uses `patch_coords_manual.h5` when present; otherwise it uses
`patch_coords.h5`. The source paths are relative to the dataset root so the
manifest is portable between machines with different root paths.

The manifest is an encoding manifest, not the final CLAM label CSV. Add the
clinical label column later to a separate training CSV with at least:

```text
case_id,slide_id,label
```

## Smoke Test and Encode

Start with a single slide. This verifies DICOM tile decoding, coordinate
selection, model loading, and feature writing without scanning the full set:

```bash
python extract_features_dicom.py \
  --dataset-root /home/jupyter/data/image_team/exp3_inbox \
  --manifest /home/jupyter/data/image_team/exp3_inbox/_clam/dicom_feature_manifest.csv \
  --feat-dir /home/jupyter/data/image_team/exp3_features \
  --model-name resnet50_trunc \
  --batch-size 64 \
  --max-patches-per-slide 512 \
  --sample-seed 0 \
  --limit 1
```

`--max-patches-per-slide` is only for a fast smoke test; its output is a
partial bag and must not be used for CLAM training. After a successful smoke
test, omit both `--limit` and `--max-patches-per-slide`. Output is compatible
with CLAM training: `<feat-dir>/h5_files/<slide_id>.h5` and
`<feat-dir>/pt_files/<slide_id>.pt`.

The reader caches 128 decoded DICOM tiles per slide by default. This reduces
repeated JPEG decoding for adjacent 256px patches. Set `--tile-cache-size 0`
only when host RAM is unusually constrained.

`resnet50_trunc` is a practical first smoke-test encoder. Use the intended
foundation encoder (for example `uni_v1` or `uni_v2`) for the final experiment
after its checkpoint availability has been verified. `uni_v1` writes
1,024-dimensional features; `uni_v2` denotes the official UNI2-h model and
writes 1,536-dimensional features, so CLAM training must use the matching
embedding dimension.

## Multi-Resolution DICOM Ablation

The original AtlasPatch coordinates and the completed UNI2-h run use level 0,
the largest DICOM pixel matrix. Do not reuse those level-0 coordinates at a
lower DICOM level: that only changes the sampling resolution of the same field
of view. Instead, generate a new 256px grid in each requested level while
using the same thumbnail tissue mask. `tissue_mask_manual.png` is used where
present; all other slides use `tissue_mask.png`. Postprocessed masks are never
used.

Store each scale in a separate feature directory under one encoder root:

```text
data/features/uni_v2/
  l0_0p25mpp_40x/
  l1_0p50mpp_20x/
  l2_1p00mpp_10x/
  l3_2p00mpp_5x/
```

The nominal magnification labels are descriptive. Use the MPP stored in each
coordinate/feature H5 as the authoritative physical resolution.

For L1/0.5 MPP, first generate coordinates and a matching manifest:

```bash
DATASET_ROOT=/home/jupyter/data/image_team/exp3_inbox
FEATURE_ROOT=/home/jupyter/data/image_team/exp3_features/uni_v2

python build_multilevel_patch_coords.py \
  --dataset-root "$DATASET_ROOT" \
  --pyramid-level 1 \
  --coords-filename patch_coords_l1.h5 \
  --skip-existing

python build_dicom_feature_manifest.py \
  --dataset-root "$DATASET_ROOT" \
  --coords-filename patch_coords_l1.h5 \
  --output _clam/dicom_feature_manifest_l1.csv
```

The feature extractor reads the level encoded in the coordinate H5; no separate
feature-extraction level argument is required. Run an L1 smoke test before the
full sharded extraction:

```bash
python extract_features_dicom.py \
  --dataset-root "$DATASET_ROOT" \
  --manifest "$DATASET_ROOT/_clam/dicom_feature_manifest_l1.csv" \
  --feat-dir "$FEATURE_ROOT/l1_0p50mpp_20x" \
  --model-name uni_v2 \
  --target-patch-size 224 \
  --batch-size 32 \
  --device cuda \
  --amp \
  --max-patches-per-slide 128 \
  --limit 1
```

Repeat with levels 2 and 3 only after L1 has passed its smoke test. Use the
same patient-level splits and model settings for every scale ablation.

## Multiple GPUs

Use one process per GPU and split the same manifest into disjoint shards. The
shards use every Nth manifest row, so they are deterministic and do not write
the same slide output. For example, to use physical GPUs 1 and 3, set
`CUDA_VISIBLE_DEVICES` separately for each process and use shard indices 0 and
1 with `--num-shards 2`. Each process sees its selected GPU as `cuda:0`, so
keep `--device cuda` in both commands. Interrupted runs can be resumed with
the same commands because already-complete `.h5` and `.pt` pairs are skipped.
Each shard writes its own failure CSV under `<feat-dir>/logs/`.
