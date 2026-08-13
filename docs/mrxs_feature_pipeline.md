# MRXS UNI2-h Multi-Resolution Pipeline

The MRXS set has nine native OpenSlide pyramid levels. Use the first four for
the same physical-scale ablation as the DICOM set:

| Level | Approx. MPP | Descriptive scale |
| --- | ---: | --- |
| L0 | 0.25 | 40x |
| L1 | 0.50 | 20x |
| L2 | 1.00 | 10x |
| L3 | 2.00 | 5x |

Run manual mask review before creating the manifests and feature bags. The
coordinate builder selects `tissue_mask_manual.png` when present and otherwise
uses `tissue_mask.png`; it does not use postprocessed masks.

## Server compatibility probe

The server must be able to call OpenSlide `read_region`, not merely discover a
`.mrxs` filename. Run this once before coordinate generation or encoding:

```bash
DATASET_ROOT=/home/jupyter/data/image_team/mrxs13_inbox
python - <<'PY'
from pathlib import Path
import openslide

path = next(Path("/home/jupyter/data/image_team/mrxs13_inbox").rglob("*.mrxs"))
with openslide.OpenSlide(str(path)) as slide:
    print(path, slide.level_count, slide.level_dimensions[:4])
    slide.read_region((0, 0), 1, (256, 256)).convert("RGB")
print("OpenSlide read_region OK")
PY
```

If this fails with `Cannot read slide position info`, install the current
self-contained OpenSlide binary into the active Python environment before
proceeding:

```bash
python -m pip install --upgrade --force-reinstall --no-cache-dir \
  "openslide-python>=1.4.0" "openslide-bin>=4.0.0"
```

`openslide-python` 1.4+ loads `openslide-bin` before a system/conda OpenSlide
library. This avoids mixing conda's incompatible JPEG dependency chain into
the existing UNI/SAM environment. Restart the shell or kernel, then repeat the
probe above.

## Coordinates and Manifests

Set the shared roots once:

```bash
DATASET_ROOT=/home/jupyter/data/image_team/mrxs13_inbox
FEATURE_ROOT=/home/jupyter/data/image_team/mrxs13_features/uni_v2
```

Build coordinates and an encoding manifest per level. L0 uses the existing
AtlasPatch/manual coordinate choice; L1-L3 generate independent 256px grids.

```bash
# L0 / 0.25 MPP / 40x
python build_dicom_feature_manifest.py \
  --dataset-root "$DATASET_ROOT" \
  --output _clam/mrxs_feature_manifest_l0.csv

# L1 / 0.50 MPP / 20x (repeat with L2 and L3)
python build_multilevel_mrxs_patch_coords.py \
  --dataset-root "$DATASET_ROOT" \
  --pyramid-level 1 \
  --coords-filename patch_coords_l1.h5 \
  --skip-existing

python build_dicom_feature_manifest.py \
  --dataset-root "$DATASET_ROOT" \
  --coords-filename patch_coords_l1.h5 \
  --output _clam/mrxs_feature_manifest_l1.csv
```

For L2/L3, replace both `1` and `l1` with `2`/`l2` or `3`/`l3`.

## UNI2-h Smoke Test and Full Extraction

Run a smoke test for each level before full extraction. A sampled smoke-test
bag must not be used in training; use a distinct `smoke` directory or delete it
before the full run.

```bash
python extract_features_mrxs.py \
  --dataset-root "$DATASET_ROOT" \
  --manifest "$DATASET_ROOT/_clam/mrxs_feature_manifest_l0.csv" \
  --feat-dir "$FEATURE_ROOT/smoke_l0" \
  --model-name uni_v2 \
  --target-patch-size 224 \
  --batch-size 32 \
  --device cuda \
  --amp \
  --max-patches-per-slide 128 \
  --limit 1
```

The final L0 path is `$FEATURE_ROOT/l0_0p25mpp_40x`; use the corresponding L1,
L2, and L3 names for the other scale ablations. Omit both `--limit` and
`--max-patches-per-slide` for final encoding.

After a full level completes, verify every `.h5`/`.pt` pair against its
manifest before training:

```bash
python verify_feature_bags.py \
  --manifest "$DATASET_ROOT/_clam/mrxs_feature_manifest_l0.csv" \
  --feat-dir "$FEATURE_ROOT/l0_0p25mpp_40x" \
  --expected-level 0 \
  --expected-embed-dim 1536
```

Two GPU processes can use the same manifest and feature directory with
`CUDA_VISIBLE_DEVICES=1`/`3` and `--num-shards 2 --shard-index 0`/`1`, exactly
as in the DICOM feature pipeline. The feature H5 files contain 1,536-dim UNI2-h
vectors, so CLAM training must use `--embed_dim 1536`.
