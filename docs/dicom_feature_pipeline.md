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

`resnet50_trunc` is a practical first smoke-test encoder. Use the intended
foundation encoder (for example `uni_v1`) for the final experiment after its
checkpoint availability has been verified.
