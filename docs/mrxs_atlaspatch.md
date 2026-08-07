# MRXS AtlasPatch Preparation

`tools/atlaspatch_mrxs/segment_mrxs_atlaspatch.py` runs the same AtlasPatch
SAM2 full-thumbnail prompt used for the DICOM data, but reads 3DHISTECH MRXS
files with OpenSlide. It writes only automatic files in each `atlaspatch/`
directory and never touches `*_manual` outputs.

Server prerequisites in the intended conda environment:

```bash
conda install -c conda-forge openslide openslide-python
python -c "import openslide, sam2; print('OpenSlide and SAM2 available')"
```

Smoke test one slide before the complete set:

```bash
DATASET_ROOT=/home/jupyter/data/image_team/mrxs13_inbox
ATLASPATCH_CHECKPOINT=/home/jupyter/data/image_team/exp3_inbox/models/AtlasPatch/model.pth
ATLASPATCH_CONFIG=/home/jupyter/data/image_team/exp3_inbox/models/AtlasPatch/sam2.1_hiera_t.yaml

bash tools/atlaspatch_mrxs/run_atlaspatch_mrxs.sh --limit 1
```

After confirming the output, run all 13 slides by omitting `--limit 1`.
The expected output per slide is `thumbnail.png`, `tissue_mask.png`,
`tissue_overlay.png`, and `patch_coords.h5`. Review and manual correction can
then write parallel `*_manual` files.
