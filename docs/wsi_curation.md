# WSI Quality and Stain Curation

This workflow creates review material before quality filtering or stain-based
ablation. It does not make clinical or image-quality exclusion decisions from
automation alone.

## 1. Build the inventory and review queues

Run on the server after the AtlasPatch thumbnails exist for both datasets.

```bash
cd /home/jupyter/image_team/projects/SMC_MIL

CURATION_ROOT=/home/jupyter/data/image_team/labels/derived/wsi_curation_v1

python tools/curation/build_wsi_curation_manifest.py \
  --dicom-root /home/jupyter/data/image_team/exp3_inbox \
  --mrxs-root /home/jupyter/data/image_team/mrxs13_inbox \
  --label-xlsx /home/jupyter/data/image_team/labels/raw/WSI_LABEL_ID_MATCH_20260814.xlsx \
  --quality-exclusions /home/jupyter/data/image_team/labels/derived/slide_quality_exclusions.csv \
  --output-dir "$CURATION_ROOT" \
  --export-quality-previews \
  --export-stain-cluster-previews
```

Generated files:

- `slide_curation_manifest.csv`: every input WSI with thumbnail/mask metrics,
  pathology-ID match status, automatic stain candidates, and blank manual fields.
- `quality_review_queue.csv`: missing/unreadable thumbnails, broad image-metric
  screening candidates, plus a random audit sample. Only this queue needs per-slide
  quality review.
- `quality_review_images/`: flat thumbnail previews corresponding to the queue.
- `stain_signature_review.csv`: stain suffix groups with slide counts and example
  paths. Map repeated signatures here rather than reviewing every WSI separately.
- `stain_color_cluster_review.csv` and `stain_cluster_images/`: 24 thumbnail-color
  clusters with representative slides. They are a second review aid when filenames
  do not encode stain names. Label only visually pure clusters; leave mixed clusters
  blank for later per-slide review.

The automatic `HE`, IHC marker, and special-stain classifications are only
high-confidence filename matches. Keep other signatures as `unknown` until
they are mapped from metadata, LIS information, or visual review.

## 2. Record manual decisions

Open `tools/curation/quality_review_notebook.ipynb` in JupyterLab and run its
single code cell. It shows one thumbnail at a time with its automatic review
trigger. `Usable`, `Usable low quality`, and `Exclude` save immediately to
`quality_review_queue.csv`; `Previous`, `Skip`, and `Clear decision` allow
navigation and correction without editing CSV rows directly.

The notebook's `CURATION_ROOT` must match the directory used to build the
manifest. Enter a reviewer name in the small text field before deciding slides.

The saved fields are:

```text
quality_manual_status: usable | usable_low_quality | exclude
quality_manual_reason
quality_reviewer
quality_reviewed_at
```

The default screening tail is 5% per metric, not a diagnostic cutoff. It is
intentionally broad because faint IHC and grid-like scan degradation can be
missed by a very small tail. The queue records its reason in
`quality_auto_flags`: low tissue area, low sharpness, pale/desaturated tissue,
low tissue contrast, or possible regular grid artifact. The latter is a
frequency-based screening signal, not a definitive artifact detector.

Do not use pale staining alone as an exclusion criterion. Reserve `exclude` for
slides that are technically unusable: blank/nearly blank scans, severe scan or
color corruption, grid/tiling degradation that prevents tissue interpretation,
or insufficient focus/resolution to inspect tissue.

After completing the first quality pass, quantify whether the screening criteria
actually enriched for exclusions before changing the tail fraction again:

```bash
python tools/curation/summarize_quality_review.py \
  --manifest "$CURATION_ROOT/slide_curation_manifest.csv" \
  --quality-decisions "$CURATION_ROOT/quality_review_queue.csv" \
  --output-dir "$CURATION_ROOT/quality_review_report"
```

This reports exclusion rates for each automatic review trigger and for random
audit slides. An exclusion found in the random-audit group is evidence that the
screening rules missed that failure mode; it is a reason to broaden or revise
the queue before freezing the cohort.

In `stain_signature_review.csv`, fill:

```text
stain_group: HE | IHC | special_other | unknown
stain_raw
stain_confidence
stain_note
```

Keep unverified groups as `unknown`. The H&E/non-H&E ablation will exclude them
from both restricted arms but retain them in the all-stains baseline.

`stain_color_cluster_review.csv` uses the same fields. It is lower confidence than
metadata or explicit filename labels, so it only fills currently `unknown` slides
and never overrides a signature-level mapping.

## 3. Produce the curated manifest

```bash
python tools/curation/apply_wsi_curation_decisions.py \
  --manifest "$CURATION_ROOT/slide_curation_manifest.csv" \
  --quality-decisions "$CURATION_ROOT/quality_review_queue.csv" \
  --stain-map "$CURATION_ROOT/stain_signature_review.csv" \
  --stain-cluster-map "$CURATION_ROOT/stain_color_cluster_review.csv" \
  --require-quality-review-complete \
  --output-dir "$CURATION_ROOT/curated"
```

This writes:

- `slide_curation_manifest_curated.csv`: final per-slide quality and stain values.
- `quality_exclusions_curated.csv`: exclusions in a form compatible with the
  existing label-building workflow.
- `curation_summary.csv`: counts by source, quality status, and stain group.
- `stain_manual_review_queue.csv`: only the quality-clean slides still classified
  as `unknown` after signature and pure-color-cluster mapping. Fill `stain_group`,
  `stain_raw`, `stain_confidence`, and `stain_note` here for the remaining
  per-slide decisions.

For a compact flat-image queue of only these remaining slides, run the same
command with `--export-unresolved-stain-previews`. Then apply completed
per-slide stain decisions in a new output folder:

```bash
python tools/curation/apply_wsi_curation_decisions.py \
  --manifest "$CURATION_ROOT/slide_curation_manifest.csv" \
  --quality-decisions "$CURATION_ROOT/quality_review_queue.csv" \
  --stain-map "$CURATION_ROOT/stain_signature_review.csv" \
  --stain-cluster-map "$CURATION_ROOT/stain_color_cluster_review.csv" \
  --stain-decisions "$CURATION_ROOT/curated/stain_manual_review_queue.csv" \
  --require-quality-review-complete \
  --output-dir "$CURATION_ROOT/curated_final"
```

Freeze this curated manifest before rebuilding task CSVs and patient-level CV
splits. Compare the all-stains baseline, quality-clean cohort, H&E-only cohort,
and non-H&E cohort using the same patient split definitions.
