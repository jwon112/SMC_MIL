# SMC Gold Dataset Audit

`tools/audit_smc_gold_dataset.py` converts the primary gold task manifests into a reproducible
cohort, multiplicity, confounder, split, and metadata-shortcut audit. It reports patient, biopsy-event,
and slide levels separately. The metadata baseline is an audit model, not a clinical prediction model.

Run the complete audit on the server:

```bash
cd /home/jupyter/image_team/projects/SMC_MIL

CURATED=/home/jupyter/data/image_team/labels/derived/wsi_curation_v2_final/slide_curation_manifest_curated.csv
FEATURE_ROOT=/home/jupyter/image_team/projects/SMC_MIL/data/features/uni_v2
AUDIT_ROOT=results/smc_dataset_audit

python tools/audit_smc_gold_dataset.py \
  --label-dir dataset_csv \
  --curation-manifest "$CURATED" \
  --feature-root "$FEATURE_ROOT" \
  --output-dir "$AUDIT_ROOT" \
  --folds 5 \
  --seeds 1 11 21 31 41 \
  --threshold 0.5
```

The script creates:

- `cohort_flow.csv`: inventory, exact-match, quality, gold-cohort, and feature-availability stages.
- `dataset_table1.csv`: patient-, event-, and slide-level sample and positive counts for each task.
- `event_multiplicity_audit.csv`: one row per task/event, including slide count, stain composition,
  quality/morphometry summaries, and event-, slide-, and patient-balanced weights.
- `confounder_audit.csv`: label prevalence by categorical metadata and positive-versus-negative
  standardized differences for numeric metadata.
- `split_audit.csv`: patient, event, slide, and positive counts for every repeated-CV split.
- `metadata_only_baseline.csv`: patient-grouped OOF logistic-regression results for acquisition,
  multiplicity, stain-only, quality-only, and combined metadata feature sets when available.

The metadata models use the same patient-level stratification rule and requested seeds as the image
experiments. Training samples are inverse-weighted by the number of events from each patient. Variables
such as total patient event count and event index intentionally test cohort shortcuts; they must not be
interpreted as prospectively available clinical predictors.

If `smc_significant_rejection_binary.csv` is absent, the script derives it by joining the ACR-high and
AMR gold manifests and applying their logical OR. Omitting `--curation-manifest` or `--feature-root`
still produces the gold-only portions of the audit, while stain, quality, and feature-flow fields are
added only when their source data are supplied.
