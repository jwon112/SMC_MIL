# Project Documentation

This directory contains documentation for the active SMC cardiac-allograft
WSI workflows. Start with the repository-level [`README`](../README.md), then
use the topic index below.

## Core pipeline

- [`dicom_feature_pipeline.md`](dicom_feature_pipeline.md): DICOM manifests,
  multiscale coordinates, manual masks, and UNI feature extraction.
- [`mrxs_atlaspatch.md`](mrxs_atlaspatch.md): AtlasPatch segmentation for MRXS.
- [`mrxs_feature_pipeline.md`](mrxs_feature_pipeline.md): MRXS manifests and
  feature extraction.
- [`smc_training_labels.md`](smc_training_labels.md): SMC task definitions,
  label generation, and patient-grouped cross-validation.
- [`code_handover_guide_ko.md`](code_handover_guide_ko.md): Korean code map for
  task definitions, training, models, and evaluation.

## Cohort review and external validation

- [`wsi_curation.md`](wsi_curation.md): WSI quality and stain review workflow.
- [`gse290577_external_validation.md`](gse290577_external_validation.md):
  GSE290577 preparation and external evaluation.

## Experiment workflows

- [`smc_repeated_5fold.md`](smc_repeated_5fold.md): repeated patient-grouped CV.
- [`smc_event_stain_mil.md`](smc_event_stain_mil.md): event/stain MIL.
- [`smc_future_pair_baseline.md`](smc_future_pair_baseline.md): future-outcome
  pair baseline.
- [`smc_prediction_ensembles.md`](smc_prediction_ensembles.md): prediction
  ensembles.
- [`smc_ensemble_size.md`](smc_ensemble_size.md): ensemble-size analysis.
- [`smc_ensemble_threshold_calibration.md`](smc_ensemble_threshold_calibration.md):
  threshold calibration.
- [`M-AQW_implementation_plan.md`](M-AQW_implementation_plan.md): M-AQW design
  and implementation notes.

## Maintenance and provenance

- [`topk_device_fix.md`](topk_device_fix.md): SmoothTop1SVM device workaround.
- [`upstream_clam/README.md`](upstream_clam/README.md): original upstream CLAM
  usage guide retained for provenance. It is not the SMC operating manual.
- [`../archive/README.md`](../archive/README.md): code retained outside the
  supported pipeline.
