# Project Documentation

This directory contains documentation for the active SMC cardiac-allograft
WSI workflows. Start with the repository-level [`README`](../README.md), then
use the topic index below.

## 현재 서버 기준 (2026-10-04)

현재 라벨과 실험 진입점은 [`smc_event_stain_mil.md`](smc_event_stain_mil.md)의
"현재 실행 기준"을 따릅니다. `wsi_curation_v1`을 사용하는 이전 문서는 라벨 생성
이력을 설명하는 문서입니다. 버전 번호가 작다는 이유만으로 입력을 제거하지 않습니다.

- 현재 stain 라벨: `labels/derived/wsi_curation_v2_provisional_20260930/`
- 현재 event·split: `labels/derived/event_stain_mil_gold_provisional_20260930/`
- 보류 25건 목록: `labels/derived/wsi_stain_provisional_20260930_upload/pending_review_25.csv`
- 과거 3-fold 실험 80개: `archive/experiments/early_cv/` 및 `weak_linkage/`.
  서버 적용 결과 파일 881개와 집계 대상 유지가 확인됐습니다.
- 정리·복구 명령: [`tools/server_cleanup/README.txt`](../tools/server_cleanup/README.txt).
- 루트·라벨 추가 점검: `python tools/server_cleanup/audit_workspace.py`.

현재 라벨, 원본 라벨, 이전 baseline은 목적이 다릅니다. 후속 버전이 있어도
이전 실험의 입력 근거는 보존합니다. 서버에만 있는 augmentation·pathomics·future
코드도 연구 이력 확인 후 Git 관리 또는 별도 보관 여부를 결정합니다.

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
- [`smc_gold_dataset_audit.md`](smc_gold_dataset_audit.md): reproducible cohort,
  multiplicity, confounder, split, and metadata-shortcut audit.
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
