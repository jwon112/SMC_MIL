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

## 서버 코드 통합 결과 (300556c, 06d9f99)

첫 통합의 39개 파일과 추가 pathomics 설정 2개를 Git으로 받아 로컬과 통합했습니다. 실행 상태나 실험 성능은
코드 존재만으로 판단하지 않으며, 아래 분류는 코드의 역할과 의존성에 근거합니다.

| 계열 | 주 진입점 | 함께 보존할 코드·입력 |
|---|---|---|
| 현재 event 대조실험 | `tools/run_smc_presence_mask_control.sh`, `tools/run_smc_patch_sampling_control.py` | provisional event/split, event trainer, models/utils |
| 기존 HE augmentation 비교 | `run_smc_gold_he_aug.py`, `run_smc_gold_he_aug_gpu0.sh` | `audit_smc_aug_reuse.py`, `augment_uni2_wsi.py`, `smc_he_aug.py`, `feature_view_bank.py`, `results/he_aug_inputs` |
| 초기 HE augmentation 탐색 | `run_he_aug_full.py`, `run_he_aug_technical.py`, `run_he_lr_check.py` | `train_clam_feature_views.py`, `clam_view_training.py`, `cohorts/he_manual_acr_v1`, `dataset_csv/stain_aug_labels.csv` |
| WSI 정량 특징 추출 | `run_wsi_pathomics.py`, `run_pathomics_parallel.py`, `run_pathomics_fast.py` | `pathomics_core.py`, `pathomics_fast.py`, `pathology_feature_extractor_v2.py`, 실행 config |
| pathomics 실행·점검 | `fast_run.sh`, `fast_check.sh`, `fast_eta.sh`, `check_eta.sh` | `pathomics_all_wsi.json`, extraction receipt 및 로그 |
| 미래 예측 | `prepare_smc_future.py`, `run_smc_future.py` | `cohorts/smc_future_v1`, augmentation config·feature bank |
| 염색·영상 점검 | `train_stain_family_uni.py`, `diagnose_dicom_pixels.py`, `diagnose_feature_coordinate_alignment.py`, review notebooks | 검토 라벨·원본 영상·좌표 |
| 환경·내보내기 | `smc.sh`, `export_pathomics.mjs`, `marker_static.ipynb` | 기존 conda, `eta.txt` 참조, workbook 입력 및 별도 Node exporter 환경 |

특히 `run_pathomics_fast.py`는 parallel launcher를 import하고 기존 engine 4개 파일의
SHA-256을 검사합니다. 이름에 fast/v2가 붙어도 이전 모듈을 대체한 독립 구현으로
간주할 수 없습니다. 해당 핵심 모듈의 이동·편집은 parity 및 signature 검증에 영향을
주므로 실행 파이프라인 재구성과 함께 진행해야 합니다.

`eta.txt`는 `smc.sh`에서 `bash eta.txt`로 참조됩니다. 단순 로그로 분류하지 않습니다.
`06d9f99`에서 받은 두 설정을 포함하면 pathomics 설정의 용도는 다음과 같습니다.

| 설정 | 대상 범위 | 출력 위치의 마지막 폴더 | review_only |
|---|---|---|---|
| `pathomics_config.json` | 기본값 gold_cohort / confirmed_he | `pathomics_v3_measurementqa` | true |
| `pathomics_extract_exploratory.json` | 기본값 gold_cohort / confirmed_he | `pathomics_exploratory_v1` | false |
| `pathomics_all_wsi.json` | 명시적 all_feature_wsi / all | `pathomics_all_wsi_v1` | false |

세 설정 모두 기존 `dataset_csv/smc_acr_binary_0r_vs_1r2r3r.csv`와
`dataset_csv/stain_aug_labels.csv`를 참조합니다. 현재 event 실험의 provisional
라벨이 pathomics에도 반영됐다고 해석하지 않습니다. 설정 파일의 존재는 실제 실행
완료의 증거가 아니며, 대상 범위가 달라 중복 설정으로 취급하지 않습니다.

데이터 CSV는 별도 관리하더라도 `dataset_csv/stain_aug_labels.csv`와
`cohorts/he_manual_acr_v1/technical_train_slides_*.txt` 등은 기존 연구 입력입니다.
서버 선별의 HOLD는 미사용/삭제 대상이라는 뜻이 아닙니다.

## 결과 복사본 통합 (2026-10-04)

서버 결과 커밋 `801c5db`를 받은 뒤 결과 파일 4,557개를 대조했습니다.
로컬 전용 파일 49개 중 CSV 27개는 대표 파일과 열 순서, 모든 셀 값, 행 순서,
중복 행 수까지 일치했습니다. 인코딩/BOM·줄바꿈 차이는 내용 차이로 보지 않았습니다.
해당 복사본은 작업 폴더에서 빼고 `.server_result_sync/consolidation_20261004/copies/`에
원본 그대로 보관했습니다. [파일별 대응표](../tools/server_cleanup/result_copy_plan.json)에
이전 경로·대표 경로·원본 SHA-256·행 수가 기록되어 있습니다.

| 자료 | 사용할 대표 위치 | 정리한 로컬 복사본 수 |
|---|---|---:|
| 통합 성능표·과거 protocol별 요약 | `smc_all_results/`, `smc_cv_standard3_comparison/`, `smc_stain_comparison/`, `smc_cv_comparison/weakunique3_partial/`, `results_old2/smc_cv_comparison/` | 6 |
| 외부 검증 | `results/gse290577_external/` | 4 |
| 반복 CV | `results/smc_repeated_cv_summary/` | 3 |
| 기존 gold event 결과 | `results/smc_event_stain_summary/` | 5 |
| 미래 예측 pair 결과 | `results/smc_future_pair_summary/` | 2 |
| 앙상블·임계값 비교 | `results/smc_prediction_ensembles/` | 7 |

첫 행의 `smc_*` 경로는 `results/` 아래입니다. 기존 gold event 결과는 578 events를
사용한 이력이며, 현재 575-event provisional 결과와 합쳐 덮어쓰지 않습니다.

중복이 아닌 로컬 자료 22개(그림 11개, CSV 10개, 설명 문서 1개)는 Git 공유에
포함했습니다. 그중 이름만으로 구별하기 어려웠던 자료 두 개는 다음처럼 정리했습니다.

- `averaged_summary (1).csv` → `results/smc_cv_fulltrain100cosine_comparison/averaged_summary.csv`:
  16개 nested3 fulltrain100cosine 실험 요약.
- `baseline_experiment_summary.md` → `results_old2/smc_cv_comparison/baseline_experiment_summary.md`:
  초기 nested3 early-stopping 실험 설명. 해당 폴더의 성능표와 함께 보존.

`tools/plot_stain_comparison_with_mixed.py`는 `(3)`, `(4)` 복사본 대신 대표 CSV를
읽습니다. 변경 전후 생성한 비교표 8행의 SHA-256이 일치했습니다.
복사본 정리 도구의 테스트 3개는 동일 셀/줄바꿈 차이, 중복 행 수 불일치 차단,
이름 변경 충돌 및 원본 보존을 검사합니다.

바이트가 같은 파일 그룹 283개도 발견했지만, 서로 다른 배율·seed 실험에 보관된
동일 split/config는 실험 재현의 근거이므로 일괄 제거하지 않았습니다.
`he_aug_int64_backup.aGq9UT`의 JSON은 현재 `he_aug_inputs` JSON과
`smc_he_aug.py`의 입력 코드 해시가 달라 복구 이력으로 유지했습니다.
서버에서 제외된 PKL/NPZ/가중치와 진행 중인 두 대조실험은 이번 내용 비교 범위에
포함되지 않습니다. 특히 `split_*_results.pkl`은 기존 집계 코드가 읽는 예측 출력으로,
불필요한 가중치 파일로 분류하면 안 됩니다.

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

- Research results: `python tools/publish_server_results.py --action plan` inventories
  results, historical results and archived experiments before sharing tables, configs,
  figures and logs. `--action stage` checks the saved file hashes and stages those
  outputs for a normal Git commit/push. Weights and feature binaries stay on the server.
  Server inventory is required before judging the total size; local results are only a subset.
- Server/local source consolidation: pull shared code first, then run
  `python tools/publish_server_sources.py --action plan` on the server.
  `--action publish` commits and pushes the selected current server source.
  See [`server cleanup README`](../tools/server_cleanup/README.txt) for scope.
- [`topk_device_fix.md`](topk_device_fix.md): SmoothTop1SVM device workaround.
- [`upstream_clam/README.md`](upstream_clam/README.md): original upstream CLAM
  usage guide retained for provenance. It is not the SMC operating manual.
