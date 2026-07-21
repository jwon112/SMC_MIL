# CLAM - Multiple Instance Learning for Whole Slide Images

## 목차
[설치](#설치) • [WSI 분할 및 패칭](#wsi-분할-및-패칭) • [특성 추출](#특성-추출) • [훈련](#훈련) • [테스트](#테스트) • [실험 결과 비교](#실험-결과-비교) • [히트맵 시각화](#히트맵-시각화)

**참고**: 특성 추출 시 이미지 패치를 224 x 224로 크기 조정합니다. 다른 크기를 사용하려면 **extract_features_fp.py**에서 `--target_patch_size`를 지정할 수 있습니다.

## 이 리포지터리의 로컬 변경 사항

이 저장소는 원본 CLAM 코드베이스를 그대로 가져온 뒤, 다음과 같은 **호환성/편의성 중심 수정 및 기능 추가**를 적용한 버전입니다.

- **훈련 및 환경 호환성**
  - **NumPy 2.0 대응**: `utils/core_utils.py`의 `EarlyStopping`에서 `np.Inf` → `np.inf`로 수정해 NumPy 2.0 환경에서 학습이 중단되지 않도록 했습니다.
  - **DataLoader shared memory 에러 방지**: `utils/utils.py`의 `get_split_loader` 및 히트맵용 `compute_from_patches` 내부에서 PyTorch `DataLoader`의 `num_workers`를 GPU일 때 `0`으로 설정해 `/dev/shm` 부족으로 인한 `Bus error`를 방지했습니다.

- **CAMELYON16 실험 설정 관련 수정**
  - **task_3_camelyon16_binary / task_4_camelyon16_multiclass**에서 `main.py`가 `--data_root_dir` 아래에 또 한 번 하드코딩된 `camelyon16_features`를 붙이던 동작을 제거하고, 사용자가 넘긴 `--data_root_dir`를 그대로 feature 디렉토리로 사용하도록 변경했습니다.  
    - 예: `--data_root_dir ./data/features` → 내부에서 추가 조인 없이 바로 `./data/features`를 사용.

- **Feature Extractor 확장**
  - **UNI v2 지원 추가**: `uni_v2` (ViT-H, 1280-dim), `uni_v2_l` (ViT-L, 1024-dim) 옵션 추가
  - **자동 다운로드**: UNI v1, v2는 HuggingFace에서 자동으로 다운로드되도록 개선 (환경변수 설정 불필요)

- **Patch Quality Filtering (Blur Filtering)**
  - **Blur profiling 도구**: `utils/blur_utils.py` 추가 - 데이터셋의 blur 분포 분석 및 적절한 threshold 추천
  - **Feature extraction 시 blur filtering**: `extract_features_fp.py`에 `--blur_mode drop`, `--blur_thr` 옵션 추가
  - Blurry 패치를 제거하여 학습 데이터 품질 향상

- **실험 결과 비교 도구**
  - **compare_experiments.py** 추가: 여러 실험의 결과를 자동으로 수집, 비교, 시각화
  - Test AUC, Test Accuracy 막대 그래프 자동 생성

- **히트맵/시각화 파이프라인 개선**
  - **자동 설정 스크립트 추가**: `setup_heatmap_config.py`를 새로 작성해,  
    - WSI 디렉토리 스캔 → `heatmaps/process_lists/*.csv` 생성,  
    - 훈련된 체크포인트 탐색 → `heatmaps/configs/*.yaml` 자동 생성,  
    - 사용자가 최소한의 인자만으로 `create_heatmaps.py`를 실행할 수 있도록 했습니다.
  - **조직 영역만 crop하는 옵션 추가**:
    - `heatmaps/configs/config_template.yaml` 및 생성 스크립트에 `auto_tissue_roi` 옵션을 추가했습니다.  
    - `use_roi=False`일 때, `create_heatmaps.py`가 `wsi_object.contours_tissue`의 bounding box를 자동 계산해 **유리 배경을 제외한 조직 영역만을 대상으로 heatmap을 생성**합니다.
  - **대형 슬라이드/히트맵 처리 안정화**:
    - 히트맵 생성 시 DataLoader `num_workers=0` 적용으로 메모리 사용을 줄이고,  
    - attention 점수 → percentile 변환 부분을 NumPy 2.0/SciPy와 호환되도록 벡터화해 대형 슬라이드에서도 안정적으로 동작하도록 수정했습니다.

## 설치
자세한 설치 지침은 [설치 가이드](docs/INSTALLATION.md)를 참조하세요.

## WSI 분할 및 패칭

다음 예제는 표준 형식(.svs, .ndpi, .tiff 등)의 전체 슬라이드 이미지가 DATA_DIRECTORY 폴더 아래에 저장되어 있다고 가정합니다.

```bash
DATA_DIRECTORY/
	├── slide_1.svs
	├── slide_2.svs
	└── ...
```

### 기본, 완전 자동 실행
``` shell
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --patch --stitch 
```

결과 디렉토리 구조:
```bash
RESULTS_DIRECTORY/
	├── masks/          # 분할 결과
	├── patches/        # 패치 좌표 (.h5)
	├── stitches/       # 시각화 (선택)
	└── process_list_autogen.csv
```

주요 옵션:
* `--patch_level`: 패치 추출 레벨 (기본값: 0)
* `--no_auto_skip`: 이미 처리된 파일도 재처리

템플릿 사용 예:
``` shell
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --preset bwh_biopsy.csv --seg --patch --stitch
```
### 두 단계 실행 (매개변수 수동 조정)
1. 분할만 수행:
``` shell
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg
```

2. CSV 파일에서 매개변수 조정 후 패칭:
``` shell
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --process_list CSV_FILE_NAME --patch --stitch
```
## 특성 추출
```bash
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size 512 --slide_ext .svs
```

### AtlasPatch DICOM WSI

AtlasPatch manual/original coordinate outputs and multi-source DICOM WSI use a
separate extractor rather than the OpenSlide/SVS command above. See
[`docs/dicom_feature_pipeline.md`](docs/dicom_feature_pipeline.md) for manifest
generation, a one-slide smoke test, and server Git synchronization steps.
결과 디렉토리 구조:
```bash
FEATURES_DIRECTORY/
    ├── h5_files/      # 특성 + 좌표
    └── pt_files/      # 특성만 (훈련용)
```

CSV 파일은 처리할 슬라이드 파일 이름 목록(확장자 없이)을 포함해야 합니다.

### 지원하는 Feature Extractor
`--model_name` 옵션:
* `resnet50_trunc`: ResNet50 (기본값, 1024-dim)
* `uni_v1`: UNI v1 (1024-dim)
* `uni_v2`: UNI v2 ViT-H (1280-dim)
* `uni_v2_l`: UNI v2 ViT-L (1024-dim)
* `conch_v1`: CONCH v1 (512-dim)
* `conch_v1_5`: CONCH v1.5 (448×448 입력)

UNI/CONCH는 HuggingFace에서 자동 다운로드됩니다.

### Patch Quality Filtering (Blur Filtering)
**Blur Profiling:**
```bash
python utils/blur_utils.py \
    --csv_path ./dataset_csv/camelyon16_binary.csv \
    --data_slide_dir /path/to/slides \
    --data_h5_dir ./data/patches \
    --slide_ext .tif \
    --out_csv blur_scores.csv
```

**Feature Extraction with Blur Filtering:**
```bash
python extract_features_fp.py \
    --blur_mode drop \
    --blur_thr 196.07 \
    ... (기타 옵션)
```

옵션:
* `--blur_mode`: `none` (필터링 없음) 또는 `drop` (blurry 패치 제거)
* `--blur_thr`: Blur threshold

### 데이터셋 구성
```bash
DATA_ROOT_DIR/
    └── DATASET_DIR/
        ├── h5_files/    # 특성 + 좌표
        └── pt_files/    # 특성만 (훈련용)
```
각 데이터셋은 DATA_ROOT_DIR 아래의 하위 폴더로 구성되며, 각 슬라이드의 특성은 **pt_files** 폴더의 .pt 파일로 저장됩니다.

CSV 파일은 다음 열을 포함해야 합니다:
* `case_id`: 환자 ID
* `slide_id`: 슬라이드 ID (.pt 파일 이름과 일치)
* 레이블 열: 슬라이드 수준 레이블

**dataset_csv** 폴더에 예제가 포함되어 있습니다.

## 훈련

### 훈련 분할 생성
여러 폴드(예: 10-fold)의 훈련/검증/테스트 분할을 생성:
``` shell
python create_splits_seq.py --task task_1_tumor_vs_normal --seed 1 --k 10
```

### 이진 분류 훈련 예제
참고: --embed_dim은 CONCH의 경우 512로, UNI 및 resnet50_trunc의 경우 1024로 설정해야 합니다. UNI v2 ViT-H의 경우 1280입니다.

``` shell
CUDA_VISIBLE_DEVICES=0 python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code task_1_tumor_vs_normal_CLAM_50 --weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type clam_sb --log_data --data_root_dir DATA_ROOT_DIR --embed_dim 1024
```

### 다중 클래스 분류 훈련 예제
``` shell
CUDA_VISIBLE_DEVICES=0 python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code task_2_tumor_subtyping_CLAM_50 --weighted_sample --bag_loss ce --inst_loss svm --task task_2_tumor_subtyping --model_type clam_sb --log_data --subtyping --data_root_dir DATA_ROOT_DIR --embed_dim 1024
``` 
**옵션**:
- `--model_type`: `clam_sb` (단일 브랜치, 기본값) 또는 `clam_mb` (다중 브랜치)
- `--B`: 클러스터링에 사용되는 패치 수
- `--log_data`: 텐서보드 로깅 활성화

결과는 **results/exp_code**에 저장됩니다.

## 테스트
```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --k 10 \
    --models_exp_code exp_code \
    --save_exp_code exp_code_cv \
    --task task_name \
    --model_type clam_sb \
    --results_dir results \
    --data_root_dir DATA_ROOT_DIR \
    --embed_dim 1024
```

각 인수에 대한 정보는 `python eval.py -h`를 참조하세요.

## 실험 결과 비교
여러 실험의 결과를 자동으로 수집하고 비교할 수 있습니다:

```bash
python compare_experiments.py \
    --results_dir ./results \
    --experiments exp1 exp2 exp3
```

이 스크립트는:
* 각 실험 디렉토리의 `summary.csv`를 자동으로 수집하여 통합 CSV 생성
* 폴드별 평균 및 표준편차 계산
* Test AUC와 Test Accuracy 막대 그래프 생성 (각 실험마다 다른 색상)
* 모든 결과를 `results/comparison/` 폴더에 저장

출력 파일:
* `combined_summary.csv`: 모든 실험의 원본 데이터 통합
* `averaged_summary.csv`: 실험별 평균값
* `test_auc_comparison.png`: Test AUC 비교 그래프
* `test_acc_comparison.png`: Test Accuracy 비교 그래프

## 히트맵 시각화
구성 파일을 작성하고 **heatmaps/configs**에 저장한 후 실행:
``` shell
CUDA_VISIBLE_DEVICES=0 python create_heatmaps.py --config config_template.yaml
```

원시 결과는 **heatmaps/heatmap_raw_results**에, 최종 결과는 **heatmaps/heatmap_production_results**에 저장됩니다.
각 구성 옵션은 **heatmaps/configs/config_template.yaml**을 참조하세요.
