아래 경로는 모두 저장소 루트 기준이다.

## 1. 분류 Task와 라벨

### `main.py`

`task_smc_*` 분기에서 각 분류 task, class 수, 사용할 CSV를 연결한다.

| Task | 라벨 CSV | 음성 / 양성 |
| --- | --- | --- |
| `task_smc_acr_binary_0r_vs_1r2r3r` | `dataset_csv/smc_acr_binary_0r_vs_1r2r3r.csv` | 0R / 1R·2R·3R |
| `task_smc_acr_binary_0r1r_vs_2r3r` | `dataset_csv/smc_acr_binary_0r1r_vs_2r3r.csv` | 0R·1R / 2R·3R |
| `task_smc_amr_binary_pamr0_vs_positive` | `dataset_csv/smc_amr_binary_pamr0_vs_positive.csv` | pAMR0 / pAMR1 이상 |
| `task_smc_any_rejection_binary` | `dataset_csv/smc_any_rejection_binary.csv` | 모두 음성 / ACR 또는 AMR 양성 |

### `build_smc_training_labels.py`

병리 라벨과 슬라이드를 연결하여 위 CSV를 생성한다. 각 행은 하나의
슬라이드 feature bag이며, `case_id`는 환자 단위 분할에 사용하는
pseudonymous patient key이다.

## 2. 실제 실험 조건

### `tools/run_smc_cv_grid.sh`

현재 비교 실험의 task, 해상도, 모델 및 학습 인자를 한곳에서 정의한다.
기본 grid는 4개 task와 다음 4개 해상도의 조합으로 총 16개 실험이다.

- `l0_0p25mpp_40x`
- `l1_0p50mpp_20x`
- `l2_1p00mpp_10x`
- `l3_2p00mpp_5x`

주요 기본 조건은 다음과 같다.

| 항목 | 설정 |
| --- | --- |
| Encoder feature | UNI2-h, 1,536차원 |
| MIL model | `clam_sb`, `small` |
| Cross-validation | 환자 단위 3-fold, train 2 folds / validation 1 fold |
| 최대 epoch | 200 |
| Early stopping | 사용 |
| Optimizer | Adam |
| Learning rate | `2e-4` |
| Weight decay | `1e-5` |
| Dropout | `0.25` |
| Bag loss | Cross entropy |
| Instance loss | SVM |
| Bag loss weight | `0.7` |
| Instance sample 수 | `B=8` |
| Class imbalance | training-only weighted sampling |

`--weak-train-root`는 validation을 그대로 유지하고 weak-label 데이터를
train에만 추가한다. `--stain-root`와 `--stain-cohort`는 train과
validation을 지정한 염색 cohort로 제한한다.

## 3. 데이터 분할

### `create_smc_cv_splits.py`

환자를 stratified 3-fold로 나누고, 매 반복에서 두 fold를 train, 나머지
한 fold를 validation으로 사용한다. 별도 test set은 없다. 동일
`case_id`의 슬라이드는 항상 같은 fold에 들어가므로 환자 누출을 방지한다.

## 4. 학습과 평가

### `utils/core_utils.py`

주요 함수는 `train()`, `train_loop_clam()`, `validate_clam()`, `summary()`다.
Bag-level classification loss와 instance clustering loss를 계산하고,
validation 결과에 따라 checkpoint와 early stopping을 관리한다.

### `compare_experiments.py`

각 fold의 결과를 모아 task와 해상도별 성능을 비교한다. AUROC뿐 아니라
class imbalance를 고려한 sensitivity, specificity, balanced accuracy와
PR-AUC를 함께 확인할 때 사용한다.

## 5. 모델 내부 구현

### `models/model_clam.py`

`CLAM_SB.forward()`에서 슬라이드별 패치 feature에 gated attention을
적용하고 가중합하여 slide-level logits를 생성한다. 실험 가설과 조건을
확인한 뒤 aggregation 구현까지 검토할 때 보면 된다.

### `dataset_modules/dataset_generic.py`

`Generic_MIL_Dataset`에서 `pt_files/<slide_id>.pt` 형태의 슬라이드 feature
bag을 읽는다. `case_id`를 이용한 환자 grouping과 split 확장 로직도
포함한다.

## 권장 검토 순서

1. `main.py`와 `dataset_csv/smc_*.csv`: 무엇을 분류하는지
2. `tools/run_smc_cv_grid.sh`: 어떤 조건으로 비교하는지
3. `create_smc_cv_splits.py`: train/validation을 어떻게 나누는지
4. `build_smc_training_labels.py`: 라벨이 어떻게 만들어지는지
5. `utils/core_utils.py`와 `compare_experiments.py`: 어떻게 학습·평가하는지
6. `models/model_clam.py`와 `dataset_modules/dataset_generic.py`: 모델 내부 구현

실험 설계를 검토하는 목적이라면 우선 1~4번을 보면 되고, 모델 구현까지
확인할 때 5~6번을 이어서 보면 된다.
