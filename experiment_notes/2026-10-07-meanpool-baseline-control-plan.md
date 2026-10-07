# 2026-10-07: 평균 요약 + 로지스틱 회귀 기준선 계획

## 질문과 비교 대상

현재 575 event·양성 14 event·133 환자·양성 환자 11명에서, 단순한 이미지 특징
요약과 선형 분류기로도 기존 attention 모델의 성능을 얻을 수 있는지 확인한다.
사용자가 실행을 승인하여 구현했다. 로컬에는 실제 feature가 없어 본 실험은 서버에서 실행한다.

| 항목 | 이번 기준선 | 주 비교 대상 |
|---|---|---|
| 모델 | 평균 요약 + L2 로지스틱 회귀 | 기존 불균형 실험의 natural CE attention |
| 이미지 | 동일 UNI v2 40x feature / 1,262 slide | 동일 |
| 환자 | 저장된 5 seed × 5 fold의 fit/stop/test | 동일 |
| 영역 선택 | slide당 최대 2,048 patch, 일정 간격 고정 선택 | 평가 시 동일, 학습 시 무작위 선택 |
| 정보 결합 | patch 평균 → slide 동일 가중 평균 | patch attention → 염색별 slide attention |
| 염색 정보 | 별도 branch·presence mask 없음 | 염색별 branch·presence mask 포함 |
| 학습 비율 | 원래 비율, class weight 없음 | natural CE |
| 선택 | stop log loss로 규제 강도 선택 | stop CE로 checkpoint 선택 |
| 최종 학습 | fit만 사용, fit+stop 재학습 없음 | 동일 |
| 판정 | 0.5 고정 | 동일 |

이 비교는 모델·정보 결합·학습 절차를 포함한 단순 기준선 비교다.
**attention 하나의 효과나 parameter 수만의 효과를 분리하는 실험은 아니다.**
기존 balanced sampler와 weighted CE 결과는 참고로 함께 보고하되 주 비교는 natural CE다.
기존 attention 모델을 다시 학습하지 않고 저장된 동일 분할의 예측을 대조한다.

## 실행 전에 고정한 조건

- 입력 event/slide CSV의 해시가 기존 실험 protocol과 일치해야 한다.
- feature 파일의 slide 목록·크기·mtime가 기존 기록과 일치해야 한다.
  이는 파일 내용 전체의 hash 검증은 아니다.
- 모든 환자는 fit/stop/test 중 하나에만 속한다. 기존 25개 분할을 그대로 재사용한다.
- feature 평균은 라벨과 무관하게 계산해 저장한다. 전체 환자를 함께 표준화하지 않는다.
- StandardScaler의 평균·분산은 각 fold의 fit에서만 계산한다.
- L2 규제 강도 C 후보는 **0.001, 0.01, 0.1, 1.0** 네 개다.
  작을수록 강한 규제다. 후보당 fit에서 학습하고 stop의 가중치 없는 log loss로 선택한다.
  정확히 동률이면 작은 C를 선택한다. 선택 후 fit+stop을 합쳐 재학습하지 않는다.
- solver=lbfgs, max_iter=2000, tol=1e-6. 수렴하지 않으면 오류로 중단한다.
- 25개 분할 × 4후보 = 100회 작은 분류기 학습이다. 추가 후보 탐색은 포함하지 않는다.
- outer test로 C, 표준화, cutoff 또는 모델을 선택하지 않는다.

구현 동작은 [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)과
[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)의
공식 문서를 확인했다. 실행 환경의 sklearn/numpy/torch 버전은 protocol에 기록한다.

## 보고할 결과

주 지표는 AP다. seed별로 기존 natural CE와의 차이를 보고하고,
AUROC, TP/FP/FN, precision/recall, F1/F2/MCC도 함께 저장한다.
5 seed 평균과 5 seed 예측을 평균한 ensemble은 따로 보고한다.
양성이 없는 개별 fold는 전체 OOF 성능과 구분한다.
기존에 반복해서 살펴본 cohort이므로 새 외부 검증으로 취급하지 않는다.

## 로컬 검증

2026-10-07 검증 결과: 테스트 4개 통과(23.124초). 실제 manifest 해시,
25개 환자 분할과 기존 15개 OOF 예측 파일의 연결도 검증했다.
575 event·양성 14 event·133 환자·양성 환자 11명·1,262 slide가 일치한다.
실제 feature를 읽는 서버 audit와 본 학습 결과는 아직 대기 중이다.


- 일정 간격 patch 선택이 기존 평가 규칙과 일치하는지 확인.
- patch 개수가 다른 slide도 동일 가중치로 평균하는지 확인.
- 표준화에 stop 자료가 들어가지 않는지, C 동률 규칙이 지켜지는지 확인.
- 합성 자료로 audit → run → summarize → 재실행을 확인.
- 변경된 manifest와 손상된 feature 평균 cache가 거부되는지 확인.
- 실제 cohort의 manifest·환자 분할·기존 예측 연결을 로컬에서 검사.
  실제 feature 평균과 학습 성능은 서버 실행 전이므로 확인하지 않았다.

## 서버 실행

프로젝트 루트와 clam_latest 환경에서 실행한다. 기본 feature 경로는 기존 불균형
실험 protocol의 feature_dir를 읽는다. GPU 지정은 필요 없다.

```bash
git pull --ff-only origin main
python tools/test_meanpool_control.py
python tools/run_smc_meanpool_control.py --action audit

mkdir -p results/smc_event_meanpool_control_20261007_exclude25
nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python tools/run_smc_meanpool_control.py --action run \
  > results/smc_event_meanpool_control_20261007_exclude25/train.log 2>&1 &

tail -f results/smc_event_meanpool_control_20261007_exclude25/train.log
```

처음에는 feature를 읽으며 [POOL]이 출력되고, 이후 fold마다 [OK]가 출력된다.
완료 시 기존 attention 결과를 포함한 비교 표와 [COMPLETE]가 출력된다.
중단 후 동일 명령을 실행하면 완성된 평균 cache와 fold 예측을 재사용한다.
tail에서 Ctrl+C를 눌러도 nohup으로 실행한 학습은 계속된다.

끝난 뒤 요약만 다시 출력하려면 다음을 실행한다.

```bash
python tools/run_smc_meanpool_control.py --action summarize
```

결과 폴더에는 per_seed_metrics.csv, seed_summary.csv,
paired_deltas_vs_natural_ce.csv, ensemble_metrics.csv와 fold별 선택 기록이 생긴다.
feature 평균 cache와 모델 계수는 NPZ로 서버에 보관한다.
