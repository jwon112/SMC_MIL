# 2026-10-05: ACR 판정 임계값 대조실험 계획

## 질문과 사전 결정

575개 event 중 ACR 양성은 14개, 양성 환자는 11명이다. 기존 0.5 판정에서
놓친 양성을 학습 환자 안에서 선택한 임계값으로 더 찾을 수 있는지 확인한다.
주 비교는 `inner_f2` 대 `fixed_0p5`, 보조 비교는 `inner_f1` 대 `fixed_0p5`다.
F2는 recall의 비중을 높이는 탐색용 기준이며, 임상적으로 합의된 비용 함수는 아니다.
바깥 평가 결과를 보고 임계값이나 선택 목적을 바꾸지 않는다.

이 단계는 구현 및 실행 전 계획이다. 서버의 실제 WSI feature 학습 결과는 아직 없다.
이번 실험은 기존 결과를 보고 정한 후속 탐색 실험이며, 독립 외부 검증으로 해석하지 않는다.

## 데이터와 모델

- manifest: `event_stain_mil_gold_provisional_20260930/acr_high`.
- pending 25개 slide 제외 상태 유지. 양성 3개 복구는 병리 확인 후 별도 버전에서 수행.
- 기존 seed 1, 11, 21, 31, 41의 patient-grouped 5-fold를 그대로 사용.
- UNI2 40x, input 1536, hidden 128, dropout 0.25.
- stain branches + observed presence mask, train/eval patch cap 2048.
- Adam lr 0.0002, weight decay 0.00001, 역빈도 weighted sampler.
- 내부 학습 최대 50 epoch, 최소 실행 10 epoch, stopping patience 10.
- 모든 임계값 조건은 동일한 바깥 모델·동일한 예측 점수를 사용.

## 평가 환자가 선택에 관여하지 않는 절차

기존 trainer는 바깥 validation loss로 checkpoint를 선택했다. 해당 checkpoint와
OOF를 재사용하면 이번 절차의 독립된 바깥 평가를 확보할 수 없어 새로 학습한다.

각 바깥 fold에서 다음을 수행한다.

1. 바깥 학습 환자를 내부 3-fold로 나눈다. 양성 event가 하나라도 있는 환자를
   양성 환자로 분류하고 고정 seed로 섞어 양성 환자부터 순환 배치한다.
   음성 환자는 먼저 각 fold에 배치한 뒤 event 수가 가장 적은 fold에 배치한다.
   같은 환자의 양성·음성 event는 항상 함께 둔다.
2. 내부 calibration fold를 남겨두고, 나머지 환자를 fit/stop으로 나눈다.
   stop 분할은 양성·음성 환자 수 중 작은 값과 5 중 최솟값의 fold 수를 사용하고,
   위와 같은 환자 배치 규칙의 첫 fold로 고정한다. 성능을 보고 분할을 고르지 않는다.
3. fit 환자로 학습하고 stop loss로 checkpoint와 epoch를 선택한다.
   calibration 환자는 학습과 checkpoint 선택 모두에 관여하지 않는다.
4. 내부 3개 calibration 예측을 합쳐 바깥 학습 event 전체의 cross-fitted 점수를 만든다.
   이 점수에서 각각 F1 또는 F2를 최대화하는 임계값을 선택한다.
   점수가 같으면 더 높은 임계값을 선택한다. 판정은 `probability >= threshold`다.
5. 내부에서 선택한 best epoch 3개의 중앙값으로 학습 길이를 고정한다.
   바깥 학습 환자 전체로 새 모델을 refit한다. 이 모델은 바깥 평가 loss를 보고 멈추지 않는다.
6. 바깥 평가 환자를 예측하고, 0.5/F1/F2 임계값을 동일한 예측에 적용한다.

총 5 seed × 5 outer fold × (3 inner fit + 1 refit) = **100번 학습**이다.
중단되면 완료된 outer fold는 건너뛰고, 미완료 outer fold의 내부 학습부터 다시 수행한다.
seed 목록·설정·입력·코드가 바뀌면 같은 결과 폴더의 재사용을 거부한다.

내부 모델보다 refit 모델의 학습 환자 수가 많아 점수 분포가 달라질 수 있다.
선택한 임계값이 refit 모델에 전달되는 성능 자체를 바깥 fold에서 평가한다.
기존 실험과 checkpoint 선택 방식이 다르므로, 기존 AP와 이번 AP 차이는 임계값 효과가 아니다.
이번 실험 안에서는 임계값을 바꾸어도 AUROC/AP가 동일해야 한다.

## 보고 항목과 해석

- seed별 575건 OOF에 fold별 선택 임계값을 적용한 TP, FP, TN, FN.
- precision, sensitivity, specificity, F1, F2, MCC, AUROC, AP.
- seed별 0.5 대비 차이와 seed 평균/표준편차.
- outer fold별 선택 임계값, 내부 best epoch, refit epoch.
- 양성·음성을 포함한 0.5 대비 판정 변경 event 목록.
- 내부 각 역할의 양성 event 수와 양성 환자 수.

seed 반복은 독립 환자 표본이 아니므로 평균/표준편차를 통계적 유의성으로 해석하지 않는다.
같은 환자가 여러 seed에서 반복되므로 seed별 TP를 합쳐 환자 수처럼 보고하지 않는다.
fold마다 임계값이 달라 5-seed 평균 확률에 평균 임계값을 적용하는 ensemble은 생성하지 않는다.
양성이 없는 평가 fold의 AUROC는 NaN으로 기록하고, seed 전체 OOF 지표로 주 비교한다.

민감도와 F2 개선이 여러 seed에서 일관적인지, 추가 TP에 비해 FP가 얼마나 늘어나는지 확인한다.
어떤 FP 비용까지 허용할지는 아직 임상적으로 정하지 않았으므로 최적 임상 임계값을 선언하지 않는다.
양성 환자 다양성이 늘어나는 실험은 아니며, 기존 점수의 판정 기준을 검증하는 실험이다.

## 구현과 검증

- 실행: [`tools/run_smc_threshold_control.py`](../tools/run_smc_threshold_control.py).
- 분할/임계값 규칙: [`utils/threshold_control.py`](../utils/threshold_control.py).
- 검증: [`tools/test_threshold_control.py`](../tools/test_threshold_control.py).
- 기존 presence-control의 5 seed OOF 환자 분할로 내부 75개 분할을 로컬 점검했다.
  fit 양성 4–8개, stop 양성 1–5개, calibration 양성 2–6개, outer test 양성 0–5개다.
  fit/stop/calibration/outer test 환자 중복 없음, calibration의 학습 event 전체 커버를 확인했다.
- 단위 검증은 혼합 라벨 환자 보존, outer label 변경과 내부 분할 독립성,
  F1/F2 선택 및 동률 규칙, 양성 환자 부족 시 차단을 포함한다.
- CPU 합성 feature로 audit → 20회 학습 → 완료 fold 재사용 → summarize를 검증했다.
  최종 `python tools/test_threshold_control.py` 실행은 4개 테스트 모두 통과했다.
  같은 바깥 예측을 사용하는 세 조건의 AUROC/AP 동일성도 확인했다.
- protocol에는 CSV·split·학습 코드 SHA256, feature 파일 크기/수정 시각을 기록한다.
  대용량 feature 자체의 content SHA256은 계산하지 않는다.

## 서버 명령어

프로젝트 루트와 `clam_latest` 환경에서:

```bash
git pull --ff-only origin main
python tools/test_threshold_control.py
python tools/run_smc_threshold_control.py --action audit
mkdir -p results/smc_event_threshold_control_20261005_exclude25
nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python tools/run_smc_threshold_control.py --action run --gpu 1 \
  > results/smc_event_threshold_control_20261005_exclude25/train.log 2>&1 &
```

진행 확인:

```bash
tail -n 20 results/smc_event_threshold_control_20261005_exclude25/train.log
```

완료 후:

```bash
python tools/run_smc_threshold_control.py --action summarize
```

결과 루트는 `results/smc_event_threshold_control_20261005_exclude25`다.
`per_seed_metrics.csv`, `paired_seed_deltas.csv`, `fold_metrics.csv`, `changed_calls.csv`와
각 `seed*/outer_*/thresholds.json`을 함께 검토한다. 기존 control 결과는 덮어쓰지 않는다.

## 방법 참고

학습 데이터와 임계값 선택 데이터를 분리해야 한다는 근거:
[scikit-learn: decision threshold tuning](https://scikit-learn.org/1.5/modules/classification_threshold.html).
임계값을 선택한 표본 자체에서 보고한 최고 F1을 독립 평가 결과로 사용하지 않는다.
