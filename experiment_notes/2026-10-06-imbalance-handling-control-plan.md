# 2026-10-06: 불균형 처리 방식 대조실험 계획

## 질문과 조건

양성 14 event/11 환자로 학습하는 현재 조건에서 양성 반복 노출이 도움이 되는지,
자연 비율 학습 또는 클래스 가중 loss가 더 나은지 직접 비교한다.

| 조건 | sampler | 학습 loss |
|---|---|---|
| `balanced_sampler` | event 역빈도, 복원 추출, epoch당 N_fit번 | 일반 CE |
| `natural_ce` | 무작위 순서, 비복원, event당 epoch에 1번 | 일반 CE |
| `natural_weighted_ce` | natural_ce와 동일 | 클래스별 가중 CE |

가중 샘플링과 가중 loss를 동시에 적용하는 조건은 없다.
주 평가지표는 seed별 held-out AP(PR-AUC)다. AUROC와 0.5 판정에서의
TP/FP/FN, precision, sensitivity, specificity, F1, F2, MCC도 함께 본다.
특정 seed나 outer fold에서 좋은 조건만 골라 전체 성능을 보고하지 않는다.

## 고정 사항과 평가 절차

- manifest: provisional exclude25의 acr_high, 575 events / 양성 14 / slides 1262.
- 기존 patient-grouped outer 5-fold × seeds 1, 11, 21, 31, 41.
- stain-aware, observed presence mask, UNI2 40x, patch cap train/eval 2048.
- 모델 input 1536, hidden 128, dropout 0.25, Adam lr 0.0002, weight decay 0.00001.
- 각 outer 학습 환자에서 별도 fit/stop 환자 분할을 만든다. 세 조건에서 동일한 환자를 사용.
- stop_partition(outer_train, seed + 100003*fold + 1009)로 고정한다.
  양성 환자/음성 환자 수와 5의 최솟값으로 fold 수를 정하고 첫 fold를 stop에 둔다.
  환자 분할 규칙은 utils/threshold_control.py의 기존 규칙을 사용한다.
- 동일 fit 환자로만 학습하고 동일 stop 환자의 **일반, 비가중 CE**로 checkpoint를 선택.
- 최대 50 epoch, 최소 실행 10 epoch, patience 10. 최소 실행보다 앞선 best epoch도 선택 가능.
- 선택된 checkpoint를 그대로 outer test에서 평가한다. refit과 임계값 조정은 수행하지 않는다.
- 바깥 평가 환자는 가중치·학습 종료·checkpoint 선택에 사용하지 않는다.

outer train 중 stop 환자는 모델의 gradient 학습에 쓰이지 않는다. 이에 따라
이전 threshold 실험의 전체 outer-training refit 모델이나 과거 outer-validation-selected
checkpoint 모델과 직접적인 단일 변수 비교로 해석하지 않는다.
이번 세 조건 간 비교가 주 비교다. 별도 stop의 양성이 적다는 한계는 남는다.

총 3조건 × 5fold × 5seed = **75번 학습**이다. 학습 횟수가 같아도 조기 종료 시점은
조건별로 달라질 수 있다. epoch당 step 수는 세 조건 모두 N_fit로 맞춘다.
즉 학습 정책과 공통 checkpoint 선택 규칙을 적용한 전체 절차를 비교한다.

## 가중 CE의 batch size 1 문제

클래스별 가중치는 **fit event만 사용하여** `w_c = N_fit/(2*N_fit_c)`로 계산한다.
자연 비율에서 이 가중치의 평균은 1이고 두 클래스가 기대 loss에 같은 총 비중을 가진다.

현재 event MIL은 batch size가 1이다. PyTorch의
`CrossEntropyLoss(weight=w, reduction='mean')`는 해당 batch의 가중치 합으로 나눠서,
batch size 1의 class-index target에서는 가중치가 상쇄된다.
따라서 구현은 `cross_entropy(..., reduction='none') * w[label]`의 평균으로 한다.
이는 샘플 수로 평균한 class-weighted CE이며 가중치 합으로 정규화한 CE와 구분한다.

이 선택은 양성을 복제한 새 환자를 만드는 방법이 아니다. 소수 양성의 gradient 비중을 바꾼다.
큰 양성 가중치로 학습 변동이 커질 수 있으므로 nonfinite loss를 차단하고 실제 학습 기록을 남긴다.

공식 정의: [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

## 대응과 노출 기록

- 조건마다 fold seed를 다시 설정하여 모델을 초기화한다.
- 실제 초기 state SHA256을 기록하고 세 조건에서 같은지 검사한다.
- sampler에는 별도 torch.Generator를 사용해 초기 모델 RNG와 분리한다.
- 두 natural 조건은 같은 sampler seed와 비복원 규칙을 사용한다.
- patch subsetting과 dropout은 확률적이다. 샘플 순서·반복이 다른 balanced 조건과
  event별 patch draw가 동일하다고 주장하지 않는다.
- 각 epoch의 실제 event draw/label을 training_draws.csv에 저장한다.
- epochs.csv에 total draw, positive draw, unique event를 기록해 과도한 반복 노출을 검토할 수 있다.
- 가중 학습 loss와 일반 학습 loss의 크기를 직접 성능처럼 비교하지 않는다.
- source CSV/split/코드 SHA256과 feature 파일 크기·수정 시각을 protocol에 기록한다.
  feature 자체의 전체 content hash는 계산하지 않는다.
- 완료된 condition/fold는 재사용한다. 설정/입력/코드가 바뀌면 같은 결과 루트 사용을 거부한다.

## 검증과 실행 전 현황

기존 실제 cohort의 25개 outer 환자 분할에서 새 fit/stop 분리를 점검했다.
환자 겹침 없음. fit에는 양성 event 6–11개/양성 환자 6–8명,
stop에는 양성 event 2–4개/양성 환자 2–3명이 들어간다.
실제 서버 WSI 학습은 아직 실행하지 않았다.

로컬 `python tools/test_imbalance_control.py` 최종 실행은 4개 테스트 모두 통과했다.

검증 스크립트는 다음을 확인한다.

- batch size 1에서 가중 CE의 양성 gradient가 실제로 w_positive 배가 되는지.
- 기본 PyTorch weighted mean에서는 같은 gradient가 되는지.
- natural sampler가 전체 event를 한 번씩 보여주는지, 역빈도 sampler가 노출 비중을 바꾸는지.
- fit/stop/test 환자 분리.
- 합성 CPU feature에서 audit → 15번 학습 → 재시작 → 요약 전체 실행.
- 세 조건의 모델 초기 state 일치, natural 조건의 draw 수/양성 노출 수.
- 설정 변경 후 같은 결과 폴더 재사용 차단.

## 서버 실행

프로젝트 루트의 clam_latest 환경에서:

```bash
git pull --ff-only origin main
python tools/test_imbalance_control.py
python tools/run_smc_imbalance_control.py --action audit
mkdir -p results/smc_event_imbalance_control_20261006_exclude25
nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python tools/run_smc_imbalance_control.py --action run --gpu 1 \
  > results/smc_event_imbalance_control_20261006_exclude25/train.log 2>&1 &
```

```bash
tail -n 20 results/smc_event_imbalance_control_20261006_exclude25/train.log
```

완료 후:

```bash
python tools/run_smc_imbalance_control.py --action summarize
```

per_seed_metrics.csv, paired_seed_deltas.csv, seed_summary.csv와 각 fold의
epochs.csv/training_draws.csv/selection.json을 함께 검토한다.
반복 seed는 독립 환자 표본이 아니며 seed 평균/std만으로 유의성을 주장하지 않는다.

## 의사결정 기준

AP가 여러 seed에서 개선되는지, 그 개선이 임계값 0.5의 TP/FP 변화와 어떻게 연결되는지 본다.
natural_ce가 0.5에서 양성을 전혀 부르지 않더라도 AP까지 나쁜지 따로 확인한다.
균형 학습은 출력 확률의 의미와 calibration에도 영향을 줄 수 있으므로
같은 0.5에서의 지표만으로 학습 방식의 우열을 확정하지 않는다.
이득이 불안정하면 추가 조건을 계속 늘리기보다 현행 양성 환자 수의 한계를 기록한다.
