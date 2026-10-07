# 2026-10-07: 평균 풀링 기준선 — 서버 보고 결과

## 자료와 확인 범위

사용자가 서버의 `train.log` 마지막 출력과 작업 종료 알림을 전달했다.
`[COMPLETE] 5 folds per seed`와 `[3] Done`으로 실행 완료가 보고됐다.
현재 로컬에는 이번 결과 원본 CSV가 없어 아래는 **전달된 요약의 해석**이다.
seed별 대응 차이, 확률 분포, 선택된 C 전체와 ensemble은 아직 독립 검산하지 않았다.

- 서버 결과: `results/smc_event_meanpool_control_20261007_exclude25`.
- [사전 계획](2026-10-07-meanpool-baseline-control-plan.md): 575 events, 양성 14, 133 환자, 양성 환자 11명, 1262 slides.
- 동일 5 seed × 5 fold fit/stop/test 환자 분할, UNI v2 40x 특징.
- patch 평균 → slide 동일 가중 평균 → L2 로지스틱 회귀.
- C는 stop log loss로 선택하고 fit+stop 재학습 없음. 판정 임계값 0.5.
- 주 비교는 자연 비율 CE attention. 기존 balanced sampler는 참고 비교.
- 비교는 pooling·분류기·염색 branch·학습 중 patch 선택 등이 함께 달라지는 단순 기준선 대조이며, attention만의 효과를 분리하지 않는다.

## 전달된 성능

5개 seed의 OOF 지표 평균이다. 평균 TP가 소수인 것은 각 seed의 정수 TP를 평균했기 때문이다.
같은 환자를 반복 평가한 것으로, 양성 14건이 70명의 독립 양성으로 늘어난 것은 아니다.

| 조건 | AUROC mean ± std | AP mean ± std | TP | FP | FN | 민감도 | 특이도 | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced sampler | 0.7666 ± 0.0293 | 0.1764 ± 0.0951 | 3.2 | 9.4 | 10.8 | 0.2286 | 0.9832 | 0.2333 | 0.2172 |
| meanpool logreg | 0.7737 ± 0.0428 | 0.2023 ± 0.0855 | 0.4 | 1.2 | 13.6 | 0.0286 | 0.9979 | 0.0485 | 0.0632 |
| natural CE | 0.7511 ± 0.0455 | 0.1645 ± 0.0218 | 1.2 | 4.0 | 12.8 | 0.0857 | 0.9929 | 0.1224 | 0.1337 |
| natural weighted CE | 0.6518 ± 0.0480 | 0.1387 ± 0.0712 | 1.4 | 4.4 | 12.6 | 0.1000 | 0.9922 | 0.1458 | 0.1539 |

## 잠정 해석

1. **단순 기준선에도 분류에 활용할 정보가 남아 있다.** 평균 AP는 natural CE보다 0.03775, AUROC는 0.02259 높다. 다만 seed별 대응 결과가 없으므로 일관된 우월성이나 통계적 차이를 확정하지 않는다. AP 표준편차는 meanpool 0.08548, natural CE 0.02181로 서로 다르다.
2. **임계값 0.5에서는 검출이 매우 부족하다.** 양성 14건 중 평균 0.4건만 검출하고 13.6건을 놓친다. 높은 특이도와 낮은 F1을 함께 보여야 한다. 0.5 판정 성능과 점수 순위 성능은 구분한다.
3. 평균 AP가 높고 F1이 낮다는 결과만으로 calibration 오류 또는 임계값만의 문제라고 단정하지 않는다. 확률 분포와 seed별 양성 순위를 먼저 확인해야 한다. 평균 풀링의 병변 희석 가능성도 이번 표만으로 입증되지 않는다.
4. 마지막 두 fold의 C=0.01/0.001은 전체 25개 선택 분포를 대표하지 않는다. 강한 규제가 일반적으로 선택됐다고 결론내리지 않는다.
5. 더 복잡한 모델 확장은 이 결과만으로 정당화되지 않는다. 작은 기준선을 유지하며 기존 attention과 대응 비교를 마친다. 이전 다른 선택 절차의 AP와 직접 우열을 비교하지 않는다.

## 다음 분석

새 학습 전에 다음 원본을 공유하여 독립 검산한다.

- `per_seed_metrics.csv`, `paired_deltas_vs_natural_ce.csv`: AP/AUROC의 5개 대응 차이, 개선 seed 수, 특정 seed에 대한 평균 민감도.
- seed별 `meanpool_logreg_oof_predictions.csv`: 양성 순위와 점수 분포, 0.5 이상 예측 수, event/patient/label/fold 일치.
- fold별 `outer_predictions.csv`, `partitions.csv`, `candidates.csv`, `selection.json`: 분할·C 선택·OOF 병합 확인.
- `ensemble_metrics.csv`와 ensemble 예측: seed 평균과 분리하여 보고.
- `protocol.json`, `embedding_audit.csv`, `embedding_cache.json`: 계획·입력·요약 기록 확인. feature 평균과 모델 계수 NPZ는 서버에 유지.

OOF 전체에서 F1이 가장 높은 임계값을 골라 독립 성능처럼 보고하지 않는다.
임계값 대조를 확장한다면 기존 fit/stop/test 분리를 보존하고 선택용 자료를 사용해야 한다.
원본을 확인하기 전 재학습이나 새로운 cutoff 탐색을 시작하지 않는다.

## 터미널 상태

`tail -f`의 Ctrl+C는 로그 보기를 종료했다. 평균 풀링 작업은 `Done`으로 표시됐다.
별도 `[2]+ Stopped ... run_aug_compare.sh ...`는 다른 작업이 정지된 상태이며 완료 알림이 아니다.
어떤 작업이 왜 정지됐는지는 이 출력만으로 알 수 없다. `jobs -l`과 해당 job의 상태를 확인하고,
계속 실행할 목적의 작업이면 동일 셸에서 `bg %2`로 재개할 수 있다. 자동으로 재개하지 않았다.
