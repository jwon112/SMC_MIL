# Stain-Aware Event MIL Pilot

## 현재 실행 기준 (2026-10-04)

서버 라벨 루트는 `/home/jupyter/data/image_team/labels/derived`입니다.

| 목적 | 현재 위치 |
|---|---|
| 전체 stain curation | `wsi_curation_v2_provisional_20260930/slide_curation_manifest_curated.csv` |
| 학습 event 및 환자별 splits | `event_stain_mil_gold_provisional_20260930/{task}/` |
| event 생성 시 제외 목록 | `wsi_stain_provisional_20260930_upload/pending_review_25.csv` |
| provisional 적용 원본 | `wsi_curation_v2_final/slide_curation_manifest_curated.csv` |
| 이전 event baseline | `event_stain_mil_gold_v1/{task}/` |

서버 생성 결과는 575 events / 1,262 slides이며 ACR high 양성은 14 events,
11 patients입니다. 전체 curation 2,668행과 학습 event cohort는 범위가 다릅니다.
현재 사용 stain 판정에는 잠정 검토 결과가 포함돼 있습니다.

현재 실행 진입점은 `tools/run_smc_presence_mask_control.sh`와
`tools/run_smc_patch_sampling_control.py`입니다. 이전 ablation 결과는
`results/smc_event_stain_provisional_20260930_exclude25`에 있습니다.
두 대조실험의 완료 여부는 각 결과 파일로 확인합니다.

feature는 프로젝트 `data/features/uni_v2` 경로를 사용합니다. 실제 위치와의
심볼릭 링크 관계는 서버 점검 결과를 확인하며 별도 복사본이라고 단정하지 않습니다.
아래 초기 pilot 및 `v2_final` 예제는 이전 실험 재현 절차로 보존합니다.

## Original pilot and baseline protocol

The prediction unit is one pathology event. Its known-stain slides are retained as H&E, IHC, or other. Patch features are pooled into a slide embedding with shared gated attention; slide embeddings are pooled within each stain branch; the three branch embeddings and branch-presence masks feed an event classifier.

The gold-only experiment uses 40x (`l0_0p25mpp_40x` features), five folds, and seeds 1/11/21/31/41. Existing seed-specific patient-grouped slide folds are mapped to event IDs, so no event or patient crosses validation folds. Unknown-stain slides are excluded. This is a new architecture and should be compared with the established slide-level CLAM baseline as exploratory work.

Presence masks can encode staining-order practice. Follow with an H&E-only event ablation and a no-mask ablation before treating a gain as morphology-driven.

For repeated CV, summarize `oof_predictions.csv` from all five seeds with `tools/summarize_smc_event_stain_mil.py`. It averages the five out-of-fold probabilities for each event; it does not count five predictions of one event as independent observations.

## Event-level ablation and scale grid

`train_smc_event_stain_mil.py --stain-mode agnostic` removes the stain-specific
branches. All slides in an event share one projection and one slide-attention
pool. `--no-presence-mask` retains the three stain branches but removes the
three indicators that reveal which stain types were ordered for an event. This
separates morphology from potential staining-order leakage. Old stain-aware
checkpoints remain compatible because both new model options default to the
original architecture when absent from their saved configuration.

Run the 40x stain ablation first, then the remaining stain-aware scales. The
grid runner skips a combination only when all requested checkpoints and its OOF
prediction file exist.

```bash
EVENT_ROOT=/home/jupyter/data/image_team/labels/derived/event_stain_mil_gold_v1
FEATURE_ROOT=/home/jupyter/image_team/projects/SMC_MIL/data/features/uni_v2
RESULT_ROOT=results/smc_event_stain_gold5x5

bash tools/run_smc_event_ablation_grid.sh \
  --gpu 1 --worker acr \
  --feature-root "$FEATURE_ROOT" --manifest-root "$EVENT_ROOT" \
  --results-root "$RESULT_ROOT" \
  --modes aware_nomask agnostic --scales 40x

bash tools/run_smc_event_ablation_grid.sh \
  --gpu 1 --worker acr \
  --feature-root "$FEATURE_ROOT" --manifest-root "$EVENT_ROOT" \
  --results-root "$RESULT_ROOT" \
  --modes aware --scales 20x 10x 5x
```

Use `--worker amr_significant --gpu 3` for the second worker. Summarize all
completed event models and create the slide-CLAM-to-event baseline as follows:

```bash
python tools/summarize_smc_event_stain_mil.py \
  --results-root "$RESULT_ROOT" \
  --output-dir results/smc_event_ablation_summary \
  --scales 40x 20x 10x 5x --modes aware

python tools/summarize_smc_slide_to_event_baseline.py \
  --results-root results --manifest-root "$EVENT_ROOT" \
  --output-dir results/smc_slide_to_event_summary \
  --scales 40x 20x 10x 5x
```

For the 40x three-way comparison, rerun the event summarizer with
`--scales 40x --modes aware aware_nomask agnostic`, then combine its CSV with the slide
baseline using `tools/compare_smc_event_ablation.py`.

## Uncertain-stain sensitivity set

The 240-slide high-resolution review manifest can be excluded without changing
the source gold labels or patient split definitions. Write this variant to a
separate manifest root and run the 40x stain-aware grid against that root.

```bash
REVIEW_240=/home/jupyter/data/image_team/labels/derived/stain_review_highres_240/highres_thumbnail_manifest.csv
EXCLUDED_ROOT=/home/jupyter/data/image_team/labels/derived/event_stain_mil_gold_v1_exclude_review240

python build_smc_event_stain_manifest.py \
  --label-dir dataset_csv \
  --curation-manifest /home/jupyter/data/image_team/labels/derived/wsi_curation_v2_final/slide_curation_manifest_curated.csv \
  --split-root splits --output-root "$EXCLUDED_ROOT" \
  --folds 5 --seeds 1 11 21 31 41 \
  --exclude-slide-ids-csv "$REVIEW_240"
```

Events that lose every usable slide are omitted and reported in the new
manifest counts. Final comparisons should restrict both full and excluded
predictions to their common event IDs, because a raw metric difference would
otherwise mix model sensitivity with a changed evaluation population.

## Completed-control reanalysis (2026-10-05)

Completed OOF predictions and historical event/slide runs were re-evaluated with
positive F1, precision, MCC, F2, NPV, Brier/log loss and paired patient-cluster
intervals. See the [dated research note](../experiment_notes/2026-10-05-event-controls-multimetric-reanalysis.md)
and `results/smc_metric_review_20261005/`. Run `python tools/reanalyze_smc_event_controls.py`
and `python tools/plot_smc_metric_review.py` to reproduce the review.
Patch 4096 improved ensemble AP but reduced AP in four of five paired seeds;
mask removal did not consistently improve the matched architecture comparison.
Retain 2048 as the reference cap and defer a final mask decision. Intervals hold
fitted OOF predictions fixed and exclude retraining/model-selection uncertainty.

## Evaluation metrics (2026-10-05)

The event and patch-control summaries report positive-class `f1`, `precision`,
`mcc`, `tp`, `fp`, `tn`, `fn`, and the prediction `threshold`, alongside AUROC,
PR-AUC (average precision), sensitivity, specificity and balanced accuracy.
The default threshold remains 0.5. Presence summaries include per-seed mean/std
for precision, F1 and MCC; patch summaries include these in per_seed_metrics.csv
and ensemble_summary.csv and the printed mean/std table.
F1 is `2*TP/(2*TP+FP+FN)`. It describes the positive class, not a weighted
average dominated by negative events. Precision is NaN when no events are
predicted positive; F1 is zero if actual positives are all missed.
These threshold-dependent metrics complement PR-AUC; they do not replace it.
Do not select a threshold by maximizing F1 on held-out OOF labels and then
report that optimized value as unbiased evaluation. Choose a threshold using
training/inner validation data or a prespecified operating rule.
With 14 positive events, one additional detected positive changes sensitivity
by 1/14. Report confusion counts and paired seed results when comparing models.
Existing OOF files can be summarized again; no model retraining is needed.

## Same-architecture presence-mask control (2026-10-04)

`aware_zero_mask` retains the exact aware architecture, including three mask
input columns, but supplies constant zeros in those columns. Zero embeddings
for missing stains remain in both conditions; this tests the additional explicit
presence signal, not removal of all missingness information.

Run both `aware` and `aware_zero_mask` in a new results root with
`--paired-fold-seeding`. It resets Python, NumPy and Torch RNGs at the start of
each fold using `seed + 100003 * fold`, so early stopping in a preceding fold
cannot change the next fold's initial weights. Both conditions have the same
parameter shapes, initialization, slides, splits, patch sampling protocol and
training settings. Epoch counts can still differ through validation-loss early
stopping. Constant-zero mask columns receive no data gradient (optimizer weight
decay may still affect their unused weights).

```bash
bash tools/run_smc_presence_mask_control.sh --gpu 1 --worker acr
```

This wrapper runs 40x, seeds 1/11/21/31/41, two modes, and uses the 575-event
provisional exclude25 manifests. It writes to a new `presence_control` results
root. Prior aware/nomask results were not generated with paired fold seeding and
must not serve as the matched control. Checkpoint model_config stores
`presence_mask_values`; older checkpoints default to observed values.

```bash
python tools/summarize_smc_event_stain_mil.py \
  --results-root results/smc_event_presence_control_20261004_exclude25 \
  --output-dir results/smc_event_presence_control_20261004_exclude25/acr_40x_summary \
  --tasks acr_high --scales 40x --modes aware aware_zero_mask \
  --seeds 1 11 21 31 41
```

## Nested threshold control (2026-10-05)

Run `python tools/run_smc_threshold_control.py --action audit`, then `--action run --gpu 1`.
The new control keeps the observed mask and patch cap 2048. Within each outer
training set, three patient-grouped calibration folds each have a separate
early-stopping patient subset. Inner cross-fitted scores select F1/F2 thresholds;
the median inner best epoch fixes the full outer-training refit duration.
Outer patients select neither checkpoints nor thresholds. Fixed 0.5, inner F1,
and inner F2 use identical outer predictions. F2 is the prespecified primary
comparison, with extra false positives reported alongside recovered positives.

Five seeds require 100 model fits. Completed outer folds support resume; changed
inputs or code require a new result root. Existing outer-validation-selected
checkpoints are not used for this experiment. Results contain paired seed deltas,
fold-specific thresholds and changed calls; no pooled cutoff is fitted to outer OOF.

Execution and interpretation details:
[threshold control research plan](../experiment_notes/2026-10-05-nested-threshold-control-plan.md).
