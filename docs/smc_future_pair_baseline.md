# SMC future-biopsy pair baseline

This exploratory task predicts the significant-rejection state of a later biopsy
from an earlier at-risk biopsy image and the interval between the two biopsies.

## Definition

- Source: a significant-rejection-negative event before the patient's first
  significant rejection.
- Target: every later observed biopsy event for that source.
- Positive target: ACR >=2R or pAMR >=1 at the target event.
- Time input: `log(1 + delta_days)`.
- Split: patient-grouped outer CV; all pairs from one patient stay in one fold.
- Weighting: each patient initially has total weight one, followed by fold-local
  pair-label mass balancing for model fitting.

The runner compares three prespecified baselines:

1. `time_only`
2. `image_only`
3. `image_plus_time`

Image features are deliberately simple for this first baseline: UNI patch features
are averaged within each slide and then averaged equally across slides in the same
source event. Logistic regression regularization and the operating threshold are
selected using patient-grouped inner CV only.

## Build pairs

```bash
cd /home/jupyter/image_team/projects/SMC_MIL

DATA_ROOT=/home/jupyter/data/image_team
PAIR_ROOT="$DATA_ROOT/labels/derived/future_significant_pair_gold_v1"
GOLD_CSV=dataset_csv/smc_significant_rejection_binary.csv

python build_smc_future_pair_task.py \
  --label-xlsx "$DATA_ROOT/labels/raw/WSI_LABEL_ID_MATCH_20260814.xlsx" \
  --gold-csv "$GOLD_CSV" \
  --output-dir "$PAIR_ROOT" \
  --folds 3 \
  --seed 1
```

Inspect the generated cohort before training:

```bash
cat "$PAIR_ROOT/splits/future_pair_fold_summary_3.csv"
python - <<'PY'
import pandas as pd
p = "/home/jupyter/data/image_team/labels/derived/future_significant_pair_gold_v1/future_pair_manifest.csv"
d = pd.read_csv(p)
print(d.groupby("label").agg(pairs=("pair_id", "size"), patients=("case_id", "nunique")))
print(pd.cut(d.delta_days, [0, 30, 90, 180, float("inf")]).value_counts().sort_index())
PY
```

## Run all magnifications

This baseline uses already extracted UNI features and scikit-learn, so it is CPU
work and does not benefit from assigning a GPU. Two workers may be used to reduce
wall-clock time.

```bash
FEATURE_ROOT=/home/jupyter/image_team/projects/SMC_MIL/data/features/uni_v2
mkdir -p results/logs

nohup bash tools/run_smc_future_pair_grid.sh \
  --task-root "$PAIR_ROOT" \
  --gold-csv "$GOLD_CSV" \
  --feature-root "$FEATURE_ROOT" \
  --folds 3 \
  --worker a \
  > results/logs/future_pair_worker_a.log 2>&1 &

nohup bash tools/run_smc_future_pair_grid.sh \
  --task-root "$PAIR_ROOT" \
  --gold-csv "$GOLD_CSV" \
  --feature-root "$FEATURE_ROOT" \
  --folds 3 \
  --worker b \
  > results/logs/future_pair_worker_b.log 2>&1 &
```

Worker A runs 40x and 10x. Worker B runs 20x and 5x.

```bash
tail -f results/logs/future_pair_worker_a.log
tail -f results/logs/future_pair_worker_b.log
```

## Outputs

Each scale writes under
`results/smc_future_pair_<scale>_uni2_meanpool_gold3cv_s1/`:

- `pair_oof_predictions.csv`: every held-out pair.
- `target_event_oof_predictions.csv`: predictions averaged for the same future event.
- `pooled_metrics.csv`: pair and target-event results with patient-cluster bootstrap CIs.
- `horizon_metrics.csv`: pair performance in 1-30, 31-90, 91-180, and >180-day bins.
- `fold_metrics.csv`: held-out fold results.
- `inner_model_selection.csv`: inner-CV C and threshold selection audit.
- `event_embedding_audit.csv`: source event slide and patch counts.

Collect all completed scales:

```bash
python tools/collect_smc_future_pair_results.py \
  --results-root results \
  --output-dir results/smc_future_pair_summary \
  --folds 3
```

Pair rows are correlated and do not create new independent positive patients. Always
report patients, positive patients, target events, and positive target events together
with pair counts. This remains a feasibility analysis, not a confirmatory prognosis
model.
