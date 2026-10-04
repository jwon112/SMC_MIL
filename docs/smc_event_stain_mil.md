# Stain-Aware Event MIL Pilot

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
