# Stain-aware versus agnostic attention after recovery

Date: 2026-10-10. This control uses the existing research provisional stain labels.
It does not validate those labels against pathology review.

## Fixed comparison

- Identical 578 events, 17 positives, 1,269 slides and saved fit/stop/test patients.
- Reuse all 25 completed `restored7` aware fits in `smc_event_research_stains_20261009`.
- Train only the 25 agnostic fits: five seeds and five folds.
- Aware: three stain branches and the observed three-component presence mask.
- Agnostic: one shared branch; stain names do not route slides or enter the classifier.
  As in the earlier agnostic experiment, retain one constant all-slides-present input.
  Every event has slides, so this input carries no stain identity.
- Both use the same UNI-v2 features, hidden dimension 128, dropout .25, patch cap
  2,048, balanced replacement sampling, unweighted CE, Adam 2e-4/1e-5, and saved
  epoch settings. Early stopping uses the same independent stop patients.
- Reproduce the saved aware initialization and verify identical initial shared
  patch-encoder weights. Whole-model weights and parameter counts differ because
  the branch and classifier structures differ.
- Event sampling sequences must match for every epoch shared by the two fits.
  Patch sampling and dropout draws can differ because architecture changes consume
  different random-number sequences. Exact patch draws are not claimed to match.
- Each mode selects its own checkpoint using stop CE; outer predictions never
  select checkpoints or decision thresholds.

This tests the complete stain-aware architecture against the agnostic architecture.
It does not isolate the mask alone or control for model capacity. Prior mask-only
controls on the smaller cohort remain separate evidence.

## Server execution

Run from the project root in the existing training environment. No label
regeneration is needed. The completed aware results and feature files must remain
at their recorded locations; the control verifies their input/code hashes and
feature membership, sizes and modification times.

```bash
git pull --ff-only origin main
python tools/test_recovered_stain_control.py
python tools/run_smc_recovered_stain_control.py --action audit
```

Proceed when audit shows `ready_to_train: true`, 578 events, 17 positives, 1,269
feature files, 25 new fits and 25 reused aware fits.

```bash
mkdir -p results/smc_event_stain_control_20261010
nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python tools/run_smc_recovered_stain_control.py --action run --gpu 1 \
  > results/smc_event_stain_control_20261010/train.log 2>&1 &
tail -f results/smc_event_stain_control_20261010/train.log
```

After completion, summaries are generated automatically. To regenerate them:

```bash
python tools/run_smc_recovered_stain_control.py --action summarize
```

The original recovery trainer and model files are unchanged, so their completed
protocol remains valid. Outputs are separate and cannot replace the aware root.
Existing control fits can resume only with the same frozen protocol.

## Evaluation

- `seed_summary.csv`: aware and agnostic on the same full 578-event and common
  575-event evaluation sets; seed means and standard deviations.
- `paired_mode_deltas.csv`: **agnostic minus aware**, per seed and evaluation set.
  Positive AUROC/AP/F1 differences favor agnostic; positive FP differences mean
  more false positives, not an improvement.
- `recovered3_predictions.csv`: both models' predictions for the three recovered
  events, reported descriptively.
- `per_seed_metrics.csv`: raw confusion counts, AUROC, AP, precision, sensitivity,
  specificity, F1, F2 and MCC at the prespecified 0.5 threshold.

Judge direction and consistency across seeds, not only the best seed or the
mean. Repeated seeds and the three recovered cases are not independent cohorts.
Generated files contain identifiers and remain in the research data environment;
only generic code, tests and this guide are intended for public Git sharing.
