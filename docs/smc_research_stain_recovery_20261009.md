# Gold-only research stain recovery

Date: 2026-10-09. These are research-team provisional stain assignments, not a
pathologist-confirmed review. Existing clinical rejection labels remain fixed.

## Decisions and scope

Only seven held slides occur in the current ACR gold source. Assign their original
review numbers as follows; the server joins numbers to its local slide identifiers.

| Original review number | Research stain group | Evidence strength |
|---|---|---|
| 228, 230, 231 | HE | Existing HE, returned HE suspicion, selected original WSI regions compatible with HE |
| 81, 143 | other | BP1B11 group context and returned labels; exact subtype unverified |
| 28, 74 | other | Preview appearance and returned method suspicion; low confidence, AI suggests IHC |

Do not interpret `other` as confirmed Congo red, toluidine blue, or another exact
method. The last two assignments are explicit working assumptions. Confidence is
an evidence category, not a calibrated probability. No new ACR or AMR diagnosis is
made. The 18 held slides outside this ACR gold input stay outside this experiment.

Outputs:

- `gold_stain_labels.csv`: the complete 1,269-slide gold stain table, not a seven-slide subset.
- `stain_decisions_gold7.csv`: seven assignments and evidence, joined to local gold records.
- `restored7/acr_high/`: primary experiment, 578 events / 17 positive / 1,269 slides.
- `restored3/acr_high/`: optional sensitivity input, same 578 events / 17 positive,
  1,265 slides, restoring only the three HE slides and keeping the other four excluded.
- `baseline575/acr_high/`: exact previous input, checked against actual protocol hashes.
- `partitions/`: independent fit/stop/test roles; `partition_audit.csv` summarizes them.
- `pending_non_gold_18.csv`: unresolved cases outside this experiment.

This is a gold-scoped research label table, not the full all-slide curation manifest.
The previous curation files, 215-label update, 25-case exclusion list and results
are preserved. The new experiment uses its own event manifests rather than passing
the old exclusion list again.

## Fixed comparison

The primary run uses the existing attention model with stain branches, observed
presence mask, balanced sampler, 2,048 patches per slide, and the prior stopping
settings (maximum 50 epochs, minimum 10, patience 10). There are five seeds and
five folds: **25 model fits**, with no architecture/loss/threshold search.

The old 575 events retain every saved patient fit/stop/test role. Three new
patients are added with a fixed seeded rule: outer test folds with the fewest
positive patients are filled first, ties are random; each new patient gets one
other stopping fold and three fitting folds. No evaluation predictions determine
their assignments. All seeds test every event exactly once.

Report separately:

1. Paired performance on the **common 575 events**, against saved balanced-sampler
   predictions. This compares the complete recovery procedure, including extra
   training/stopping patients and extra slides; it does not isolate one mechanism.
2. Performance on all **578 events**, with the changed prevalence stated.
3. Predictions for the **three recovered events**, descriptively.

Use AP, AUROC, TP/FP/FN, precision, sensitivity, specificity, F1, F2 and MCC at the
prespecified threshold 0.5. Do not claim improvement by directly comparing AP on
578 events with AP on 575 events. Seeds are repeated fits, not independent cohorts.

## Server commands

Run from the project root in the existing training environment. The code and
number-only specification can be obtained through Git; the generated identifier
tables and review images stay in the research data environment.

```bash
git pull --ff-only origin main
python tools/test_research_stain_partitions.py
python tools/prepare_smc_research_stains.py
python tools/run_smc_research_stains.py --action audit
```

Preparation reads the existing server 575-event manifest, existing gold source,
old 25-case pending CSV and saved imbalance-control partitions. It verifies that
the new table contains exactly the seven held gold slides. Re-running accepts
identical outputs and refuses differing outputs at the same location.

Audit must show `ready_to_train: true`, 578 events, 17 positive events, 136 patients,
14 positive patients, 25 model fits and 1,269 feature files. The feature path defaults
to the actual saved imbalance-control protocol. If necessary, pass the complete
40x UNI-v2 directory using `--feature-dir` to both audit and run. The run refuses
missing bags and checks the seven restored bags have finite 1,536-dimensional features.

```bash
mkdir -p results/smc_event_research_stains_20261009
nohup env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python tools/run_smc_research_stains.py --action run --gpu 1 \
  > results/smc_event_research_stains_20261009/train.log 2>&1 &
tail -f results/smc_event_research_stains_20261009/train.log
```

After completion:

```bash
python tools/run_smc_research_stains.py --action summarize
```

`paired_common575_deltas.csv`, `seed_summary.csv` and `recovered3_predictions.csv`
are the primary review files. Existing results are not overwritten.

The `restored3` input is prepared for a possible follow-up check of the four
non-HE assumptions; it is **not run by default**. Use `--cohorts restored3` with
a separate `--results-root` if that sensitivity check is later needed.

## Local preparation without server manifests

If the baseline event manifest is absent locally, supply `--stain-reference-csv`,
`--previous-decisions-csv` and `--pending-csv` pointing to preserved local records.
The reconstructed baseline must match both actual training-manifest hashes.
Local preparation and partition audit need no feature bags. Actual training
requires the complete server feature collection.
