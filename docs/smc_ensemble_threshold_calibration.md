# Ensemble threshold calibration

This tool selects operating thresholds from internal OOF ensemble predictions
only, then applies the unchanged thresholds to GSE290577. It evaluates all
single-scale and fixed multiscale ensembles, while marking these internal-OOF
primary candidates:

- high-grade ACR: 40x seed ensemble
- AMR positive: four-scale seed-and-scale ensemble
- significant rejection: 40x seed ensemble

Threshold rules are:

- `fixed_0p5`: existing exploratory threshold
- `youden_internal_oof`: maximum sensitivity + specificity - 1 on internal OOF
- `target_sensitivity_80pct` and `target_sensitivity_90pct`: highest internal
  OOF threshold that attains the target sensitivity

The internal calibration metrics are apparent OOF operating-point estimates,
because the same OOF predictions choose the threshold. The external transfer
table is the appropriate check of threshold portability.

## Run

```bash
cd /home/jupyter/image_team/projects/SMC_MIL

python tools/calibrate_smc_ensemble_thresholds.py \
  --predictions-csv results/smc_prediction_ensembles/ensemble_predictions.csv \
  --output-dir results/smc_prediction_ensembles/threshold_calibration \
  --target-sensitivity 0.80 0.90 \
  --baseline-threshold 0.5
```

## Outputs

- `primary_candidate_thresholds.csv`: internal OOF calibration for the three primary candidates
- `primary_candidate_external_transfer.csv`: GSE transfer for the same candidates
- `internal_threshold_calibration.csv`: all evaluated ensemble configurations
- `external_threshold_transfer.csv`: GSE transfer for all configurations
