# Repeated-seed and multiscale prediction ensembles

This analysis uses completed gold-only repeated five-fold experiments. It does
not retrain CLAM or re-extract features.

Two fixed analyses are produced:

1. Average the five seed predictions within each scale.
2. Average probabilities across `40x+20x`, `40x+5x`, and all four scales.

Internal results use one held-out OOF prediction per slide, scale, and seed.
External seed predictions are already five-fold checkpoint ensembles, so the
five-seed average combines 25 checkpoints per scale. Multiscale fusion uses
equal weights fixed before examining its results.

## Run

```bash
cd /home/jupyter/image_team/projects/SMC_MIL

python tools/analyze_smc_prediction_ensembles.py \
  --results-root results \
  --external-root results/gse290577_external/gold5_repeated \
  --output-dir results/smc_prediction_ensembles \
  --folds 5 \
  --seeds 1 11 21 31 41 \
  --threshold 0.5
```

The external root may contain `worker_a` and `worker_b`; prediction files are
found recursively.

## Outputs

- `seed_ensemble_metrics.csv`: five-seed ensemble for every single scale
- `seed_ensemble_gain.csv`: ensemble metrics and change from the five single-seed mean
- `multiscale_ensemble_metrics.csv`: combined seed and multiscale ensembles
- `multiscale_gain_vs_40x.csv`: combined multiscale change from the 40x seed ensemble
- `multiscale_per_seed_gain_vs_40x.csv`: paired scale-fusion change from 40x within each seed
- `multiscale_per_seed_summary.csv`: mean and variation of scale fusion across seeds
- `ensemble_metrics.csv`: all metrics, including each seed's scale fusion
- `ensemble_predictions.csv`: auditable slide-level probabilities

Sensitivity and specificity use the exploratory threshold of 0.5. Model or
fusion selection should primarily use internal OOF results. GSE290577 remains
a secondary external domain-shift analysis.
