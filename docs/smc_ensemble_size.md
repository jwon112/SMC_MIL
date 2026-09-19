# Repeated-seed ensemble-size analysis

This analysis uses the existing `single_seed` and `scale_fusion_per_seed`
prediction rows. For each task, cohort, scale or fixed multiscale fusion, it
evaluates every possible subset of the five completed seeds:

- one seed: 5 subsets
- two seeds: 10 subsets
- three seeds: 10 subsets
- four seeds: 5 subsets
- five seeds: 1 subset

No model is selected by its observed score. Each subset uses equal-weight
probability averaging. The output therefore estimates whether performance is
still improving from four to five seeds, before launching five additional
training seeds.

## Run

```bash
cd /home/jupyter/image_team/projects/SMC_MIL

python tools/analyze_smc_ensemble_size.py \
  --predictions-csv results/smc_prediction_ensembles/ensemble_predictions.csv \
  --output-dir results/smc_prediction_ensembles/ensemble_size \
  --seeds 1 11 21 31 41 \
  --threshold 0.5
```

## Outputs

- `ensemble_size_combinations.csv`: metrics for all 31 non-empty seed subsets
- `ensemble_size_summary.csv`: mean, standard deviation, range by ensemble size
- `five_seed_gain_summary.csv`: full five-seed result versus the mean of one- and four-seed subsets

Use AUROC and AUPRC for the seed-count decision. The threshold-dependent
metrics remain exploratory until a threshold is selected within internal OOF
data.
