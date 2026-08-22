# SMC CV Metric Analysis

For imbalanced binary tasks, do not interpret accuracy alone.  After all
outer-fold experiments have completed, run the following from the repository
root on the server:

```bash
python analyze_smc_cv_metrics.py \
  --results-root results \
  --output-dir results/smc_cv_comparison
```

The command reads each experiment's `split_<fold>_results.pkl` and writes:

- `oof_bag_predictions.csv`: probability for every held-out bag.
- `fold_imbalance_metrics.csv`: metrics for each outer fold and pooled OOF
  predictions.
- `averaged_imbalance_metrics.csv`: outer-fold mean and standard deviation,
  plus pooled OOF metrics.

Reported metrics are AUROC, average precision (PR-AUC), balanced accuracy,
sensitivity, specificity, precision, F1, MCC, and the confusion counts at a
fixed probability threshold of 0.5.  Threshold selection on the held-out test
fold is intentionally avoided because it would introduce evaluation leakage.

All output metrics remain **bag/slide-level**.  The split generation keeps a
patient's bags in one outer fold, but patients with more bags still contribute
more heavily.  Clinical reporting should therefore add a separately defined
event- or patient-level probability aggregation before calculating final
metrics.
