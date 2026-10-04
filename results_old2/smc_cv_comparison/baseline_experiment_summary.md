# SMC CLAM Baseline Summary

## Scope

This document summarizes the completed 16-experiment baseline grid in this
directory. It is the **early-stopping baseline**, not the subsequent
full-outer-training, 100-epoch cosine-schedule protocol.

All experiments used UNI2-h embeddings (1536 dimensions), CLAM-SB, dropout
0.25, weighted bag sampling, bag cross-entropy, instance SVM loss, and four
separate image scales:

| Scale | Pyramid level | Approximate resolution |
|---|---:|---:|
| L0 | 0 | 0.25 um/px (40x) |
| L1 | 1 | 0.50 um/px (20x) |
| L2 | 2 | 1.00 um/px (10x) |
| L3 | 3 | 2.00 um/px (5x) |

## Cross-validation Protocol

- Patient-grouped outer 3-fold CV: every patient appears in one held-out
  outer fold exactly once.
- Each outer training set was split again into 80% train and 20% validation
  for early stopping.
- Therefore, per fold the approximate patient allocation was 53% train, 14%
  inner validation, and 33% outer held-out evaluation.
- There is no separate, fixed final test cohort. Existing `test_*` columns
  mean outer-CV held-out performance.
- Metrics are currently **WSI bag-level**, although folds are patient-grouped.

## Task Sample Counts

| Task | Negative bags | Positive bags | Negative events | Positive events | Split-negative patients | Split-positive patients |
|---|---:|---:|---:|---:|---:|---:|
| ACR 0R vs 1R/2R/3R | 873 | 396 | 409 | 169 | 54 | 82 |
| ACR 0R/1R vs 2R/3R | 1,229 | 40 | 561 | 17 | 122 | 14 |
| AMR pAMR0 vs positive | 1,198 | 71 | 559 | 19 | 122 | 14 |
| Any rejection | 816 | 453 | 394 | 184 | 51 | 85 |

The dataset contains 1,269 WSI bags from 578 pathology events and 136 unique
patients. Split-level patient class is the maximum event label for that patient.

## Baseline Results

Values are outer-fold mean plus/minus standard deviation. PR-AUC is reported
as average precision. Sensitivity, specificity, F1, and MCC use the fixed
probability threshold of 0.5.

| Task | Scale | AUROC | PR-AUC | Balanced accuracy | Sensitivity | Specificity | F1 | MCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ACR 0R vs 1R/2R/3R | L0 | 0.650 +/- 0.008 | 0.451 +/- 0.061 | 0.586 +/- 0.035 | 0.383 +/- 0.232 | 0.790 +/- 0.163 | 0.380 +/- 0.151 | 0.198 +/- 0.013 |
|  | L1 | 0.637 +/- 0.021 | 0.435 +/- 0.052 | 0.577 +/- 0.029 | 0.328 +/- 0.172 | 0.825 +/- 0.132 | 0.359 +/- 0.124 | 0.187 +/- 0.044 |
|  | L2 | 0.588 +/- 0.018 | 0.395 +/- 0.028 | 0.551 +/- 0.022 | 0.273 +/- 0.169 | 0.829 +/- 0.131 | 0.304 +/- 0.079 | 0.132 +/- 0.041 |
|  | L3 | 0.563 +/- 0.028 | 0.375 +/- 0.050 | 0.513 +/- 0.034 | 0.179 +/- 0.182 | 0.848 +/- 0.137 | 0.197 +/- 0.196 | 0.041 +/- 0.057 |
| ACR 0R/1R vs 2R/3R | L0 | 0.736 +/- 0.222 | 0.190 +/- 0.117 | 0.581 +/- 0.078 | 0.190 +/- 0.171 | 0.973 +/- 0.017 | 0.164 +/- 0.143 | 0.140 +/- 0.144 |
|  | L1 | 0.686 +/- 0.148 | 0.143 +/- 0.108 | 0.568 +/- 0.078 | 0.181 +/- 0.133 | 0.955 +/- 0.041 | 0.170 +/- 0.151 | 0.142 +/- 0.132 |
|  | L2 | 0.713 +/- 0.068 | 0.198 +/- 0.121 | 0.616 +/- 0.083 | 0.248 +/- 0.163 | 0.984 +/- 0.009 | 0.284 +/- 0.182 | 0.270 +/- 0.169 |
|  | L3 | 0.707 +/- 0.082 | 0.160 +/- 0.092 | 0.560 +/- 0.063 | 0.142 +/- 0.128 | 0.977 +/- 0.009 | 0.151 +/- 0.134 | 0.127 +/- 0.129 |
| AMR pAMR0 vs positive | L0 | 0.638 +/- 0.012 | 0.107 +/- 0.003 | 0.561 +/- 0.025 | 0.180 +/- 0.081 | 0.942 +/- 0.032 | 0.161 +/- 0.071 | 0.115 +/- 0.060 |
|  | L1 | 0.673 +/- 0.026 | 0.127 +/- 0.054 | 0.531 +/- 0.047 | 0.116 +/- 0.098 | 0.947 +/- 0.005 | 0.110 +/- 0.095 | 0.057 +/- 0.085 |
|  | L2 | 0.639 +/- 0.065 | 0.135 +/- 0.071 | 0.555 +/- 0.026 | 0.156 +/- 0.029 | 0.954 +/- 0.025 | 0.170 +/- 0.024 | 0.128 +/- 0.023 |
|  | L3 | 0.572 +/- 0.021 | 0.112 +/- 0.038 | 0.543 +/- 0.053 | 0.143 +/- 0.094 | 0.943 +/- 0.016 | 0.140 +/- 0.089 | 0.087 +/- 0.093 |
| Any rejection | L0 | 0.650 +/- 0.016 | 0.511 +/- 0.037 | 0.609 +/- 0.024 | 0.430 +/- 0.104 | 0.788 +/- 0.092 | 0.474 +/- 0.067 | 0.235 +/- 0.032 |
|  | L1 | 0.644 +/- 0.039 | 0.503 +/- 0.036 | 0.606 +/- 0.019 | 0.413 +/- 0.062 | 0.798 +/- 0.094 | 0.468 +/- 0.033 | 0.232 +/- 0.027 |
|  | L2 | 0.622 +/- 0.064 | 0.494 +/- 0.057 | 0.586 +/- 0.050 | 0.405 +/- 0.092 | 0.768 +/- 0.192 | 0.451 +/- 0.042 | 0.201 +/- 0.066 |
|  | L3 | 0.621 +/- 0.076 | 0.479 +/- 0.042 | 0.594 +/- 0.034 | 0.425 +/- 0.082 | 0.763 +/- 0.124 | 0.464 +/- 0.035 | 0.205 +/- 0.044 |

## Interpretation

- **ACR 0R vs 1R/2R/3R:** L0 is the most consistent baseline. Performance
  declines with lower magnification, suggesting useful high-resolution signal.
- **High-grade ACR:** L2 has the strongest threshold metrics, while L0 has the
  largest mean AUROC but high fold variance. Only 14 positive patients and no
  3R bags are available; this is exploratory only.
- **AMR:** L1 has the highest AUROC, while L2 has stronger PR-AUC, F1, and MCC
  at threshold 0.5. The discrepancy indicates that 0.5 is not an established
  clinical operating threshold.
- **Any rejection:** L0 and L1 are close; adding AMR to ACR does not yet show a
  clear AUROC improvement over ACR 0R vs 1R/2R/3R.

## Limitations And Next Protocol

- Accuracy is not a primary metric for high-grade ACR or AMR because of severe
  class imbalance.
- A threshold must be chosen without using the outer held-out fold. This
  baseline uses 0.5 only as a descriptive reference.
- Final clinical reporting should aggregate WSI probabilities to pathology
  event and, where appropriate, patient level before calculating metrics.
- The next protocol is full outer-fold training: all two training folds are
  used, no inner validation or early stopping, and the last model is evaluated
  after a pre-specified 100 epochs with cosine learning-rate decay.
