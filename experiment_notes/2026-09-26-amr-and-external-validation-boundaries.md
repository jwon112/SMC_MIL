---
date: 2026-09-26
title: "AMR input scope and external-validation terminology"
status: decided
tags: [amr, pamr, external-validation, reporting, risk-of-bias]
related_papers:
  - colvin2015_aha_amr
  - amancherla2025_spatial_transcriptomics
  - seraphin2023_cardiac_ssl
  - collins2024_tripod_ai
  - moons2025_probast_ai
related_claims:
  - claim-amr-multimodal
  - claim-gse290577-role
  - claim-external-performance-varies
  - claim-tripod-reporting
  - claim-probast-appraisal
  - claim-patient-cluster-reporting
supersedes: []
---

# AMR input scope and external-validation terminology

## Question

Clarify three interpretation rules before they are used in experiment plans or
manuscripts:

1. What can an H&E-only model claim about pAMR?
2. What counts as a routine-WSI external-validation cohort?
3. How should TRIPOD+AI and PROBAST+AI be used in this project?

## Work performed

Reviewed the already registered evidence atoms for cardiac AMR nomenclature,
GSE290577, multicenter cardiac-rejection WSI studies, TRIPOD+AI, and
PROBAST+AI. No model experiment was run for this note.

## Observations

- The pAMR categories combine histopathologic and immunopathologic findings.
  H&E shows morphology but does not directly provide C4d/CD68 or other
  immunopathologic results.
- Consequently, an H&E-only input cannot directly observe the distinction
  between morphology-negative/immunopathology-negative and
  morphology-negative/immunopathology-positive cases, or between
  morphology-positive/immunopathology-negative and both-positive cases.
- GSE290577 contains selected tissue-microarray cores with spatial
  transcriptomics and registered H&E images. It is not a consecutive cohort of
  routine full-slide endomyocardial-biopsy WSIs sampled for the intended
  clinical screening workflow.
- Published cardiac-rejection studies show that independent-cohort performance
  can differ from internal cross-validation, reinforcing the need to keep
  development and external evaluation distinct.
- TRIPOD+AI specifies what a clinical prediction-model study should report.
  PROBAST+AI evaluates development quality, evaluation risk of bias, and
  applicability.

## Interpretation

- An H&E-only model may predict AMR or reference pAMR labels and may serve as a
  morphology-based screening model. This does not establish equivalence to the
  complete pAMR diagnostic process because part of the defining input is
  unobserved.
- “Routine-WSI external validation” is reserved for a genuinely independent,
  intended-use-matched cohort of full clinical WSIs evaluated after freezing
  preprocessing, model parameters, calibration, and decision thresholds.
- A selected tissue-core resource can support biological interpretation,
  multimodal analysis, or a core-level task. It must not be reported as
  external validation of a routine full-slide model unless the target task and
  sampling frame are changed accordingly.
- TRIPOD+AI and PROBAST+AI are complementary: the former governs transparent
  reporting; the latter governs critical appraisal of bias and applicability.

## Decision and follow-up

- Decision: Describe H&E-only outputs as “prediction of AMR/pAMR labels from
  H&E morphology” or “AMR risk screening,” not as complete pAMR diagnosis.
- Decision: Do not count GSE290577 as routine-WSI external validation for the
  current slide-level pipeline. Use it only for explicitly named biological,
  multimodal, or core-level analyses.
- Decision: Evaluate all repeated-biopsy models with patient-level separation,
  frozen external evaluation, and explicit leakage controls.
- Decision: Use TRIPOD+AI as the reporting checklist and PROBAST+AI as the
  design/evaluation risk-of-bias checklist.
- Next action: Carry these wording and evaluation rules into future experiment
  notes, protocol amendments, result summaries, and manuscripts.

## Deviations and limitations

- This note records terminology and evidence boundaries; it does not select a
  final AMR label schema or clinical decision threshold.
- It does not rule out useful H&E-based AMR prediction or future validation of
  a tissue-core model.
- It does not establish that any current architecture is clinically deployable
  or locally transportable.

## Reference links

- `paper_id: colvin2015_aha_amr`
- `paper_id: amancherla2025_spatial_transcriptomics`
- `paper_id: seraphin2023_cardiac_ssl`
- `paper_id: collins2024_tripod_ai`
- `paper_id: moons2025_probast_ai`
- `claim_id: claim-amr-multimodal`
- `claim_id: claim-gse290577-role`
- `claim_id: claim-external-performance-varies`
- `claim_id: claim-tripod-reporting`
- `claim_id: claim-probast-appraisal`
- `claim_id: claim-patient-cluster-reporting`
- `atom_id: atom-colvin-pamr`
- `atom_id: atom-amancherla-gse-role`
- `atom_id: atom-seraphin-external-gap`
- `atom_id: atom-collins-tripod-reporting`
- `atom_id: atom-moons-probast-appraisal`
- `atom_id: atom-moons-leakage-appraisal`
