# Experiment Notes

This directory is the chronological research record for this project. It
captures what was attempted, observed, interpreted, and decided during an
experiment. It is intentionally separate from the project-wide literature
repository in [`../reference/`](../reference/README.md).

## Scope

Store here:

- experiment plans and dated amendments;
- dataset, split, model, and evaluation settings actually used;
- result summaries and links to local artifacts;
- failures, deviations, interpretation, and follow-up decisions;
- concise execution history when it explains a scientific or reproducibility
  decision.

Do not store here:

- source papers or extracted paper text;
- reusable literature summaries and evidence indexes;
- stable pipeline instructions that belong in [`../docs/`](../docs/);
- large model checkpoints, feature bags, logs, or generated result trees.

An experiment note may cite a `paper_id`, `claim_id`, or `atom_id` from
`../reference/`, but it should remain readable without loading the entire
literature store.

## Recent records

- [2026-10-05: nested patient-grouped threshold control plan](2026-10-05-nested-threshold-control-plan.md)
- [2026-10-05: event controls and historical experiments, multimetric reanalysis](2026-10-05-event-controls-multimetric-reanalysis.md)

## Naming

Use a date-prefixed, descriptive filename:

```text
YYYY-MM-DD-short-experiment-title.md
YYYY-MM-DD-to-YYYY-MM-DD-multi-day-study.md
```

Start new records from [`TEMPLATE.md`](TEMPLATE.md). The template is guidance,
not a form that must be filled mechanically. Omit sections that do not add
useful information.

## Record policy

1. Record the question and primary evaluation target before interpreting the
   result whenever possible.
2. Preserve meaningful negative results and protocol deviations.
3. After a conclusion has been used downstream, amend or supersede it instead
   of silently rewriting its history.
4. Distinguish observed results from interpretation and future hypotheses.
5. Link reproducibility anchors: Git commit, task/split identifier, config,
   output path, and key artifact hash when practical.
6. Do not treat queue status, a PID, or a running log as scientific evidence.

## Relationship to other directories

| Directory | Role |
|---|---|
| `docs/` | Stable instructions for operating the code and data pipeline |
| `experiment_notes/` | Dated records of experiments, observations, and decisions |
| `reference/` | Reusable literature sources, evidence units, and retrieval indexes |
| `results/` | Generated artifacts; reviewed tables/figures/configs are shared through Git, weights remain on the server |
