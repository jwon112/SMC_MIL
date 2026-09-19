# Stain-Aware Event MIL Pilot

The prediction unit is one pathology event. Its known-stain slides are retained as H&E, IHC, or other. Patch features are pooled into a slide embedding with shared gated attention; slide embeddings are pooled within each stain branch; the three branch embeddings and branch-presence masks feed an event classifier.

The first run is gold-only, 40x, seed 1, five-fold. Existing patient-grouped slide folds are mapped to event IDs, so no event or patient crosses validation folds. Unknown-stain slides are excluded. This is a structural pilot and should initially be compared with the established slide-level CLAM baseline as exploratory work.

Presence masks can encode staining-order practice. Follow with an H&E-only event ablation and a no-mask ablation before treating a gain as morphology-driven.
