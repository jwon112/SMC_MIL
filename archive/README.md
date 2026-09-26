# Archived Code

Nothing under this directory is part of the supported SMC workflow. The files
remain in Git so old experiments can be inspected or recovered without
cluttering the active project root.

## Contents

- `legacy_experiments/artifact_synthesis/`: exploratory ArtiDiffuser,
  LatentArtiFusion, and related image-synthesis utilities. These paths are
  isolated from the current SMC training and evaluation commands.
- `upstream_clam/saved_patch_pipeline/`: the original storage-heavy CLAM patch
  extraction path and its legacy guide. The active code uses coordinate-based
  extraction or the project-specific DICOM/MRXS extractors.

The feature-level DDPM denoiser (`train_ddpm_feature.py` and
`models/ddpm_feature.py`) is intentionally not archived because it is still
wired into the CLAM training entry point.
