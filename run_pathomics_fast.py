#!/usr/bin/env python3
"""Opt-in acceleration with unchanged originals, parity gate and audit trail."""
from __future__ import annotations

# Import launcher FIRST so thread limits precede numerical library imports.
import run_pathomics_parallel as parallel
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import partial
import json
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image
import pathomics_core as core
import pathomics_fast as fast
import pathology_feature_extractor_v2 as v2

extraction = parallel.extraction
ORIGINAL_PROCESS = extraction.process_slide
LEGACY_HASHES = {
    "run_wsi_pathomics.py": "4e0a7f7a1676cb3024e72b7436847110a9262365749353c940996c045faa07b4",
    "pathomics_core.py": "d4d5f296da53c1677e0bad6bcfd4dc0caf28e16cc7c327666530f43979d60271",
    "pathology_feature_extractor_v2.py": "0882a23614e38b0a2fe2a494334073a6683d4f5db8e5db12035acde814ed4fd2",
    "run_pathomics_parallel.py": "0bbe3b47a0dada2010c99818c658907dcb73a78a879f0c6fc3368a3ec21d6cab",
}


def engine_identity():
    modules = [extraction, core, v2, parallel]
    observed = {Path(m.__file__).name: extraction.sha(m.__file__) for m in modules}
    if observed != LEGACY_HASHES:
        raise ValueError("Original code differs from the tested version. Do NOT edit hashes or overwrite original files; inspect versions first: " + json.dumps(observed))
    return dict(engine="pathomics_fast_v1", originals=observed,
                fast_sha256=extraction.sha(fast.__file__), launcher_sha256=extraction.sha(__file__))


def compare(reference, optimized):
    if list(reference) != list(optimized):
        raise AssertionError("Feature/QC keys or column order differ")
    for key, a in reference.items():
        b = optimized[key]
        if isinstance(a, str):
            if a != b: raise AssertionError("String/QC mismatch: " + key)
        else:
            np.testing.assert_allclose(a, b, rtol=0, atol=0, equal_nan=True, err_msg=key)


def validation_record(cfg):
    identity = engine_identity()
    signature, _ = extraction.config_signature(cfg, False)
    return dict(**identity, extraction_signature=signature)


def verify(cfg, patches=8):
    """Small real-image test; writes only a separate validation receipt."""
    parallel.run(cfg, 1, check_only=True)
    expected = validation_record(cfg)
    root = Path(cfg["output"])/"full"
    _, jobs, _ = extraction.cohort_and_jobs(cfg)
    groups = {}
    # Prefer previously completed slides; still sample all source/HE branches.
    for job in sorted(jobs, key=lambda j: not (root/"slides"/j["slide_id"]/"complete.json").exists()):
        groups.setdefault((job["spec"]["source"], job["he_applicable"]), job)
    results = []
    s = core.Settings(**cfg.get("settings", {}))
    import cv2
    cv2.setNumThreads(1)
    for group, job in groups.items():
        coords, size, _, _ = extraction.input_descriptor(job, cfg)
        keep, _ = core.select_nonoverlapping(coords, size)
        indices = keep[np.unique(np.linspace(0, len(keep)-1, min(patches, len(keep)), dtype=int))]
        reader = None
        try:
            if job["spec"]["source"] != "images":
                reader, read, mx, my, _, _ = extraction.open_wsi(job, cfg)
            for index in indices:
                x, y = map(int, coords[index])
                start = time.perf_counter()
                if reader is not None:
                    halo = int(math.ceil(s.halo_um/min(mx, my)))
                    if x < 0 or y < 0 or x+size > reader.total_w or y+size > reader.total_h:
                        raise ValueError("Source core exceeds WSI bounds")
                    rgb = np.asarray(read(x-halo, y-halo, size+2*halo))
                    box = (halo, halo, size, size)
                else:
                    row = job["spec"]["images"][int(index)]
                    mx, my = float(row["mpp_x"]), float(row["mpp_y"])
                    with Image.open(row["image_path"]) as im: rgb = np.asarray(im.convert("RGB"))
                    box = (0, 0, size, size)
                read_seconds = time.perf_counter()-start
                masks, _ = extraction.mask_bundle(cfg, job["slide_id"], int(index))
                args = (rgb, box, mx, my, s, masks)
                times = {}
                # Alternate ordering to reduce systematic warm-cache advantage.
                for name in (["legacy", "fast"] if len(results)%2 == 0 else ["fast", "legacy"]):
                    f = core.describe_patch if name == "legacy" else fast.describe_patch_fast
                    start = time.perf_counter()
                    values, preview = f(*args, he_applicable=job["he_applicable"])
                    times[name] = time.perf_counter()-start
                    if name == "legacy": original, original_preview = values, preview
                    else: optimized, optimized_preview = values, preview
                compare(original, optimized)
                np.testing.assert_array_equal(original_preview, np.asarray(Image.fromarray(optimized_preview)))
                results.append(dict(slide_id=job["slide_id"], patch_index=int(index), source=group[0],
                                    he_applicable=group[1], qc_pass=original["qc_pass"],
                                    read_seconds=read_seconds, legacy_seconds=times["legacy"], fast_seconds=times["fast"]))
            print(f"PARITY OK: {job['slide_id']}; patches={len(indices)}; HE={job['he_applicable']}", flush=True)
        finally:
            if reader is not None and hasattr(reader, "close"): reader.close()
    if not results: raise ValueError("No comparison patches")
    slow = sum(r["legacy_seconds"] for r in results)
    quick = sum(r["fast_seconds"] for r in results)
    record = dict(**expected, status="passed", checked_utc=datetime.now(timezone.utc).isoformat(),
                  cases=results, patches=len(results), calculation_speedup=slow/quick,
                  note="Exact feature/QC/preview parity on selected real patches only, not proof for every WSI. Timing excludes WSI I/O and fast preview materialization. Source/coordinate issues are not bypassed.")
    root.mkdir(parents=True, exist_ok=True)
    extraction.write_json(root/"fast_validation.json", record)
    print(f"Exact parity passed: {len(results)} patches. Calculation-only speedup on samples: {slow/quick:.2f}x", flush=True)
    print("Saved fast_validation.json. Not a whole-run speed guarantee. No full extraction started.", flush=True)


def require_validation(cfg):
    expected = validation_record(cfg)
    path = Path(cfg["output"])/"full"/"fast_validation.json"
    if not path.exists(): raise ValueError("Run verify first; no fast_validation.json")
    actual = json.loads(path.read_text())
    if actual.get("status") != "passed" or any(actual.get(k) != v for k, v in expected.items()):
        raise ValueError("Validation/code/input fingerprint mismatch. Run verify again; do not bypass.")
    return expected


def install_worker():
    engine_identity()
    # Process-local binding only; original source files and hash checks stay intact.
    extraction.describe_patch = fast.describe_patch_fast


def process_fast(job, cfg, out, signature, smoke=False, audit=False):
    if smoke or audit:
        return ORIGINAL_PROCESS(job, cfg, out, signature, smoke, audit)
    folder = Path(out)/"slides"/job["slide_id"]
    if not (folder/"complete.json").exists():
        folder.mkdir(parents=True, exist_ok=True)
        extraction.write_json(folder/"optimization.json", dict(**engine_identity(),
            extraction_signature=signature, validation_sha256=extraction.sha(Path(out)/"fast_validation.json")))
    return ORIGINAL_PROCESS(job, cfg, out, signature, smoke, audit)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["check", "verify", "run"])
    p.add_argument("--config", default="pathomics_all_wsi.json")
    p.add_argument("--workers", type=int, default=30)
    p.add_argument("--patches", type=int, default=8)
    p.add_argument("--allow-verified-extent-mpp", action="store_true")
    p.add_argument("--require-validation", action="store_true", help="For check: also require a matching successful parity receipt")
    args = p.parse_args()
    if not 1 <= args.workers <= 32 or not 1 <= args.patches <= 64: p.error("workers 1..32; patches 1..64")
    cfg = extraction.load_config(args.config)
    if args.allow_verified_extent_mpp: cfg["allow_verified_extent_mpp"] = True
    engine_identity()
    if args.action == "check":
        parallel.run(cfg, args.workers, check_only=True)
        if args.require_validation:
            require_validation(cfg)
            print("Matching real-image parity validation found.", flush=True)
        print("Fast engine compatible. Next: verify (small real-image parity/timing test).", flush=True)
    elif args.action == "verify":
        verify(cfg, args.patches)
    else:
        require_validation(cfg)
        parallel.ProcessPoolExecutor = partial(ProcessPoolExecutor, initializer=install_worker)
        extraction.process_slide = process_fast
        print("FAST ENGINE: same features/QC; lazy previews, cached Gabor kernels, skip discarded texture calculations.", flush=True)
        parallel.run(cfg, args.workers)


if __name__ == "__main__": main()
