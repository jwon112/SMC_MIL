#!/usr/bin/env python3
"""Parallel all-WSI launcher; reuse the unchanged, fingerprinted extractor.

Only orchestration changes. Original run_config and completed-slide checks
remain mandatory. A separate execution receipt records this launcher.
"""
from __future__ import annotations

import os

# Set BEFORE importing numpy/scipy or the extraction module. Each process is
# already a unit of parallelism; do not multiply BLAS/OpenMP threads within it.
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
for name in THREAD_ENV:
    os.environ[name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path

import run_wsi_pathomics as extraction


def run(cfg, workers, check_only=False):
    if not 1 <= workers <= 32:
        raise ValueError("workers must be 1..32")
    if cfg.get("review_only", False):
        raise ValueError("Full extraction is disabled by this review_only config")
    if cfg.get("population") != "all_feature_wsi" or cfg.get("stain_scope") != "all":
        raise ValueError("Use the all_feature_wsi / all-stain config, not the old HE-only config")
    out = Path(cfg["output"])/"full"
    if out.resolve() == Path(cfg["baseline_dir"]).resolve():
        raise ValueError("Output cannot overwrite original features")
    cohort, jobs, exclusions = extraction.cohort_and_jobs(cfg)
    signature, record = extraction.config_signature(cfg, False)
    lock = out/"run_config.json"

    def verify_existing():
        if lock.exists() and json.loads(lock.read_text())["signature"] != signature:
            raise ValueError("Config/code/input signature differs. Do not bypass this check; restore matching inputs or use a new output directory.")

    verify_existing()
    existing = sum((out/"slides"/j["slide_id"]/"complete.json").is_file() for j in jobs)
    print(f"Parallel preflight: population={len(cohort)}, eligible={len(jobs)}, unavailable={len(exclusions)}, workers={workers}, numeric_threads_per_worker=1", flush=True)
    print(f"Existing completion receipts={existing}; every receipt/output will be verified before reuse. Output={out}", flush=True)
    print("Extraction signature matches existing run." if lock.exists() else "New extraction output.", flush=True)
    if check_only:
        print("Check only: no extraction or writes. WSI/MPP and individual receipt checks occur during processing.", flush=True)
        return

    out.mkdir(parents=True, exist_ok=True)
    with (out/".writer.lock").open("a") as process_lock:
        try:
            fcntl.flock(process_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another pathomics process is writing this output directory")
        verify_existing()
        if not lock.exists():
            extraction.write_json(lock, dict(signature=signature, **record))
        started = datetime.now(timezone.utc)
        receipt = out/f"parallel_execution_{started.strftime('%Y%m%dT%H%M%S%fZ')}_{os.getpid()}.json"
        execution = dict(started_utc=started.isoformat(), pid=os.getpid(), workers=workers,
                         thread_environment={k:os.environ[k] for k in THREAD_ENV},
                         extraction_signature=signature, launcher_sha256=extraction.sha(__file__),
                         eligible=len(jobs), status="running", completed=0, reused=0)
        extraction.write_json(receipt, execution)
        failed = []
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(extraction.process_slide, job, cfg, str(out), signature, False): job for job in jobs}
                for f in as_completed(futures):
                    job = futures[f]
                    try:
                        result = f.result()
                        status = result["status"]
                        execution[status] += 1
                        print(f"[{status}] {job['slide_id']}: {result['patches_read']} patches; QC pass={result['patches_qc_pass']}; done={execution['completed']+execution['reused']}/{len(jobs)}", flush=True)
                    except Exception as exc:
                        failed.append(dict(slide_id=job["slide_id"], error=str(exc)))
                        print(f"[FAILED] {job['slide_id']}: {exc}", flush=True)
            extraction.write_json(out/"failures.json", failed)
            extraction.make_report(cohort, jobs, exclusions, cfg, out, signature, False)
            execution["status"] = "failed" if failed else "complete"
            execution["failed"] = len(failed)
            if failed:
                raise RuntimeError(f"{len(failed)} slides failed; partial report saved; originals untouched")
            print(f"Saved to {out}", flush=True)
        except BaseException:
            execution["status"] = "failed_or_interrupted"
            raise
        finally:
            execution["finished_utc"] = datetime.now(timezone.utc).isoformat()
            extraction.write_json(receipt, execution)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pathomics_all_wsi.json")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--allow-verified-extent-mpp", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cfg = extraction.load_config(args.config)
    if args.allow_verified_extent_mpp:
        cfg["allow_verified_extent_mpp"] = True
    run(cfg, args.workers, args.check_only)


if __name__ == "__main__":
    main()
