#!/usr/bin/env python3
"""SMC server: existing original WSI coordinates -> patch/slide/event pathomics.

Audit first; synthetic/image fixtures supported separately from WSI mode.
Existing images, UNI tensors, labels, and splits are strictly read-only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from PIL import Image
from pathomics_core import (CORE_FEATURES, Settings, aggregate, describe_patch,
                           rook_moran, select_nonoverlapping, validate_settings)

VERSION = "smc_pathomics_1.3.0_all_feature_wsi"
META = {"slide_id", "patch_index", "x_source", "y_source", "x_level0", "y_level0",
        "patch_size", "source_level", "source_mpp_x", "source_mpp_y", "qc_pass", "qc_reason",
        "analysis_mpp", "analysis_width_px", "analysis_height_px", "resampling_size_relative_error",
        "stain_norm_ok", "seg_h_threshold", "tissue_area_mm2", "nucleus_occupied_area_mm2",
        "nuclei_context_edge_excluded", "mask_status"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, Path): return str(value)
    return value


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(tmp, path)


def read_csv(path, required=(), unique=None):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if set(required)-set(frame.columns): raise ValueError(f"Missing columns {set(required)-set(frame.columns)} in {path}")
    if unique and (frame[unique].eq("").any() or frame[unique].duplicated().any()):
        raise ValueError(f"Blank/duplicate {unique} in {path}")
    return frame


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    cfg["config_path"] = str(Path(path).resolve())
    s = Settings(**cfg.get("settings", {}))
    validate_settings(s)
    if cfg.get("label_source", "manual_only") not in {"manual_only", "manual_or_filename"}:
        raise ValueError("Invalid label_source")
    if cfg.get("population", "gold_cohort") not in {"gold_cohort", "all_feature_wsi"}:
        raise ValueError("Invalid population")
    if cfg.get("stain_scope", "confirmed_he") not in {"confirmed_he", "all"}:
        raise ValueError("Invalid stain_scope")
    if cfg.get("population") == "all_feature_wsi" and cfg.get("stain_scope") != "all":
        raise ValueError("all_feature_wsi requires stain_scope=all")
    for key, default in [("smoke_patches",64),("smoke_slides_per_source",2),("preview_count",3)]:
        value=cfg.get(key,default)
        if not isinstance(value,int) or isinstance(value,bool) or value < (0 if key=="preview_count" else 1):
            raise ValueError(f"{key} must be an integer >= {0 if key=='preview_count' else 1}")
    return cfg


def config_signature(cfg, smoke):
    # Include formulas and installed reader code in reproducibility lock.
    import pathology_feature_extractor_v2 as v2
    import pathomics_core
    files = [Path(__file__), Path(pathomics_core.__file__), Path(v2.__file__), Path(cfg["cohort_csv"]), Path(cfg["stain_csv"])]
    for source in cfg.get("sources", {}).values(): files.append(Path(source["manifest"]))
    if cfg.get("image_manifest"): files.append(Path(cfg["image_manifest"]))
    if cfg.get("mpp_overrides_csv"): files.append(Path(cfg["mpp_overrides_csv"]))
    if not cfg.get("image_manifest"):
        for module in ["extract_features_dicom", "extract_features_mrxs", "mrxs_pyramid", "dicom_pyramid"]:
            p = Path(cfg["project"])/f"{module}.py"
            if p.exists(): files.append(p)
    payload = dict(version=VERSION, config={k:v for k,v in cfg.items() if k!="config_path"},
                   smoke=smoke, files={str(p.resolve()): sha(p) for p in files})
    if cfg.get("population") == "all_feature_wsi" and not cfg.get("image_manifest"):
        payload["original_pt_inventory"] = sorted(p.name for p in (Path(cfg["baseline_dir"])/"pt_files").glob("*.pt") if p.is_file())
    import scipy, skimage, cv2, pywt
    payload["versions"] = dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                              skimage=skimage.__version__, cv2=cv2.__version__, pywt=pywt.__version__)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest, payload


def cohort_and_jobs(cfg):
    cohort = read_csv(cfg["cohort_csv"], ["slide_id", "case_id", "event_id", "biopsy_date", "source_dataset"], "slide_id")
    for key in ["case_id", "event_id", "biopsy_date"]:
        if cohort[key].eq("").any(): raise ValueError(f"Cohort has blank {key}; exact event mapping required")
    dates = pd.to_datetime(cohort.biopsy_date, errors="coerce")
    if dates.isna().any(): raise ValueError("Invalid biopsy_date; do not infer from filenames")
    cohort["biopsy_date"] = dates.dt.strftime("%Y-%m-%d")
    if cohort.groupby(["case_id", "event_id"]).biopsy_date.nunique().max() > 1:
        raise ValueError("One patient/event_id maps to multiple dates")
    cohort["examid"] = ["BIO_"+hashlib.sha256(f"{c}|{e}|{d}".encode()).hexdigest()[:20]
                         for c,e,d in zip(cohort.case_id,cohort.event_id,cohort.biopsy_date)]
    stains = read_csv(cfg["stain_csv"], ["manual_label", "final_label", "reference_source"], "slide_id").set_index("slide_id")
    specs = {}
    if cfg.get("image_manifest"):
        images = read_csv(cfg["image_manifest"], ["slide_id","image_path","x","y","patch_size","mpp_x","mpp_y","patch_index"])
        for sid, frame in images.groupby("slide_id", sort=False):
            specs[sid] = {"source": "images", "images": frame.to_dict("records")}
    else:
        sys.path.insert(0, str(Path(cfg["project"]).resolve()))
        from extract_features_dicom import read_manifest
        for name, source in cfg["sources"].items():
            if source["kind"] not in {"dicom", "mrxs"}: raise ValueError("Unsupported WSI source")
            for spec in read_manifest(Path(source["manifest"]), Path(source["root"])):
                if spec.slide_id in specs: raise ValueError("Duplicate slide across source manifests")
                specs[spec.slide_id] = dict(source=source["kind"], source_dataset=name,
                    root=source["root"], coords_path=str(spec.coords_path), slide_rel_path=spec.slide_rel_path)
    if cfg.get("population") == "all_feature_wsi":
        ids = sorted(specs) if cfg.get("image_manifest") else sorted(
            p.stem for p in (Path(cfg["baseline_dir"])/"pt_files").glob("*.pt") if p.is_file())
        if not ids: raise ValueError("No original .pt files in configured baseline_dir/pt_files")
        linked = cohort.set_index("slide_id").to_dict("index")
        rows = []
        for sid in ids:
            row = dict.fromkeys(cohort.columns, "")
            row.update(linked.get(sid, {}))
            row["slide_id"] = sid
            row["linkage_status"] = "exact_cohort_match" if sid in linked else "unmapped"
            if not row["source_dataset"]: row["source_dataset"] = specs.get(sid, {}).get("source_dataset", "")
            rows.append(row)
        cohort = pd.DataFrame(rows)
    jobs, exclusions = [], []
    annotated = []
    for row in cohort.to_dict("records"):
        sid = row["slide_id"]
        if not re.fullmatch(r"[A-Za-z0-9._-]+", sid): raise ValueError("Unsafe slide ID")
        reason = ""
        stain = stains.loc[sid].to_dict() if sid in stains.index else {}
        manual = stain.get("reference_source") == "manual_review" and stain.get("manual_label") == "HE"
        filename = (cfg.get("label_source") == "manual_or_filename" and
                    stain.get("reference_source") == "filename_rule" and stain.get("final_label") == "HE")
        label = stain.get("manual_label") if stain.get("reference_source") == "manual_review" else stain.get("final_label", "")
        normalized = str(label or "").strip().upper()
        group = {"HE":"HE", "H&E":"HE", "IHC":"IHC", "OTHERS":"OTHER", "OTHER":"OTHER", "SPECIAL_OTHER":"OTHER", "SPECIAL STAIN":"OTHER"}.get(normalized, "UNKNOWN")
        row.update(stain_group=group, stain_label_raw=label or "", stain_reference_source=stain.get("reference_source", ""),
                   he_applicable=bool(manual or filename))
        annotated.append(row)
        if cfg.get("stain_scope", "confirmed_he") != "all" and not (manual or filename): reason = "no_confirmed_HE_label"
        elif sid not in specs: reason = "missing_source_manifest"
        elif not cfg.get("image_manifest") and not (Path(cfg["baseline_dir"])/"pt_files"/f"{sid}.pt").is_file(): reason = "missing_original_pt"
        elif not cfg.get("image_manifest") and row["source_dataset"] != specs[sid]["source_dataset"]: reason = "source_dataset_mismatch"
        if reason:
            exclusions.append(dict(slide_id=sid,examid=row["examid"],reason=reason))
        else:
            jobs.append(dict(**row, spec=specs[sid]))
    if not jobs: raise ValueError("No eligible WSI with required source data; check manifests/features")
    return pd.DataFrame(annotated), jobs, exclusions


def input_descriptor(job, cfg):
    spec = job["spec"]
    if spec["source"] == "images":
        rows = spec["images"]
        sizes = {int(r["patch_size"]) for r in rows}
        if len(sizes)!=1: raise ValueError("Image patch sizes must agree within slide")
        indices = [int(r["patch_index"]) for r in rows]
        if len(set(indices))!=len(indices): raise ValueError("Duplicate patch_index")
        if indices != list(range(len(rows))): raise ValueError("Image patch_index must be contiguous original row order from 0")
        coords = np.array([[int(r["x"]),int(r["y"])] for r in rows])
        size = sizes.pop()
        input_sha = hashlib.sha256(json.dumps([(r,sha(r["image_path"])) for r in rows], sort_keys=True).encode()).hexdigest()
        return coords, size, input_sha, False
    sys.path.insert(0, cfg["project"])
    from extract_features_dicom import read_coordinates
    c = read_coordinates(Path(spec["coords_path"]))
    if c.pyramid_level != 0: raise ValueError("v1 supports level-0 source coordinates only")
    if c.coords_level0 is not None and not np.array_equal(c.coords_level, c.coords_level0):
        raise ValueError("Source level0 coordinate arrays disagree")
    import h5py
    baseline = Path(cfg["baseline_dir"])/"h5_files"/f"{job['slide_id']}.h5"
    if not baseline.exists(): raise ValueError("Original feature H5 is required to verify full patch row count")
    with h5py.File(baseline) as h:
        if h["features"].shape[0] != len(c.coords_level): raise ValueError("Original features are not a full source-coordinate bag")
        mismatch = "coords" not in h or not np.array_equal(h["coords"][:], c.coords_level)
        if mismatch and spec["source"] != "mrxs": raise ValueError("DICOM baseline/source coordinate mismatch")
        # MRXS stale output coords are recorded, NOT used to read pixels or join
        # features. This does not claim re-encoded UNI tensor parity.
    return c.coords_level, c.patch_size, sha(spec["coords_path"]), bool(mismatch)


def dicom_mpp(ds):
    """Resolve standard PixelSpacing, never guess from magnification.

    DICOM values are (row, column) in mm, while outputs are (x, y) in um.
    Per-frame-only spacing must be present and consistent in every frame.
    Volume extent is diagnostic only; it is not an automatic scale fallback.
    https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.16.2.html
    """
    info = {"sources": [], "issues": []}
    candidates = []

    def add(value, source):
        if value is None: return False
        try:
            if len(value) != 2: raise ValueError("expected row,column pair")
            xy = (float(value[1])*1000, float(value[0])*1000)
            if any(not math.isfinite(v) or v <= 0 for v in xy):
                raise ValueError("nonpositive/nonfinite spacing")
            candidates.append(xy)
            if source not in info["sources"]: info["sources"].append(source)
            return True
        except (TypeError, ValueError, IndexError) as exc:
            issue = f"{source}: {exc}"
            if issue not in info["issues"]: info["issues"].append(issue)
            return False

    add(getattr(ds, "PixelSpacing", None), "top_level.PixelSpacing")
    shared = getattr(ds, "SharedFunctionalGroupsSequence", []) or []
    for group in shared:
        for pm in getattr(group, "PixelMeasuresSequence", []) or []:
            add(getattr(pm, "PixelSpacing", None), "shared.PixelMeasuresSequence.PixelSpacing")
    shared_or_top = bool(candidates)
    per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", []) or []
    frame_valid = 0
    for group in per_frame:
        valid = False
        for pm in getattr(group, "PixelMeasuresSequence", []) or []:
            valid = add(getattr(pm, "PixelSpacing", None), "per_frame.PixelMeasuresSequence.PixelSpacing") or valid
        frame_valid += int(valid)
    info["per_frame_items"] = len(per_frame)
    info["frames_with_valid_spacing"] = frame_valid
    if per_frame and not shared_or_top:
        if frame_valid != len(per_frame): info["issues"].append("Incomplete per-frame-only spacing")
        expected = getattr(ds, "NumberOfFrames", None)
        if expected is not None and int(expected) != len(per_frame):
            info["issues"].append("Per-frame sequence length differs from NumberOfFrames")
    if candidates and not np.allclose(candidates, candidates[0], rtol=1e-5, atol=1e-8):
        info["issues"].append("Conflicting PixelSpacing values")
    info["unique_mpp_xy_um"] = [list(xy) for xy in dict.fromkeys(candidates)][:8]
    # Whitelist geometry fields: no patient name, ID, birth date, or full dump.
    for key in ["TotalPixelMatrixColumns", "TotalPixelMatrixRows", "ImagedVolumeWidth", "ImagedVolumeHeight"]:
        value = getattr(ds, key, None)
        try: info[key] = float(value) if value is not None else None
        except (TypeError, ValueError): info[key] = "invalid"
    try:
        w,h,cols,rows = [info[k] for k in ["ImagedVolumeWidth","ImagedVolumeHeight","TotalPixelMatrixColumns","TotalPixelMatrixRows"]]
        if all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in [w,h,cols,rows]):
            info["extent_derived_mpp_xy_um_REVIEW_ONLY"] = [w*1000/cols,h*1000/rows]
    except (TypeError, ValueError): pass
    valid = bool(candidates) and not info["issues"]
    return (candidates[0] if valid else (np.nan,np.nan)), info


def stored_mpp_diagnostics(job, cfg):
    """Read only a small, whitelisted subset of existing H5 metadata."""
    import h5py
    paths = []
    if job["spec"].get("coords_path"): paths.append(Path(job["spec"]["coords_path"]))
    if cfg.get("baseline_dir"):
        paths.append(Path(cfg["baseline_dir"])/"h5_files"/f"{job['slide_id']}.h5")
    result = []
    keys = ["mpp_x_um","mpp_y_um","mpp_x","mpp_y","mpp","patch_level","pyramid_level","coordinate_storage_space"]
    for path in paths:
        if not path.is_file(): continue
        try:
            with h5py.File(path) as f:
                attrs = {}
                for group_name, obj in [("file",f), ("coords",f.get("coords"))]:
                    if obj is None: continue
                    for k in keys:
                        if k in obj.attrs:
                            val = obj.attrs[k]
                            if isinstance(val,bytes): val=val.decode(errors="replace")
                            elif isinstance(val,np.ndarray): val=val.tolist()
                            attrs[f"{group_name}.{k}"] = clean(val)
                result.append(dict(file=path.name,stored_attributes_REVIEW_ONLY=attrs))
        except Exception as exc: result.append(dict(file=path.name,error=type(exc).__name__))
    return result


def verified_extent_mpp(job, headers, reader):
    """Opt-in metadata consistency check, NOT independent scale calibration."""
    import h5py
    from extract_features_dicom import read_coordinates
    if not headers or any(d.get("issues") or d.get("sources") for d in headers):
        raise ValueError("Extent fallback requires absent PixelSpacing in ALL instances, without spacing errors")
    pairs = []
    for d in headers:
        pair = d.get("extent_derived_mpp_xy_um_REVIEW_ONLY")
        if pair is None or not np.isfinite(pair).all() or min(pair) <= 0:
            raise ValueError("Missing valid volume extent / total matrix dimensions")
        if (d.get("TotalPixelMatrixColumns") != reader.total_w or
                d.get("TotalPixelMatrixRows") != reader.total_h):
            raise ValueError("DICOM total matrix differs from selected reader level-0 dimensions")
        pairs.append(pair)
    if not np.allclose(pairs, pairs[0], rtol=1e-5, atol=1e-8):
        raise ValueError("Extent MPP differs across selected DICOM instances")
    path = Path(job["spec"]["coords_path"])
    c = read_coordinates(path)
    if c.pyramid_level != 0:
        raise ValueError("Extent fallback requires level-0 source coordinates")
    if c.coords_level0 is not None and not np.array_equal(c.coords_level, c.coords_level0):
        raise ValueError("Source level-0 coordinate arrays disagree")
    stored = []
    with h5py.File(path, "r") as f:
        for name, obj in [("file", f), ("coords", f.get("coords"))]:
            if obj is None: continue
            attrs = obj.attrs
            if "mpp_x_um" not in attrs and "mpp_y_um" not in attrs: continue
            try:
                xy = [float(attrs["mpp_x_um"]), float(attrs["mpp_y_um"])]
            except (KeyError, TypeError, ValueError):
                raise ValueError("Incomplete/invalid source-coordinate MPP attributes") from None
            if not np.isfinite(xy).all() or min(xy) <= 0 or not np.allclose(xy, pairs[0], rtol=1e-5, atol=1e-8):
                raise ValueError("Source-coordinate MPP disagrees with DICOM extent MPP")
            stored.append(dict(location=name, mpp_xy_um=xy))
    if not stored:
        raise ValueError("Source coordinate H5 lacks mpp_x_um/mpp_y_um corroboration")
    evidence = dict(method="volume_extent_divided_by_total_matrix_checked_against_source_H5",
                    independent_calibration=False, source_coordinate_sha256=sha(path),
                    source_coordinate_level=0, stored_mpp=stored, dicom_headers=headers,
                    agreement_rtol=1e-5, agreement_atol=1e-8,
                    note="Sources may share derivation; consistency is not independent physical calibration.")
    return tuple(pairs[0]), evidence


def open_wsi(job, cfg):
    spec = job["spec"]
    sys.path.insert(0, cfg["project"])
    diagnostics = []; origin = "WSI_header"
    if spec["source"] == "dicom":
        from extract_features_dicom import DicomPyramidReader
        import pydicom
        reader = DicomPyramidReader(Path(spec["root"])/spec["slide_rel_path"], 0)
        spacing = []
        for path in reader.paths:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            pair,detail = dicom_mpp(ds)
            diagnostics.append(dict(file=Path(path).name,**detail))
            spacing.append(pair)
        consistent = bool(spacing) and np.isfinite(spacing).all() and np.allclose(spacing,spacing[0],rtol=1e-5,atol=1e-8)
        mx,my = spacing[0] if consistent else (np.nan,np.nan)
        origin = "WSI_header:"+"|".join(sorted({s for d in diagnostics for s in d.get("sources",[])}))
        if not consistent and cfg.get("allow_verified_extent_mpp", False):
            try:
                (mx,my), evidence = verified_extent_mpp(job, diagnostics, reader)
                reader.pathomics_mpp_evidence = evidence
                origin = "DICOM_extent_source_H5_consistent_opt_in"
            except ValueError as exc:
                diagnostics.append(dict(extent_fallback_rejected=str(exc)))
        if any(not math.isfinite(v) or v <= 0 for v in [mx,my]):
            diagnostics.append(dict(issues=["Missing/invalid/inconsistent MPP across selected DICOM instances"]))
        read = lambda x,y,n: reader.read_patch(x,y,n,n)
        inputs = list(reader.paths)
    else:
        from extract_features_mrxs import MrxsPyramidReader
        from mrxs_pyramid import choose_mrxs
        reader = MrxsPyramidReader(choose_mrxs(Path(spec["root"])/spec["slide_rel_path"]), 0)
        try:
            mx,my = float(reader.mpp_x_um),float(reader.mpp_y_um)
        except (AttributeError, TypeError, ValueError): mx,my=np.nan,np.nan
        props = getattr(getattr(reader,"slide",None),"properties",{})
        if not np.isfinite([mx,my]).all() or min(mx,my)<=0:
            try:
                mx,my=float(props["openslide.mpp-x"]),float(props["openslide.mpp-y"])
                origin="WSI_header:openslide_level0_properties"
            except (KeyError,TypeError,ValueError): mx,my=np.nan,np.nan
        diagnostics.append(dict(file=Path(reader.path).name,
            reader_mpp_xy_um=clean([mx,my]),
            openslide_mpp_x=props.get("openslide.mpp-x"),openslide_mpp_y=props.get("openslide.mpp-y")))
        read = reader.read_patch
        inputs = [Path(reader.path)]
    if cfg.get("mpp_overrides_csv"):
        overrides = read_csv(cfg["mpp_overrides_csv"], ["slide_id","mpp_x","mpp_y","review_source"], "slide_id").set_index("slide_id")
        if job["slide_id"] in overrides.index:
            r=overrides.loc[job["slide_id"]]
            if not r.review_source: raise ValueError("MPP override requires review_source")
            mx,my,origin=float(r.mpp_x),float(r.mpp_y),"reviewed_override:"+r.review_source
            reader.pathomics_mpp_evidence = dict(method="reviewed_override", review_source=r.review_source)
    if any(not math.isfinite(v) or v<=0 for v in [mx,my]):
        if hasattr(reader,"close"): reader.close()
        report=dict(slide_id=job["slide_id"],source=spec["source"],
                    selected_instance_count=len(inputs),headers=diagnostics[:4],
                    stored_metadata=stored_mpp_diagnostics(job,cfg),
                    note="REVIEW_ONLY values are NOT applied. No physical size assumed from 40x.")
        raise ValueError("Missing/conflicting valid WSI MPP. MPP_DIAGNOSTIC="+
                         json.dumps(clean(report),ensure_ascii=False))
    inventory=[dict(path=str(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in inputs]
    return reader,read,mx,my,origin,inventory


def mask_bundle(cfg, slide_id, index):
    if not cfg.get("mask_root"): return {},"not_supplied"
    path=Path(cfg["mask_root"])/slide_id/f"{index:08d}.npz"
    if not path.exists(): return {},"not_supplied"
    metadata=path.with_suffix(".json")
    if not metadata.exists(): raise ValueError("Mask requires sidecar JSON with review_source and coordinate binding")
    meta=json.loads(metadata.read_text())
    if not meta.get("review_source"): raise ValueError("Unreviewed mask provenance")
    allowed={"myocardium","stroma","lymphocyte_instances"}
    with np.load(path, allow_pickle=False) as f:
        if set(f.files)-allowed: raise ValueError(f"Unknown mask fields {set(f.files)-allowed}")
        masks={k:f[k] for k in f.files}
    return masks,meta


def select_measurement_review(frame, per_group=3):
    """Deterministic review queue, not automatic focus/segmentation rejection."""
    chosen={}
    def add(rows, reason):
        for index in rows.patch_index:
            chosen.setdefault(int(index),[]).append(reason)
    good=frame.loc[frame.qc_pass==1]
    add(good.head(per_group),'first_basic_QC_pass')
    add(frame.loc[frame.qc_pass==0].head(per_group),'first_basic_QC_fail')
    for field,reason in [('qc_blur_laplacian_var','lowest_focus_score_relative_only'),
                         ('qc_gray_std_in_tissue','lowest_structure_score_relative_only')]:
        add(good.loc[np.isfinite(good[field])].sort_values([field,'patch_index']).head(per_group),reason)
    add(good.loc[good.qc_detected_nuclei_before_exclusion==0].head(per_group),'zero_detected_nuclei_review')
    diff=good[['patch_index','qc_candidate_nucleus_count','qc_detected_nuclei_before_exclusion']].copy()
    diff['candidate_count_difference']=(diff.qc_candidate_nucleus_count-diff.qc_detected_nuclei_before_exclusion).abs()
    add(diff.loc[diff.candidate_count_difference>0].sort_values(['candidate_count_difference','patch_index'],ascending=[False,True]).head(per_group),
        'largest_support_comparison_difference')
    return [dict(patch_index=i,selection_reasons=';'.join(reasons)) for i,reasons in chosen.items()]


def process_slide(job,cfg,out,signature,smoke=False,audit=False):
    import cv2
    cv2.setNumThreads(1)
    out=Path(out); sid=job["slide_id"]
    coords,size,coord_hash,mismatch=input_descriptor(job,cfg)
    keep,drop=select_nonoverlapping(coords,size)
    if not len(keep): raise ValueError("No source coordinates")
    # Select spread-out smoke samples without consulting label or attention.
    indices=keep[np.unique(np.linspace(0,len(keep)-1,min(cfg.get("smoke_patches",64),len(keep)),dtype=int))] if smoke else keep
    reader=None; read=None
    if job["spec"]["source"] != "images":
        reader,read,mx,my,mpp_source,inventory=open_wsi(job,cfg)
    else:
        rows=job["spec"]["images"]
        spacings={(float(r["mpp_x"]),float(r["mpp_y"])) for r in rows}
        if len(spacings)!=1: raise ValueError("One slide must use one native MPP")
        mx,my=spacings.pop();mpp_source="image_manifest";inventory=[]
    try:
        receipt=dict(slide_id=sid,examid=job["examid"],source_patch_count=len(coords),
                     stain_group=job.get("stain_group", "HE"), he_applicable=job.get("he_applicable", True),
                     selected_nonoverlap_patches=len(keep),overlap_excluded=len(drop),
                     coordinate_sha256=coord_hash,baseline_h5_coords_mismatch=mismatch,
                     source_mpp_x=mx,source_mpp_y=my,mpp_source=mpp_source,patch_size=size,
                     mpp_evidence=getattr(reader,"pathomics_mpp_evidence",{}),
                     signature=signature,smoke_only=smoke,source_inventory=inventory)
        if audit: return receipt
        folder=out/"slides"/sid;folder.mkdir(parents=True,exist_ok=True)
        final=folder/"patch_features.csv.gz";done=folder/"complete.json"
        mask_fingerprint = []
        if cfg.get("mask_root"):
            mp=Path(cfg["mask_root"])/sid
            mask_fingerprint=[(str(p),sha(p)) for p in sorted(mp.glob("*")) if p.suffix in {".json",".npz"}]
        receipt["mask_sha256"] = hashlib.sha256(json.dumps(mask_fingerprint).encode()).hexdigest()
        if done.exists():
            old=json.loads(done.read_text())
            if (any(old.get(k)!=v for k,v in receipt.items()) or not final.exists()
                    or sha(final)!=old.get("patch_csv_sha256") or not (folder/"summary.json").exists()
                    or sha(folder/"summary.json")!=old.get("summary_sha256")
                    or not (folder/"preview_index.json").exists()
                    or sha(folder/"preview_index.json")!=old.get("preview_index_sha256")
                    or (smoke and (not (folder/'measurement_review.csv').exists()
                                   or sha(folder/'measurement_review.csv')!=old.get('measurement_review_sha256')))):
                raise ValueError("Existing slide output or config differs; use a new output directory")
            return dict(old,status="reused")
        # A crashed partial slide is recomputed, never accepted as completed.
        tmp=folder/"patch_features.csv.gz.partial"
        s=Settings(**cfg.get("settings",{})); halo=int(math.ceil(s.halo_um/min(mx,my)))
        def load_patch(index):
            x,y=map(int,coords[index]); core=(halo,halo,size,size)
            if read is not None:
                if x<0 or y<0 or x+size>reader.total_w or y+size>reader.total_h:
                    raise ValueError("Source core exceeds WSI bounds; audit coordinates first")
                image=read(x-halo,y-halo,size+2*halo)
            else:
                with Image.open(job["spec"]["images"][int(index)]["image_path"]) as image_file:
                    image=image_file.convert("RGB")
                if image.size!=(size,size): raise ValueError("Fixture image size mismatch")
                core=(0,0,size,size)
            masks,mask_meta=mask_bundle(cfg,sid,int(index))
            if isinstance(mask_meta,dict):
                binding=dict(slide_id=sid,patch_index=int(index),x_source=x,y_source=y,patch_size=size,coordinate_sha256=coord_hash)
                if any(mask_meta.get(k)!=v for k,v in binding.items()): raise ValueError("Mask coordinate binding mismatch")
            return x,y,image,core,masks
        start=time.monotonic();writer=None; count_ok=0
        preview_records=[];preview_buckets={};reason_counts={};lowest_blur=None
        preview_limit=cfg.get("preview_count",3)  # per PASS/failure-reason bucket
        with gzip.open(tmp,"wt",encoding="utf-8",newline="") as f:
            for n,index in enumerate(indices):
                x,y,image,core,masks=load_patch(index)
                vals,preview=describe_patch(np.asarray(image),core,mx,my,s,masks,diagnostics={} if smoke else None,
                                            he_applicable=job.get("he_applicable", True))
                row=dict(slide_id=sid,patch_index=int(index),x_source=x,y_source=y,
                         x_level0=x,y_level0=y,source_level=0,patch_size=size,
                         source_mpp_x=mx,source_mpp_y=my,mask_status="reviewed_supplied" if masks else "not_supplied",**vals)
                if writer is None:
                    writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                writer.writerow(clean(row));count_ok+=vals["qc_pass"]
                bucket="pass" if vals["qc_pass"] else vals["qc_reason"]
                reason_counts[bucket]=reason_counts.get(bucket,0)+1
                record=dict(patch_index=int(index),x_source=x,y_source=y,qc_pass=vals["qc_pass"],
                            qc_reason=vals["qc_reason"],tissue_fraction=vals["qc_tissue_fraction"],
                            blur_laplacian_var=vals["qc_blur_laplacian_var"])
                if preview_buckets.get(bucket,0) < preview_limit:
                    name=f"preview_{int(index):08d}.png"
                    Image.fromarray(preview).save(folder/name)
                    preview_records.append(dict(file=name,selection="first_per_QC_bucket",**record))
                    preview_buckets[bucket]=preview_buckets.get(bucket,0)+1
                blur=vals["qc_blur_laplacian_var"]
                if preview_limit and vals["qc_pass"] and math.isfinite(blur) and (lowest_blur is None or blur<lowest_blur["blur_laplacian_var"]):
                    name="preview_lowest_blur_pass.png"
                    Image.fromarray(preview).save(folder/name)
                    lowest_blur=dict(file=name,selection="lowest_blur_QC_pass_review_only",**record)
                if (n+1)%100==0:
                    print(f"[{sid}] patches {n+1}/{len(indices)}, QC {count_ok}, elapsed {time.monotonic()-start:.1f}s",flush=True)
        os.replace(tmp,final)
        if lowest_blur is not None: preview_records.append(lowest_blur)
        write_json(folder/"preview_index.json",preview_records)
        frame=pd.read_csv(final)
        review_rows=[]
        if smoke:
            for chosen in select_measurement_review(frame):
                index=chosen['patch_index'];x,y,image,core,masks=load_patch(index)
                diagnostic={}
                vals,_=describe_patch(np.asarray(image),core,mx,my,s,masks,diagnostics=diagnostic,
                                     he_applicable=job.get("he_applicable", True))
                packet=folder/'measurement_review'/f'{index:08d}';packet.mkdir(parents=True,exist_ok=True)
                raw=np.asarray(image.convert('RGB') if isinstance(image,Image.Image) else image)
                left,top,_,_=core;raw_core=raw[top:top+size,left:left+size]
                Image.fromarray(raw).save(packet/'raw_context.png')
                Image.fromarray(raw_core).save(packet/'raw_core.png')
                Image.fromarray(diagnostic['comparison']).save(packet/'comparison.png')
                direct_mae=None
                if read is not None:
                    direct=np.asarray(read(x,y,size))
                    if direct.shape!=raw_core.shape: raise ValueError('Direct/context raw read shape differs')
                    direct_mae=float(np.abs(direct.astype(float)-raw_core.astype(float)).mean())
                    Image.fromarray(direct).save(packet/'direct_core.png')
                record=dict(slide_id=sid,**chosen,x_level0=x,y_level0=y,patch_size=size,
                    source_mpp_x=mx,source_mpp_y=my,source=job['spec']['source'],
                    reader_class=f'{type(reader).__module__}.{type(reader).__name__}' if reader is not None else 'image_fixture',
                    reader_level=0,reader_dimensions=[reader.total_w,reader.total_h] if reader is not None else [size,size],
                    source_coordinate_sha256=coord_hash,source_paths=[v['path'] for v in inventory],
                    context_origin_x=x-left,context_origin_y=y-top,halo_pixels=left,
                    raw_core_is_reader_RGB_not_native_compressed_bytes=True,
                    direct_vs_context_core_mae=direct_mae,
                    independent_viewer_verified=False,measurement_status=vals['qc_measurement_status'],
                    basic_qc_pass=vals['qc_pass'],basic_qc_reason=vals['qc_reason'],
                    tissue_fraction=vals['qc_tissue_fraction'],blur_laplacian_var=vals['qc_blur_laplacian_var'],
                    gray_std_in_tissue=vals['qc_gray_std_in_tissue'],h_p90_minus_p10=vals['qc_h_p90_minus_p10'],
                    current_nucleus_count=diagnostic['current_count'],candidate_nucleus_count=diagnostic['candidate_count'],
                    candidate_used_for_features=False,candidate_support_min_pixels=diagnostic['candidate_support_min_pixels'],
                    review_directory=str(packet),review_source='',review_decision='pending',
                    note=('Candidate changes HE segmentation support only; not used for features. ' if job.get('he_applicable',True)
                          else 'Generic RGB/texture only; no HE nuclear segmentation. ')
                         +'Raw comparison is same-reader consistency, NOT independent viewer verification.')
                record['file_sha256']={p.name:sha(p) for p in sorted(packet.glob('*.png'))}
                write_json(packet/'review.json',record)
                review_rows.append(record)
            pd.DataFrame([{k:v for k,v in r.items() if k not in {'file_sha256','source_paths'}} for r in review_rows]).to_csv(folder/'measurement_review.csv',index=False,encoding='utf-8-sig')
        features=[c for c in frame.columns if c not in META and not c.startswith("qc_")]
        summary=aggregate(frame,features)
        summary.update(rook_moran(frame))
        identity={k:job[k] for k in ["slide_id","case_id","event_id","examid","biopsy_date","source_dataset"]}
        identity.update({k:job.get(k, "") for k in ["stain_group", "stain_label_raw", "stain_reference_source", "he_applicable", "linkage_status"]})
        write_json(folder/"summary.json",dict(**identity, **summary))
        receipt.update(patches_read=len(indices),patches_qc_pass=count_ok,seconds=time.monotonic()-start,
                       qc_reason_counts=reason_counts,preview_index_sha256=sha(folder/"preview_index.json"),
                       tissue_mask_method=s.tissue_mask_method if job.get("he_applicable", True) else "generic_mean_OD_candidate_v1",
                       blur_filter_enabled=s.min_blur_var>0,
                       measurement_status='unvalidated',measurement_review_patches=len(review_rows),
                       measurement_review_sha256=sha(folder/'measurement_review.csv') if smoke else None,
                       patch_csv_sha256=sha(final),summary_sha256=sha(folder/"summary.json"),feature_columns=features)
        write_json(done,receipt)
        return dict(receipt,status="completed")
    finally:
        if reader is not None and hasattr(reader,"close"): reader.close()


def feature_dictionary(features):
    rows=[]
    for name in features:
        group="nuclear" if name.startswith("nucleus") else name.split("_")[0]
        source="user_v2"
        interpretation="Exploratory image descriptor; not a clinical diagnosis"
        if name.startswith("lymph_like"):
            interpretation="Unvalidated morphology proxy, NOT confirmed lymphocyte count"
        if name.startswith(("annotated_", "compartment_")):
            source="CARE/CACHE-inspired measurement; NOT a reproduction"
            interpretation="Requires reviewed pixel/instance masks; absent mask is NA, not zero"
        if name.startswith("radius_") or name.startswith("delaunay") or name.startswith("nn_"):
            interpretation="Patch-local nuclear spatial descriptor; patch edge effects remain"
        unit="dimensionless or intensity"
        for token in ["per_mm2","um2","um","px2","px"]:
            if token in name: unit=token;break
        if name.endswith(("_cv","_skew","_kurtosis")): unit="dimensionless"
        equation="See the named feature function in bundled pathology_feature_extractor_v2.py"
        if "density_per_mm2" in name: equation="N / (tissue_pixel_count * analysis_mpp^2 / 1e6)"
        elif name=="nucleus_area_fraction": equation="core segmented nuclear pixels / core tissue pixels; NOT N/C ratio"
        elif name.startswith("nucleus_circularity"): equation="clip(4*pi*area/perimeter^2,0,1), then indicated nuclear statistic"
        elif name.startswith("nucleus_solidity"): equation="nuclear area / convex hull area, then indicated statistic"
        elif name=="nucleus_orientation_coherence": equation="abs(mean(exp(2j*orientation)))"
        elif name=="compartment_stroma_myocardium_ratio": equation="reviewed stroma area / reviewed myocardium area"
        elif name.startswith("compartment_") and "fraction" in name: equation="reviewed compartment pixels intersect tissue / tissue pixels"
        rows.append(dict(feature=name,group=group,unit=unit,source=source,interpretation=interpretation,
                         core_report=int(name in CORE_FEATURES),aggregation="valid_n;mean;median;sample std;min;max;p10;p90 over QC-passed cores",formula=equation))
    return rows


def make_all_stain_report(cohort, jobs, exclusions, cfg, out, signature, smoke,
                          all_features, slides, completed, receipts):
    """All original-feature WSI retained; only exact event links aggregated.

    Event rows are LONG by stain family. No cross-stain averaging and no
    artificial event identity for unlinked WSI. Main workbook is WSI-level.
    """
    common_core = ["gray_mean", "gray_std", "gray_entropy", "rgb_R_mean", "rgb_G_mean", "rgb_B_mean",
                   "glcm_contrast_mean", "glcm_homogeneity_mean", "glcm_entropy_mean", "fft_high_frequency_fraction"]
    selected = set(CORE_FEATURES + common_core)
    def compact(row):
        return {k:v for k,v in row.items() if "__" not in k or k.split("__")[0] in selected}
    summary_by_id = {s["slide_id"]: s for s in slides}
    exclusion_by_id = {r["slide_id"]:r["reason"] for r in exclusions}
    failures_path = out/"failures.json"
    failure_by_id = {r["slide_id"]:r["error"] for r in json.loads(failures_path.read_text())} if failures_path.exists() else {}
    eligible = {j["slide_id"] for j in jobs}
    inventory = []
    for row in cohort.to_dict("records"):
        sid = row["slide_id"]
        metadata = {k:row.get(k, "") for k in ["slide_id", "case_id", "event_id", "examid", "biopsy_date", "source_dataset",
                                                    "stain_group", "stain_label_raw", "stain_reference_source", "he_applicable", "linkage_status"]}
        metadata.update(summary_by_id.get(sid, {}))
        status = "complete" if sid in completed else "failed" if sid in failure_by_id else "not_completed" if sid in eligible else "unavailable"
        metadata.update(feature_status=status, unavailable_reason="" if sid in completed else exclusion_by_id.get(sid, failure_by_id.get(sid, "")),
                        smoke_only=smoke, measurement_status="unvalidated" if sid in completed else "not_processed")
        inventory.append(metadata)
    event_rows = []
    linked = cohort.loc[cohort.examid.ne("")]
    for (examid, stain), g in linked.groupby(["examid", "stain_group"], sort=True):
        ids = g.slide_id.tolist()
        available = [sid for sid in ids if sid in completed]
        row = dict(examid=examid, case_id=g.case_id.iloc[0], event_id=g.event_id.iloc[0],
                   biopsy_date=g.biopsy_date.iloc[0], stain_group=stain, slides_expected=len(ids),
                   slides_completed=len(available), feature_status="complete" if len(available)==len(ids) else "incomplete",
                   smoke_only=smoke, measurement_status="unvalidated")
        if available and len(available)==len(ids):
            f = pd.concat([pd.read_csv(completed[sid]/"patch_features.csv.gz") for sid in available], ignore_index=True)
            row.update(aggregate(f, all_features))
            if not row["patches_qc_pass"]: row["feature_status"] = "no_QC_pass"
        event_rows.append(row)
    pd.DataFrame(inventory).to_csv(out/"slide_features_all.csv", index=False, encoding="utf-8-sig")
    small = [compact(r) for r in inventory]
    pd.DataFrame(small).to_csv(out/"slide_features_core.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(event_rows, columns=None if event_rows else ["examid","stain_group","feature_status"]).to_csv(
        out/"biopsy_features_by_stain.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([compact(r) for r in event_rows]).to_csv(out/"biopsy_features_core_by_stain.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(exclusions, columns=["slide_id","examid","reason"]).to_csv(out/"exclusions.csv", index=False, encoding="utf-8-sig")
    cohort.to_csv(out/"cohort_metadata.csv", index=False, encoding="utf-8-sig")
    dictionary = feature_dictionary(all_features)
    for r in dictionary:
        name = r["feature"]
        r["applicability"] = ("all_stains_RGB_texture" if name.startswith(("gray", "rgb_", "glcm_", "lbp_", "gabor_", "wavelet_", "fft_"))
                              else "reviewed_external_masks_required" if name.startswith(("compartment_", "annotated_"))
                              else "confirmed_HE_only")
    pd.DataFrame(dictionary).to_csv(out/"feature_dictionary.csv", index=False, encoding="utf-8-sig")
    write_json(out/"candidate_feature_columns.json", [k for k in pd.DataFrame(small).columns if "__" in k and not k.endswith("__valid_n")])
    write_json(out/"inventory_summary.json", dict(original_feature_slides=len(cohort),
        mapped_slides=int(cohort.examid.ne("").sum()), unmapped_slides=int(cohort.examid.eq("").sum()),
        stain_counts=cohort.stain_group.value_counts().to_dict(), eligible=len(jobs), completed=len(completed),
        unavailable=len(exclusions), incomplete=len(jobs)-len(completed), smoke_only=smoke))
    write_json(out/"workbook_data.json", dict(version=VERSION, smoke_only=smoke, row_unit="WSI", main=small,
        slides=[{k:v for k,v in r.items() if "__" not in k} for r in inventory],
        events=[compact(r) for r in event_rows],
        dictionary=dictionary, exclusions=exclusions, receipts=[{k:v for k,v in r.items() if k not in {"feature_columns","source_inventory"}} for r in receipts],
        notes=[
            ["Population", "Every original .pt in configured baseline_dir/pt_files; not restricted to gold or stain. Missing source/failed WSI retained as status rows."],
            ["Rows", "Main: one WSI per row, including unlinked WSI. Exact cohort links only; missing case/event/date left blank. Outcomes kept separately."],
            ["Events", "biopsy_features_by_stain.csv: one event AND stain family per row; no cross-stain pooling. Incomplete event/stain groups not aggregated."],
            ["Features", "RGB/gray/GLCM/LBP/Gabor/wavelet/FFT for all stains. H/E components, heuristic nuclei and lymph-like proxies only for confirmed HE; otherwise NA, not zero."],
            ["Mask/QC", "Non-HE uses mean-OD foreground candidate without saturation constraint. Dark fraction flagged, not automatically rejected (dark stain vs artifact unresolved). HE retains previous mask. All measurements exploratory/unvalidated."],
            ["Heterogeneity", "IHC markers and special stains are heterogeneous. Family-level descriptive pooling is not marker positivity or comparable stain concentration. Keep WSI-level provenance; do not assume unknown is Other."],
            ["Coverage", "Full run covers nonoverlapping existing source patch coordinates, not previously unsampled WSI pixels. QC-failed patches retained with missing measurements."],
            ["Physical scale", "Requires source MPP or opted-in extent/H5 consistency; never assumes 40x=0.25. Failure listed, no fabricated values."],
            ["Smoke", "smoke_only=true uses limited patches and cannot be used as full extraction."],
            ["Privacy", "Linkage metadata can identify patients. Keep on authorized storage. Features are not validated clinical measurements."],
        ]))
    print(f"REPORT: {len(inventory)} WSI rows ({len(cohort)-len(linked)} unlinked); {len(completed)}/{len(jobs)} processed; {len(event_rows)} event/stain rows", flush=True)


def make_report(cohort,jobs,exclusions,cfg,out,signature,smoke=False):
    out=Path(out); all_features=None;slides=[];completed={};receipts=[]
    for job in jobs:
        folder=out/"slides"/job["slide_id"]
        done=folder/"complete.json"
        if not done.exists(): continue
        r=json.loads(done.read_text())
        if r["signature"]!=signature or r["smoke_only"]!=smoke: raise ValueError("Receipt config mismatch")
        if sha(folder/"patch_features.csv.gz")!=r["patch_csv_sha256"] or sha(folder/"summary.json")!=r["summary_sha256"]:
            raise ValueError("Output checksum mismatch")
        if all_features is None: all_features=r["feature_columns"]
        if all_features!=r["feature_columns"]: raise ValueError("Feature schema mismatch")
        s=json.loads((folder/"summary.json").read_text());s.pop("spec",None)
        slides.append(s);completed[job["slide_id"]]=folder;receipts.append(r)
    if cfg.get("stain_scope") == "all":
        return make_all_stain_report(cohort,jobs,exclusions,cfg,out,signature,smoke,
                                     all_features or [],slides,completed,receipts)
    if not completed: raise ValueError("No complete slide outputs")
    full=[];small=[]
    eligible={j["slide_id"] for j in jobs}
    core_features=[f for f in CORE_FEATURES if f in all_features]
    for examid,g in cohort.groupby("examid",sort=True):
        ids=g.slide_id.tolist(); wanted=[i for i in ids if i in eligible];available=[i for i in ids if i in completed]
        meta=dict(examid=examid,case_id=g.case_id.iloc[0],event_id=g.event_id.iloc[0],biopsy_date=g.biopsy_date.iloc[0],
                  slides_in_cohort=len(ids),confirmed_HE_eligible=len(wanted),HE_completed=len(available),
                  feature_status="no_eligible_HE" if not wanted else "complete" if len(available)==len(wanted) else "incomplete",
                  smoke_only=smoke,measurement_status='unvalidated' if available else 'not_processed')
        fullrow=dict(meta); corerow=dict(meta)
        # Incomplete events remain explicitly NA; do not train on partial events.
        if available and len(available)==len(wanted):
            f=pd.concat([pd.read_csv(completed[i]/"patch_features.csv.gz") for i in available],ignore_index=True)
            a=aggregate(f,all_features)
            fullrow.update({"HE__"+k:v for k,v in a.items()})
            keep_core={"patches_read","patches_qc_pass","analyzed_core_tissue_mm2","nucleus_pooled_density_per_mm2","lymph_like_pooled_density_per_mm2","nucleus_pooled_area_fraction"}
            corerow.update({"HE__"+k:v for k,v in a.items() if k in keep_core or any(k.startswith(x+"__") for x in core_features)})
            if a["patches_qc_pass"]==0: fullrow["feature_status"]=corerow["feature_status"]="no_QC_pass"
        full.append(fullrow);small.append(corerow)
    # Define complete wide schemas even if the first event has no HE.
    pd.DataFrame(full).to_csv(out/"biopsy_features_all.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(small).to_csv(out/"biopsy_features_core.csv",index=False,encoding="utf-8-sig")
    write_json(out/"candidate_feature_columns.json",[k for k in pd.DataFrame(small).columns if k.startswith("HE__") and "__valid_n" not in k])
    pd.DataFrame(slides).to_csv(out/"slide_features_all.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(exclusions,columns=["slide_id","examid","reason"]).to_csv(out/"exclusions.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(feature_dictionary(all_features)).to_csv(out/"feature_dictionary.csv",index=False,encoding="utf-8-sig")
    review_tables=[pd.read_csv(folder/'measurement_review.csv',keep_default_na=False)
                   for folder in completed.values() if (folder/'measurement_review.csv').exists()]
    if review_tables:
        pd.concat(review_tables,ignore_index=True).to_csv(out/'measurement_review_queue.csv',index=False,encoding='utf-8-sig')
    # Outcomes are not feature columns. Keep all task metadata separately.
    cohort.to_csv(out/"cohort_metadata.csv",index=False,encoding="utf-8-sig")
    write_json(out/"workbook_data.json",dict(version=VERSION,smoke_only=smoke,
        main=clean(pd.DataFrame(small).replace({np.nan:None}).to_dict("records")),
        slides=[{k:v for k,v in s.items() if k in {"slide_id","examid","case_id","event_id","biopsy_date","source_dataset","patches_read","patches_qc_pass","analyzed_core_tissue_mm2","patch_moran_I_lymph_like_density","patch_moran_nodes","patch_moran_edges"} or k in {"nucleus_pooled_density_per_mm2","lymph_like_pooled_density_per_mm2"}} for s in slides],
        dictionary=feature_dictionary(all_features),exclusions=exclusions,
        receipts=[{k:v for k,v in r.items() if k not in {"feature_columns","source_inventory"}} for r in receipts],
        notes=[
            ["Purpose","Research-only quantitative original H&E descriptors; not an ACR/AMR classifier."],
            ["Rows","One biopsy event per row. examid is a stable case/event/date token. No patient-wide future pooling."],
            ["Columns","HE__feature__stat resembles ECG example. Detailed all-feature and patch files remain in CSV."],
            ["Labels","Labels stay in cohort_metadata.csv; not mixed into features. No ECG/EHR data synthesized."],
            ["Image QC",f"Mask={cfg.get('settings',{}).get('tissue_mask_method','he_od_saturation_v1')}; fixed exploratory thresholds, not validated segmentation. Preview: original / green tissue mask / nuclei. Blur filter enabled={cfg.get('settings',{}).get('min_blur_var',0)>0}."],
            ["Nuclei","Otsu/watershed heuristic with halo; cyan nuclear boundaries/yellow lymphocyte-like boundaries. Not a validated cell classifier."],
            ["Area","Nonoverlap core sampled area, NOT entire WSI area. Separate sections may re-observe cells; pooled area is not a 3D tissue volume."],
            ["Physical scale","Header or explicitly opted-in extent/source-H5-consistent MPP; provenance in audit. Not independent calibration. Anisotropic resize; core extent rounding error <=1%."],
            ["Smoke QA","Spread-out source indices, not random prevalence sampling. Complete means eligible slides processed, NOT all WSI patches if smoke_only=true. Lowest-blur passing preview is review-only, not automatic exclusion."],
            ["Measurement validity","Basic QC PASS is NOT validated morphology. Measurement status remains unvalidated. Review queue ranks image scores, not rejection risk. Candidate support comparison is never used in feature columns; raw/context parity is same-reader, not independent viewer verification."],
            ["Spatial","Nuclear graphs are patch-local. Rook Moran is per slide on touching cores; no across-slide graph or inferential p-value."],
            ["Optional masks","myocardium/stroma/lymphocyte_instances from reviewed segmentation; no automatic myocardium/lymphocyte model supplied."],
            ["Unavailable","CD68/C4d vessel features, validated lymphocyte foci, edge-corrected Ripley/enrichment, and longitudinal changes are NOT implemented in v1."],
            ["Stains","Only confirmed HE selected; IHC/Others excluded from HE formulas but all cohort events retained. PLHE must be labeled HE in stain CSV."],
            ["Missing","NA means unavailable/undefined/QC failure; never replace all missing values with zero."],
            ["Modeling","Choose small descriptors; imputation/scaling/selection fit in training patients only. Test data not used to tune extraction/QC."],
            ["Privacy","case/event/date are linkage metadata, not model features. Keep report on authorized storage; not de-identified for public release."],
            ["Sources","User pathology_feature_extractor_v2.py; https://pubmed.ncbi.nlm.nih.gov/33982079/ ; https://pmc.ncbi.nlm.nih.gov/articles/PMC10940208/"],
        ]))
    print(f"REPORT: {len(full)} biopsy rows, {len(completed)}/{len(jobs)} HE slides, {len(all_features)} patch features",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["audit","smoke","extract","report","all"])
    p.add_argument("--config",default="pathomics_config.json")
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--limit",type=int,help="Audit only: check first N eligible slides; does not imply full audit passed")
    p.add_argument("--allow-verified-extent-mpp",action="store_true",
                   help="Opt in to missing DICOM PixelSpacing fallback: volume extent must match source-coordinate H5 MPP at level 0; not independent calibration")
    args=p.parse_args()
    if args.workers<1 or args.workers>16: p.error("workers must be 1..16")
    if args.limit is not None and (args.limit<1 or args.action!="audit"):
        p.error("--limit requires audit and a positive integer")
    cfg=load_config(args.config);smoke=args.action=="smoke"
    if args.allow_verified_extent_mpp: cfg["allow_verified_extent_mpp"] = True
    if cfg.get('review_only',False) and args.action in {'all','extract'}:
        p.error('This config is review_only: inspect raw-viewer parity and nuclear comparison before enabling full extraction in a separately reviewed config.')
    out=Path(cfg["output"])/("smoke" if smoke else "full")
    if out.resolve()==Path(cfg.get("baseline_dir",".")).resolve(): raise ValueError("Output cannot overwrite original features")
    cohort,jobs,exclusions=cohort_and_jobs(cfg)
    signature,record=config_signature(cfg,smoke)
    print(f"Population={cfg.get('population','gold_cohort')}; {len(cohort)} slides/{cohort.loc[cohort.examid.ne(''),'examid'].nunique()} linked events; eligible jobs {len(jobs)}; unavailable {len(exclusions)}; stains={cohort.stain_group.value_counts().to_dict()}",flush=True)
    if args.action=="audit":
        failures=[];audit_jobs=jobs[:args.limit] if args.limit is not None else jobs
        for i,job in enumerate(audit_jobs,1):
            print(f"[AUDIT {i}/{len(audit_jobs)}] {job['slide_id']} source={job['spec']['source']}",flush=True)
            try:
                info=process_slide(job,cfg,out,signature,audit=True)
                print(json.dumps(clean(info),ensure_ascii=False),flush=True)
            except Exception as exc:
                failures.append(job["slide_id"])
                print(f"[AUDIT_FAILED] {job['slide_id']}: {exc}",flush=True)
        print(f"Audit checked {len(audit_jobs)}/{len(jobs)} eligible slides; failed={len(failures)}. No files modified.",flush=True)
        if failures: raise SystemExit(1)
        print("Partial audit only." if len(audit_jobs)<len(jobs) else "Ready for smoke QA, not validated morphology.")
        return
    out.mkdir(parents=True,exist_ok=True)
    # Prevent concurrent writers to the same config/output (Linux and macOS).
    import fcntl
    process_lock=(out/".writer.lock").open("a")
    try: fcntl.flock(process_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError("Another pathomics command is writing this output directory")
    lock=out/"run_config.json"
    if lock.exists() and json.loads(lock.read_text())["signature"]!=signature:
        raise ValueError("Config/code/input changed. Use a NEW output directory.")
    if not lock.exists(): write_json(lock,dict(signature=signature,**record))
    if smoke:
        # Small deterministic per-source QA, without outcome/attention selection.
        chosen={};work=[]
        for j in jobs:
            source=(j["source_dataset"], j["stain_group"]) if cfg.get("stain_scope")=="all" else j["source_dataset"]
            if chosen.get(source,0) < cfg.get("smoke_slides_per_source",2):
                work.append(j);chosen[source]=chosen.get(source,0)+1
        print(f"Smoke: {len(work)} slides, up to {cfg.get('smoke_patches',64)} spread-out patches each; not full extraction.",flush=True)
    else: work=jobs
    if args.action!="report":
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(process_slide,j,cfg,str(out),signature,smoke):j for j in work}
            failed=[]
            for f in as_completed(futures):
                j=futures[f]
                try:
                    r=f.result();print(f"[{r['status']}] {j['slide_id']}: {r['patches_read']} patches; QC pass={r['patches_qc_pass']}; reasons={r.get('qc_reason_counts',{})}",flush=True)
                except Exception as e:
                    failed.append(dict(slide_id=j["slide_id"],error=str(e)))
                    print(f"[FAILED] {j['slide_id']}: {e}",flush=True)
        if failed:
            write_json(out/"failures.json",failed)
            if cfg.get("stain_scope") == "all":
                make_report(cohort,jobs,exclusions,cfg,out,signature,smoke)
            raise RuntimeError(f"{len(failed)} slide(s) failed; original data untouched. Fix then rerun same config.")
        write_json(out/"failures.json", [])
    make_report(cohort,jobs,exclusions,cfg,out,signature,smoke)
    print(f"Saved to {out}. CSVs open in Excel. For native XLSX: node export_pathomics.mjs {out}/workbook_data.json",flush=True)


if __name__=="__main__": main()
