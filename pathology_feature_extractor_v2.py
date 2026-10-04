#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pathology_feature_extractor_v2.py

H&E pathology tiles -> stain normalization -> QC -> quantitative features
-> tile / slide(biopsy) / patient level CSV  (multimodal dataset building)

Changes from v1
---------------
[Bug fixes]
- Watershed split nuclei were re-merged by measure.label(keep_mask).
  -> area filter now keeps labels (lookup table + relabel_sequential).
- Duplicate file stems across sub-folders overwrote each other.
  -> unique tile_uid (relative path) + patient_id / slide_id columns.
- Feature column names changed depending on nucleus count / mpp / Delaunay
  failure. -> fixed schema (px + um columns always present, NaN if unknown).
- Nuclei are segmented only inside the tissue mask.
- Orientation summarized with axial circular statistics (mean/std invalid).
- GLCM computed on tissue pixels only (background level removed) and
  entropy averaged per matrix (v1 summed 12 matrices).
- DAB channel dropped (meaningless for H&E).
- Global warnings suppression removed; skimage-version-safe morphology.

[New]
- Macenko stain normalization (fixed reference, or fitted from --stain_ref).
- QC features: blur (Laplacian variance, gradient), dark/artifact fraction,
  lumen/white-space fraction, qc_pass flag.
- Nuclear hematoxylin intensity, nuclear area fraction (N/C proxy).
- Lymphocyte-like nucleus HEURISTIC (small, round, dark nuclei; needs mpp).
- Multi-scale LBP, fast Gabor (cv2), radius graph components normalized.
- Slide(biopsy)-level and patient-level aggregation over QC-passed tiles.
- tile_manifest.csv (same QC'd / normalized tiles for the image branch).
- --workers (multiprocessing), --resume, run_config.json, feature_dictionary.csv

Input layout (default, when no manifest)
----------------------------------------
    input/
      <patient_id>/<slide_id>/<tile>.png      -> patient, slide from folders
      <patient_id>/<tile>.png                 -> slide_id = patient_id

Optional manifest CSV (--manifest) overrides IDs. Columns:
    rel_path (path relative to --input, e.g. P001/S01/tile_0001.png)  [required]
    patient_id, slide_id                                              [required]
    biopsy_date, mpp                                                  [optional]

IMPORTANT
---------
- Generic H&E feature extractor, NOT an ACR/AMR classifier. The
  lymphocyte-like counts are a morphology heuristic, not a validated cell
  classifier. For cell types use HoVer-Net / CellViT / StarDist.
- Use tiles at the same physical resolution (same MPP) across the cohort.
- For time-aware prediction, join at SLIDE (biopsy) level using biopsy_date.
  Patient-level output mixes biopsies from different time points.

Example
-------
python pathology_feature_extractor_v2.py \
    --input "D:/pathology/tiles" \
    --output "D:/pathology/features_v2" \
    --mpp 0.50 --workers 4 --save_normalized

Dependencies
------------
pip install numpy pandas scipy scikit-image>=0.19 opencv-python PyWavelets
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pywt
import scipy
import skimage

from scipy import ndimage as ndi
from scipy.spatial import cKDTree, Delaunay
from scipy.stats import skew, kurtosis

from skimage import color, feature, filters, measure, morphology, segmentation
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor_kernel
from skimage.io import imread, imsave
from skimage.segmentation import watershed

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
STAT_NAMES = ["mean", "std", "median", "iqr", "cv", "min", "max", "skew", "kurtosis"]

# Macenko reference (standard values from Macenko et al. 2009 implementations)
HE_REF = np.array([[0.5626, 0.2159],
                   [0.7201, 0.8012],
                   [0.4062, 0.5581]])
MAXC_REF = np.array([1.9705, 1.0308])

META_COLS = [
    "tile_uid", "rel_path", "patient_id", "slide_id", "biopsy_date",
    "width_px", "height_px", "mpp", "stain_norm_ok", "qc_pass", "qc_reason",
]


# =====================================================================
# Utility
# =====================================================================
def ensure_rgb_uint8(img):
    if img is None:
        raise ValueError("Image is None")
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = color.rgba2rgb(img)  # float [0,1]
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img / 257.0).clip(0, 255).astype(np.uint8)
    if np.issubdtype(img.dtype, np.floating):
        scale = 255.0 if img.max() <= 1.0 else 1.0
        return (img * scale).clip(0, 255).astype(np.uint8)
    x = img.astype(np.float64)
    x = (x - x.min()) / max(x.max() - x.min(), 1e-12) * 255.0
    return x.astype(np.uint8)


def to_py(v):
    """numpy scalar -> python scalar (json / csv safe)."""
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return int(v)
    return v


def nan_stats(prefix):
    return {f"{prefix}_{s}": np.nan for s in STAT_NAMES}


def safe_stats(values, prefix):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return nan_stats(prefix)
    mean_v = float(np.mean(x))
    std_v = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    q25, q75 = np.percentile(x, [25, 75])
    const = np.allclose(x, x[0])
    return {
        f"{prefix}_mean": mean_v,
        f"{prefix}_std": std_v,
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_cv": float(std_v / mean_v) if abs(mean_v) > 1e-12 else np.nan,
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_skew": float(skew(x, bias=False)) if len(x) >= 3 and not const else np.nan,
        f"{prefix}_kurtosis": float(kurtosis(x, bias=False)) if len(x) >= 4 and not const else np.nan,
    }


def stats_px_um(values_px, prefix, mpp, power=1):
    """Stats in pixel units and physical units (um or um^2). Fixed schema."""
    unit_px = "px" if power == 1 else "px2"
    unit_um = "um" if power == 1 else "um2"
    out = safe_stats(values_px, f"{prefix}_{unit_px}")
    if mpp is not None and np.isfinite(mpp):
        out.update(safe_stats(np.asarray(values_px, float) * (mpp ** power), f"{prefix}_{unit_um}"))
    else:
        out.update(nan_stats(f"{prefix}_{unit_um}"))
    return out


def entropy_from_array(arr, bins=64):
    x = np.asarray(arr, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    hist, _ = np.histogram(x, bins=bins)
    p = hist.astype(float)
    if p.sum() == 0:
        return np.nan
    p /= p.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# ---- skimage-version-independent morphology --------------------------
def remove_small(mask, min_size):
    """Remove connected components with area < min_size."""
    mask = np.asarray(mask, dtype=bool)
    if min_size <= 1 or not mask.any():
        return mask
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[lab]


def fill_small_holes(mask, max_hole):
    """Fill holes with area < max_hole (holes touching the border are kept)."""
    mask = np.asarray(mask, dtype=bool)
    inv = ~mask
    lab, n = ndi.label(inv)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    small = sizes < max_hole
    small[0] = False
    small[border] = False
    return mask | small[lab]


def disk(r):
    return morphology.disk(r).astype(bool)


# =====================================================================
# Stain normalization (Macenko)
# =====================================================================
def macenko_stain_matrix(rgb, mask, Io=240, alpha=1.0, beta=0.15, min_pixels=500):
    od = -np.log((rgb.reshape(-1, 3).astype(np.float64) + 1.0) / Io)
    sel = (np.asarray(mask).ravel() & np.all(od >= beta, axis=1))
    od_hat = od[sel]
    if len(od_hat) < min_pixels:
        return None, None, od, sel

    _, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
    plane = eigvecs[:, 1:3]
    t_hat = od_hat @ plane
    phi = np.arctan2(t_hat[:, 1], t_hat[:, 0])
    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)
    v_min = plane @ np.array([np.cos(min_phi), np.sin(min_phi)])
    v_max = plane @ np.array([np.cos(max_phi), np.sin(max_phi)])

    # Hematoxylin has the larger red-channel OD component
    he = np.array([v_min, v_max]).T if v_min[0] > v_max[0] else np.array([v_max, v_min]).T
    for k in range(2):
        if he[:, k].sum() < 0:
            he[:, k] *= -1
    he /= np.linalg.norm(he, axis=0, keepdims=True)

    conc = np.linalg.lstsq(he, od.T, rcond=None)[0]
    max_c = np.percentile(conc[:, sel], 99, axis=1)
    return he, max_c, od, sel


def macenko_normalize(rgb, mask, he_ref=HE_REF, maxc_ref=MAXC_REF, Io=240):
    """Return (normalized_rgb_uint8, ok_flag)."""
    try:
        he, max_c, od, _ = macenko_stain_matrix(rgb, mask, Io=Io)
        if he is None or not np.all(np.isfinite(he)) or np.any(max_c <= 1e-6):
            return rgb.copy(), 0
        conc = np.linalg.lstsq(he, od.T, rcond=None)[0]
        conc *= (maxc_ref / max_c)[:, None]
        norm = Io * np.exp(-he_ref @ conc)
        norm = norm.T.reshape(rgb.shape).clip(0, 255).astype(np.uint8)
        return norm, 1
    except Exception:
        return rgb.copy(), 0


def fit_reference_stain(ref_path):
    rgb = ensure_rgb_uint8(imread(str(ref_path)))
    mask, _, _ = make_tissue_mask(rgb)
    he, max_c, _, _ = macenko_stain_matrix(rgb, mask)
    if he is None:
        raise ValueError(f"Could not fit stain matrix from reference: {ref_path}")
    return he, max_c


# =====================================================================
# Tissue mask + QC
# =====================================================================
def make_tissue_mask(rgb, min_object=500, max_fill_hole=500, lumen_min_px=50):
    """
    Returns
      tissue_mask : cleaned tissue mask (small holes filled) used for features
      raw_mask    : mask before hole filling
      lumen_mask  : enclosed white regions (vessel lumen / adipose / tears)
    """
    hsv = color.rgb2hsv(rgb.astype(np.float32) / 255.0)
    raw = (hsv[..., 1] > 0.04) & (hsv[..., 2] < 0.98)
    raw = remove_small(raw, min_object)

    # all enclosed holes -> lumen candidates
    filled_all = ndi.binary_fill_holes(raw)
    lumen = remove_small(filled_all & ~raw, lumen_min_px)

    tissue = fill_small_holes(raw, max_fill_hole)
    tissue = ndi.binary_closing(tissue, structure=disk(3))
    tissue &= filled_all | raw
    return tissue, raw, lumen


def extract_qc_features(rgb, tissue_mask, lumen_mask):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    total = tissue_mask.size
    t_px = int(tissue_mask.sum())
    out = {
        "qc_tissue_area_px": t_px,
        "qc_tissue_fraction": t_px / max(total, 1),
        "qc_lumen_fraction": float(lumen_mask.sum()) / max(t_px + lumen_mask.sum(), 1),
    }
    inner = ndi.binary_erosion(tissue_mask, structure=disk(2)) if t_px else tissue_mask
    if inner.sum() < 100:
        inner = tissue_mask
    if inner.sum() > 0:
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        out["qc_blur_laplacian_var"] = float(np.var(lap[inner]))
        out["qc_gradient_mean"] = float(np.mean(np.hypot(gx, gy)[inner]))
        hsv = color.rgb2hsv(rgb.astype(np.float32) / 255.0)
        out["qc_dark_pixel_fraction"] = float(np.mean(hsv[..., 2][inner] < 0.20))
        out["qc_saturation_mean"] = float(np.mean(hsv[..., 1][inner]))
    else:
        for k in ["qc_blur_laplacian_var", "qc_gradient_mean",
                  "qc_dark_pixel_fraction", "qc_saturation_mean"]:
            out[k] = np.nan
    return out


# =====================================================================
# Nuclear segmentation
# =====================================================================
def segment_nuclei(h_channel, tissue_mask, min_area=25, max_area=2500, min_distance=4):
    """
    Hematoxylin (from normalized RGB) + Otsu inside tissue + distance watershed.
    Returns int32 label image (0 = background).
    """
    empty = np.zeros(h_channel.shape, dtype=np.int32)
    if tissue_mask.sum() < 100:
        return empty, np.nan

    h = filters.gaussian(h_channel, sigma=1.0, preserve_range=True)
    vals = h[tissue_mask]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0 or np.allclose(vals.min(), vals.max()):
        return empty, np.nan

    th = float(filters.threshold_otsu(vals))
    nuc = (h > th) & tissue_mask
    nuc = ndi.binary_opening(nuc, structure=disk(1))
    nuc = remove_small(nuc, min_area)
    nuc = fill_small_holes(nuc, 20)
    if not nuc.any():
        return empty, th

    dist = ndi.distance_transform_edt(nuc)
    dist_s = filters.gaussian(dist, sigma=1.0, preserve_range=True)
    cc, _ = ndi.label(nuc)
    coords = feature.peak_local_max(
        dist_s, labels=cc, min_distance=min_distance, exclude_border=False
    )
    markers = np.zeros(nuc.shape, dtype=np.int32)
    if len(coords):
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    # components without a peak still get a marker (their centroid-ish max)
    missing = np.setdiff1d(np.unique(cc[nuc]), np.unique(cc[markers > 0]))
    if len(missing):
        pos = ndi.maximum_position(dist, labels=cc, index=missing)
        start = markers.max() + 1
        for k, (r, c) in enumerate(pos):
            markers[int(r), int(c)] = start + k

    labels = watershed(-dist_s, markers, mask=nuc)

    # ---- v1 bug fix: filter by area WITHOUT re-labeling the binary mask ----
    areas = np.bincount(labels.ravel())
    bad = (areas < min_area) | (areas > max_area)
    bad[0] = True
    lut = np.where(bad, 0, np.arange(len(areas)))
    labels = lut[labels]
    labels, _, _ = segmentation.relabel_sequential(labels)
    return labels.astype(np.int32), th


# =====================================================================
# Nuclear morphology / intensity
# =====================================================================
NUC_LINEAR = ["perimeter", "major_axis", "minor_axis", "equiv_diameter"]
NUC_UNITLESS = ["eccentricity", "solidity", "extent", "circularity", "aspect_ratio",
                "h_mean", "h_std"]


def per_nucleus_table(labels, h_channel):
    cols = ["label", "centroid_y", "centroid_x", "area_px", "perimeter_px", "major_axis_px",
            "minor_axis_px", "equiv_diameter_px", "eccentricity", "solidity", "extent",
            "orientation", "circularity", "aspect_ratio", "h_mean", "h_std"]
    n = int(labels.max())
    if n == 0:
        return pd.DataFrame(columns=cols)

    p = measure.regionprops_table(
        labels,
        properties=("label", "centroid", "area", "perimeter", "axis_major_length",
                    "axis_minor_length", "equivalent_diameter_area", "eccentricity",
                    "solidity", "extent", "orientation"),
    )
    df = pd.DataFrame({
        "label": p["label"],
        "centroid_y": p["centroid-0"],
        "centroid_x": p["centroid-1"],
        "area_px": p["area"],
        "perimeter_px": p["perimeter"],
        "major_axis_px": p["axis_major_length"],
        "minor_axis_px": p["axis_minor_length"],
        "equiv_diameter_px": p["equivalent_diameter_area"],
        "eccentricity": p["eccentricity"],
        "solidity": p["solidity"],
        "extent": p["extent"],
        "orientation": p["orientation"],
    })
    df["circularity"] = np.clip(
        4.0 * np.pi * df["area_px"] / np.maximum(df["perimeter_px"] ** 2, 1e-12), 0, 1
    )
    df["aspect_ratio"] = df["major_axis_px"] / np.maximum(df["minor_axis_px"], 1e-12)
    idx = df["label"].values
    df["h_mean"] = ndi.mean(h_channel, labels=labels, index=idx)
    df["h_std"] = ndi.standard_deviation(h_channel, labels=labels, index=idx)
    return df[cols]


def extract_nuclear_features(nuc_df, tissue_px, mpp):
    out = {}
    n = len(nuc_df)
    out["nucleus_count"] = n
    out["nucleus_density_per_mpx_tissue"] = n / (tissue_px / 1e6) if tissue_px > 0 else np.nan
    if mpp is not None and tissue_px > 0:
        out["nucleus_density_per_mm2"] = n / (tissue_px * (mpp / 1000.0) ** 2)
    else:
        out["nucleus_density_per_mm2"] = np.nan
    out["nucleus_area_fraction"] = (
        float(nuc_df["area_px"].sum()) / tissue_px if tissue_px > 0 else np.nan
    )

    out.update(stats_px_um(nuc_df["area_px"].values, "nucleus_area", mpp, power=2))
    for c in NUC_LINEAR:
        out.update(stats_px_um(nuc_df[f"{c}_px"].values, f"nucleus_{c}", mpp, power=1))
    for c in NUC_UNITLESS:
        out.update(safe_stats(nuc_df[c].values, f"nucleus_{c}"))

    # Orientation is axial (theta ~ theta + pi): use doubled-angle statistics
    if n >= 2:
        z = np.exp(1j * 2.0 * nuc_df["orientation"].values.astype(float))
        R = float(np.abs(np.mean(z)))
        out["nucleus_orientation_coherence"] = R
        out["nucleus_orientation_circstd"] = (
            float(np.sqrt(-2.0 * np.log(max(R, 1e-12))) / 2.0)
        )
    else:
        out["nucleus_orientation_coherence"] = np.nan
        out["nucleus_orientation_circstd"] = np.nan
    return out


def extract_lymph_like_features(nuc_df, tissue_px, mpp, args):
    """
    HEURISTIC: small, round, solid, hyperchromatic nuclei (lymphocyte-like).
    Not a validated classifier. Requires mpp.
    """
    keys = ["lymph_like_count", "lymph_like_fraction", "lymph_like_density_per_mm2",
            "lymph_like_nn_distance_um_mean", "lymph_like_cluster_fraction"]
    out = {k: np.nan for k in keys}
    if mpp is None or not np.isfinite(mpp) or len(nuc_df) == 0 or tissue_px <= 0:
        return out, np.zeros(len(nuc_df), dtype=bool)

    area_um2 = nuc_df["area_px"].values * mpp ** 2
    h_ref = np.nanmedian(nuc_df["h_mean"].values)
    is_l = (
        (area_um2 >= args.lymph_min_um2) & (area_um2 <= args.lymph_max_um2)
        & (nuc_df["circularity"].values >= args.lymph_min_circularity)
        & (nuc_df["solidity"].values >= 0.90)
        & (nuc_df["h_mean"].values >= h_ref)
    )
    n_l = int(is_l.sum())
    tissue_mm2 = tissue_px * (mpp / 1000.0) ** 2
    out["lymph_like_count"] = n_l
    out["lymph_like_fraction"] = n_l / len(nuc_df)
    out["lymph_like_density_per_mm2"] = n_l / tissue_mm2

    if n_l >= 2:
        pts = nuc_df.loc[is_l, ["centroid_x", "centroid_y"]].values * mpp
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=2)
        out["lymph_like_nn_distance_um_mean"] = float(np.mean(d[:, 1]))
        neigh = tree.query_ball_point(pts, r=args.lymph_cluster_radius_um)
        n_neigh = np.array([len(v) - 1 for v in neigh])
        out["lymph_like_cluster_fraction"] = float(np.mean(n_neigh >= 3))
    return out, is_l


# =====================================================================
# Intensity (normalized H / E)
# =====================================================================
def extract_stain_intensity_features(rgb_norm, hed, tissue_mask):
    mask = tissue_mask if tissue_mask.sum() > 0 else np.ones(tissue_mask.shape, bool)
    gray = color.rgb2gray(rgb_norm.astype(np.float32) / 255.0)
    out = {}
    out.update(safe_stats(gray[mask], "gray"))
    out["gray_entropy"] = entropy_from_array(gray[mask])
    for idx, name in [(0, "hematoxylin"), (1, "eosin")]:
        ch = hed[..., idx][mask]
        out.update(safe_stats(ch, name))
        out[f"{name}_entropy"] = entropy_from_array(ch)
    h_m, e_m = np.mean(hed[..., 0][mask]), np.mean(hed[..., 1][mask])
    out["eosin_hematoxylin_ratio"] = float(e_m / h_m) if abs(h_m) > 1e-12 else np.nan
    for idx, name in enumerate(["R", "G", "B"]):
        ch = rgb_norm[..., idx].astype(float)[mask]
        out[f"rgb_{name}_mean"] = float(np.mean(ch))
        out[f"rgb_{name}_std"] = float(np.std(ch))
    return out


# =====================================================================
# Texture
# =====================================================================
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]


def extract_glcm_features(gray_u8, tissue_mask, levels=32):
    out = {}
    for prop in GLCM_PROPS:
        out[f"glcm_{prop}_mean"] = np.nan
        out[f"glcm_{prop}_std"] = np.nan
    out["glcm_entropy_mean"] = np.nan
    if tissue_mask.sum() < 100:
        return out

    # tissue pixels -> 1..levels, background -> 0 (dropped afterwards)
    q = np.floor(gray_u8.astype(np.float32) / 256.0 * levels).astype(np.int32) + 1
    q = np.clip(q, 1, levels)
    q[~tissue_mask] = 0
    glcm = graycomatrix(
        q.astype(np.uint8),
        distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels + 1,
        symmetric=True,
        normed=False,
    ).astype(np.float64)[1:, 1:, :, :]
    s = glcm.sum(axis=(0, 1), keepdims=True)
    valid = (s > 0).ravel()
    glcm = np.divide(glcm, s, out=np.zeros_like(glcm), where=s > 0)

    for prop in GLCM_PROPS:
        vals = graycoprops(glcm, prop).ravel()[valid]
        out[f"glcm_{prop}_mean"] = float(np.nanmean(vals)) if len(vals) else np.nan
        out[f"glcm_{prop}_std"] = float(np.nanstd(vals)) if len(vals) else np.nan

    ents = []
    for d in range(glcm.shape[2]):
        for a in range(glcm.shape[3]):
            p = glcm[:, :, d, a]
            p = p[p > 0]
            if len(p):
                ents.append(-(p * np.log2(p)).sum())
    out["glcm_entropy_mean"] = float(np.mean(ents)) if ents else np.nan
    return out


LBP_SCALES = [(8, 1), (16, 2)]


def extract_lbp_features(gray_u8, tissue_mask):
    out = {}
    for P, R in LBP_SCALES:
        keys = [f"lbp_P{P}R{R}_hist_{i}" for i in range(P + 2)] + [f"lbp_P{P}R{R}_entropy"]
        inner = ndi.binary_erosion(tissue_mask, structure=disk(R + 1))
        if inner.sum() < 100:
            out.update({k: np.nan for k in keys})
            continue
        lbp = local_binary_pattern(gray_u8, P=P, R=R, method="uniform")
        hist, _ = np.histogram(lbp[inner], bins=np.arange(0, P + 3))
        hist = hist.astype(float) / max(hist.sum(), 1)
        for i, v in enumerate(hist):
            out[f"lbp_P{P}R{R}_hist_{i}"] = float(v)
        p = hist[hist > 0]
        out[f"lbp_P{P}R{R}_entropy"] = float(-(p * np.log2(p)).sum())
    return out


GABOR_FREQS = [0.10, 0.20]
GABOR_THETAS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def fill_background(gray_f, tissue_mask):
    work = gray_f.copy()
    if tissue_mask.sum() > 0:
        work[~tissue_mask] = np.median(gray_f[tissue_mask])
    return work


def extract_gabor_features(gray_f, tissue_mask):
    out = {}
    inner = ndi.binary_erosion(tissue_mask, structure=disk(3))
    if inner.sum() < 100:
        inner = tissue_mask
    work = fill_background(gray_f, tissue_mask).astype(np.float32)
    for freq in GABOR_FREQS:
        for i, theta in enumerate(GABOR_THETAS):
            prefix = f"gabor_f{freq:.2f}_theta{i}"
            if inner.sum() == 0:
                for s in ["mean", "std", "energy"]:
                    out[f"{prefix}_{s}"] = np.nan
                continue
            k = gabor_kernel(freq, theta=theta)
            re = cv2.filter2D(work, cv2.CV_32F, np.real(k).astype(np.float32),
                              borderType=cv2.BORDER_REFLECT)
            im = cv2.filter2D(work, cv2.CV_32F, np.imag(k).astype(np.float32),
                              borderType=cv2.BORDER_REFLECT)
            mag = np.sqrt(re ** 2 + im ** 2)[inner]
            out[f"{prefix}_mean"] = float(np.mean(mag))
            out[f"{prefix}_std"] = float(np.std(mag))
            out[f"{prefix}_energy"] = float(np.mean(mag ** 2))
    return out


def tissue_crop(gray_f, tissue_mask):
    if tissue_mask.any():
        rows, cols = np.where(tissue_mask)
        r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        work = gray_f[r0:r1, c0:c1].copy()
        cm = tissue_mask[r0:r1, c0:c1]
        work[~cm] = np.median(work[cm])
        return work, cm
    return gray_f.copy(), np.ones(gray_f.shape, bool)


def extract_wavelet_features(gray_f, tissue_mask):
    keys = ["wavelet_A2_energy", "wavelet_A2_entropy"]
    for lv in [2, 1]:
        for b in ["H", "V", "D"]:
            keys += [f"wavelet_L{lv}_{b}_{s}" for s in ["energy", "std", "entropy"]]
    out = {k: np.nan for k in keys}
    work, _ = tissue_crop(gray_f, tissue_mask)
    if min(work.shape) < 16:
        return out
    coeffs = pywt.wavedec2(work, wavelet="db2", level=2)
    out["wavelet_A2_energy"] = float(np.mean(coeffs[0] ** 2))
    out["wavelet_A2_entropy"] = entropy_from_array(coeffs[0])
    for lv, detail in zip([2, 1], coeffs[1:]):
        for b, band in zip(["H", "V", "D"], detail):
            out[f"wavelet_L{lv}_{b}_energy"] = float(np.mean(band ** 2))
            out[f"wavelet_L{lv}_{b}_std"] = float(np.std(band))
            out[f"wavelet_L{lv}_{b}_entropy"] = entropy_from_array(band)
    return out


def extract_fft_features(gray_f, tissue_mask, mpp=None, low_cutoff=0.05,
                         mid_cutoff=0.15, radial_bins=128):
    """Global 2-D FFT features (logic kept from v1; units cycles/pixel)."""
    names = [
        "fft_total_power", "fft_low_frequency_power", "fft_mid_frequency_power",
        "fft_high_frequency_power", "fft_low_frequency_fraction",
        "fft_mid_frequency_fraction", "fft_high_frequency_fraction", "fft_low_high_ratio",
        "fft_spectral_centroid_cpp", "fft_spectral_spread_cpp", "fft_spectral_entropy",
        "fft_dominant_spatial_frequency_cpp", "fft_spectral_centroid_cycles_per_mm",
        "fft_spectral_spread_cycles_per_mm", "fft_dominant_spatial_frequency_cycles_per_mm",
    ]
    out = {n: np.nan for n in names}
    mask = np.asarray(tissue_mask, bool)
    if mask.any():
        work, cm = tissue_crop(gray_f.astype(np.float64), mask)
        center = float(np.mean(work[cm]))
    else:
        work = gray_f.astype(np.float64).copy()
        center = float(np.mean(work))
    if min(work.shape) < 8 or not np.isfinite(work).all():
        return out

    work = (work - center) * np.outer(np.hanning(work.shape[0]), np.hanning(work.shape[1]))
    power = np.abs(np.fft.fftshift(np.fft.fft2(work, norm="ortho"))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(work.shape[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(work.shape[1]))
    fxg, fyg = np.meshgrid(fx, fy)
    rf = np.sqrt(fxg ** 2 + fyg ** 2)
    non_dc = rf > 0
    p, f = power[non_dc], rf[non_dc]
    n_coef = float(power.size)
    total = float(p.sum())
    if not np.isfinite(total) or total <= 1e-20:
        for k in names[:4]:
            out[k] = 0.0
        return out

    lo, mi, hi = f <= low_cutoff, (f > low_cutoff) & (f <= mid_cutoff), f > mid_cutoff
    ls, ms, hs = float(p[lo].sum()), float(p[mi].sum()), float(p[hi].sum())
    out.update({
        "fft_total_power": total / n_coef,
        "fft_low_frequency_power": ls / n_coef,
        "fft_mid_frequency_power": ms / n_coef,
        "fft_high_frequency_power": hs / n_coef,
        "fft_low_frequency_fraction": ls / total,
        "fft_mid_frequency_fraction": ms / total,
        "fft_high_frequency_fraction": hs / total,
        "fft_low_high_ratio": ls / max(hs, 1e-20),
    })
    w = p / total
    centroid = float(np.sum(f * w))
    spread = float(np.sqrt(np.sum(((f - centroid) ** 2) * w)))
    pw = w[w > 0]
    ent = float(-np.sum(pw * np.log2(pw)))
    if len(pw) > 1:
        ent /= float(np.log2(len(pw)))

    edges = np.linspace(0.0, float(f.max()), radial_bins + 1)
    bi = np.clip(np.digitize(f, edges) - 1, 0, radial_bins - 1)
    ssum = np.bincount(bi, weights=p, minlength=radial_bins)
    scnt = np.bincount(bi, minlength=radial_bins)
    smean = np.divide(ssum, scnt, out=np.zeros_like(ssum), where=scnt > 0)
    db = int(np.argmax(smean))
    dom = float(0.5 * (edges[db] + edges[db + 1]))

    out["fft_spectral_centroid_cpp"] = centroid
    out["fft_spectral_spread_cpp"] = spread
    out["fft_spectral_entropy"] = ent
    out["fft_dominant_spatial_frequency_cpp"] = dom
    if mpp is not None and np.isfinite(mpp):
        k = 1000.0 / mpp
        out["fft_spectral_centroid_cycles_per_mm"] = centroid * k
        out["fft_spectral_spread_cycles_per_mm"] = spread * k
        out["fft_dominant_spatial_frequency_cycles_per_mm"] = dom * k
    return out


# =====================================================================
# Spatial / graph (fixed schema)
# =====================================================================
def extract_spatial_features(nuc_df, mpp):
    out = {}
    out.update(stats_px_um([], "nn_distance", mpp))
    out.update(stats_px_um([], "delaunay_edge", mpp))
    out.update(nan_stats("delaunay_degree"))
    for k in ["radius_graph_radius_px", "radius_graph_radius_um", "radius_graph_mean_degree",
              "radius_graph_degree_std", "radius_graph_components_per_100_nuclei",
              "radius_graph_largest_component_fraction", "radius_graph_isolated_fraction"]:
        out[k] = np.nan

    n = len(nuc_df)
    if n < 2:
        return out
    pts = nuc_df[["centroid_x", "centroid_y"]].values.astype(float)
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    nn = d[:, 1]
    out.update(stats_px_um(nn, "nn_distance", mpp))

    radius = max(float(np.median(nn) * 2.0), 1.0)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    deg = np.bincount(pairs.ravel(), minlength=n) if len(pairs) else np.zeros(n, int)
    out["radius_graph_radius_px"] = radius
    out["radius_graph_radius_um"] = radius * mpp if mpp is not None else np.nan
    out["radius_graph_mean_degree"] = float(np.mean(deg))
    out["radius_graph_degree_std"] = float(np.std(deg))
    out["radius_graph_isolated_fraction"] = float(np.mean(deg == 0))

    if len(pairs):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        n_comp, comp = connected_components(g, directed=False)
    else:
        n_comp, comp = n, np.arange(n)
    out["radius_graph_components_per_100_nuclei"] = 100.0 * n_comp / n
    out["radius_graph_largest_component_fraction"] = float(np.bincount(comp).max() / n)

    if n >= 3:
        try:
            tri = Delaunay(pts)
            s = tri.simplices
            e = np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])
            e = np.unique(np.sort(e, axis=1), axis=0)
            lengths = np.linalg.norm(pts[e[:, 0]] - pts[e[:, 1]], axis=1)
            out.update(stats_px_um(lengths, "delaunay_edge", mpp))
            out.update(safe_stats(np.bincount(e.ravel(), minlength=n), "delaunay_degree"))
        except Exception:
            pass  # collinear points etc. -> NaN already set
    return out


# =====================================================================
# Per-tile processing
# =====================================================================
def save_qc_overlay(rgb, labels, lymph_mask_img, path):
    ov = segmentation.mark_boundaries(rgb.astype(np.float32) / 255.0, labels,
                                      color=(1, 0, 0), mode="outer")
    if lymph_mask_img is not None and lymph_mask_img.any():
        ov = segmentation.mark_boundaries(ov, lymph_mask_img.astype(np.int32),
                                          color=(0, 0.8, 1), mode="outer")
    imsave(str(path), (ov * 255).clip(0, 255).astype(np.uint8), check_contrast=False)


def process_tile(task):
    """task: dict (picklable). Returns (row_dict, error_or_None)."""
    args = task["args"]
    rel_path = task["rel_path"]
    tile_uid = task["tile_uid"]
    mpp = task["mpp"]
    out_dir = Path(args.output)
    cv2.setNumThreads(1)

    try:
        rgb = ensure_rgb_uint8(imread(task["abs_path"]))
        tissue_mask, _, lumen_mask = make_tissue_mask(rgb, lumen_min_px=args.lumen_min_px)

        row = {
            "tile_uid": tile_uid, "rel_path": rel_path,
            "patient_id": task["patient_id"], "slide_id": task["slide_id"],
            "biopsy_date": task.get("biopsy_date", ""),
            "width_px": int(rgb.shape[1]), "height_px": int(rgb.shape[0]),
            "mpp": mpp if mpp is not None else np.nan,
        }

        # QC on the original image
        row.update(extract_qc_features(rgb, tissue_mask, lumen_mask))

        # Stain normalization
        if args.no_stain_norm:
            rgb_n, ok = rgb, 0
        else:
            he_ref = np.array(task["he_ref"]) if task.get("he_ref") is not None else HE_REF
            maxc_ref = np.array(task["maxc_ref"]) if task.get("maxc_ref") is not None else MAXC_REF
            rgb_n, ok = macenko_normalize(rgb, tissue_mask, he_ref, maxc_ref)
        row["stain_norm_ok"] = ok

        hed = color.rgb2hed(rgb_n.astype(np.float32) / 255.0)
        h_ch = hed[..., 0]
        gray_u8 = cv2.cvtColor(rgb_n, cv2.COLOR_RGB2GRAY)
        gray_f = gray_u8.astype(np.float32) / 255.0
        tissue_px = int(tissue_mask.sum())

        labels, h_th = segment_nuclei(
            h_ch, tissue_mask, min_area=args.min_nucleus_area,
            max_area=args.max_nucleus_area, min_distance=args.min_distance,
        )
        row["seg_h_threshold"] = h_th

        nuc_df = per_nucleus_table(labels, h_ch)
        row.update(extract_nuclear_features(nuc_df, tissue_px, mpp))
        lymph, is_l = extract_lymph_like_features(nuc_df, tissue_px, mpp, args)
        row.update(lymph)
        row.update(extract_stain_intensity_features(rgb_n, hed, tissue_mask))
        row.update(extract_glcm_features(gray_u8, tissue_mask))
        row.update(extract_lbp_features(gray_u8, tissue_mask))
        row.update(extract_gabor_features(gray_f, tissue_mask))
        row.update(extract_wavelet_features(gray_f, tissue_mask))
        row.update(extract_fft_features(gray_f, tissue_mask, mpp=mpp,
                                        low_cutoff=args.fft_low_cutoff,
                                        mid_cutoff=args.fft_mid_cutoff))
        row.update(extract_spatial_features(nuc_df, mpp))

        # QC decision
        reasons = []
        if row["qc_tissue_fraction"] < args.min_tissue_fraction:
            reasons.append("low_tissue")
        if args.min_blur_var > 0 and not (row["qc_blur_laplacian_var"] >= args.min_blur_var):
            reasons.append("blur")
        if row["qc_dark_pixel_fraction"] > args.max_dark_fraction:
            reasons.append("dark_artifact")
        if not args.no_stain_norm and ok == 0:
            reasons.append("stain_norm_failed")
        row["qc_pass"] = int(len(reasons) == 0)
        row["qc_reason"] = ";".join(reasons)

        # ---- save side outputs (mirrored folder structure -> no collisions)
        rel_stem = Path(rel_path).with_suffix("")
        if not args.no_side_outputs:
            p = out_dir / "per_nucleus" / rel_stem.parent
            p.mkdir(parents=True, exist_ok=True)
            nd = nuc_df.copy()
            nd.insert(0, "tile_uid", tile_uid)
            nd["lymph_like"] = is_l.astype(int) if len(nd) else []
            nd.to_csv(p / f"{rel_stem.name}_nuclei.csv", index=False, encoding="utf-8-sig")

            p = out_dir / "qc_overlay" / rel_stem.parent
            p.mkdir(parents=True, exist_ok=True)
            lym_img = None
            if len(nuc_df) and is_l.any():
                lut = np.zeros(int(labels.max()) + 1, dtype=bool)
                lut[nuc_df.loc[is_l, "label"].values.astype(int)] = True
                lym_img = np.where(lut[labels], labels, 0)
            save_qc_overlay(rgb_n, labels, lym_img, p / f"{rel_stem.name}_overlay.png")

            p = out_dir / "tissue_mask" / rel_stem.parent
            p.mkdir(parents=True, exist_ok=True)
            imsave(str(p / f"{rel_stem.name}_mask.png"),
                   tissue_mask.astype(np.uint8) * 255, check_contrast=False)

        row["normalized_path"] = ""
        if args.save_normalized:
            p = out_dir / "normalized_tiles" / rel_stem.parent
            p.mkdir(parents=True, exist_ok=True)
            npath = p / f"{rel_stem.name}.png"
            imsave(str(npath), rgb_n, check_contrast=False)
            row["normalized_path"] = str(npath.relative_to(out_dir).as_posix())

        return {k: to_py(v) for k, v in row.items()}, None
    except Exception as e:
        return None, {"rel_path": rel_path, "error": repr(e),
                      "traceback": traceback.format_exc(limit=3)}


# =====================================================================
# Task building (IDs)
# =====================================================================
def build_tasks(args, he_ref, maxc_ref):
    in_dir = Path(args.input)
    files = sorted(p for p in in_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No supported image files under: {in_dir}")

    manifest = None
    if args.manifest:
        manifest = pd.read_csv(args.manifest, dtype=str)
        need = {"rel_path", "patient_id", "slide_id"}
        if not need.issubset(manifest.columns):
            raise ValueError(f"manifest must contain columns {need}")
        manifest["rel_path"] = manifest["rel_path"].str.replace("\\", "/", regex=False)
        manifest = manifest.set_index("rel_path")

    tasks, skipped = [], []
    for p in files:
        rel = p.relative_to(in_dir).as_posix()
        mpp = args.mpp
        bdate = ""
        if manifest is not None:
            if rel not in manifest.index:
                skipped.append(rel)
                continue
            m = manifest.loc[rel]
            pid, sid = str(m["patient_id"]), str(m["slide_id"])
            if "biopsy_date" in manifest.columns and pd.notna(m.get("biopsy_date")):
                bdate = str(m["biopsy_date"])
            if "mpp" in manifest.columns and pd.notna(m.get("mpp")):
                mpp = float(m["mpp"])
        else:
            dirs = Path(rel).parts[:-1]
            pid = dirs[0] if len(dirs) >= 1 else "UNKNOWN"
            sid = dirs[1] if len(dirs) >= 2 else pid

        tasks.append({
            "abs_path": str(p), "rel_path": rel,
            "tile_uid": Path(rel).with_suffix("").as_posix(),
            "patient_id": pid, "slide_id": sid, "biopsy_date": bdate, "mpp": mpp,
            "he_ref": he_ref, "maxc_ref": maxc_ref, "args": args,
        })
    if skipped:
        print(f"[WARN] {len(skipped)} images not in manifest -> skipped")
    if manifest is None and all(t["patient_id"] == "UNKNOWN" for t in tasks):
        print("[WARN] Images are directly under --input; patient_id=UNKNOWN. "
              "Use <patient>/<slide>/tile folders or --manifest.")
    return tasks


# =====================================================================
# Aggregation
# =====================================================================
POOLED_SUM_COLS = ["qc_tissue_area_px", "nucleus_count", "lymph_like_count"]
NON_FEATURE = set(META_COLS) | {"normalized_path", "width_px", "height_px", "mpp",
                                "stain_norm_ok", "qc_pass"}


def aggregate(tile_df, keys, stats):
    feat_cols = [c for c in tile_df.columns
                 if c not in NON_FEATURE and c not in keys
                 and pd.api.types.is_numeric_dtype(tile_df[c])]
    base = (tile_df.groupby(keys, dropna=False)
            .agg(n_tiles_total=("tile_uid", "size"), n_tiles_pass=("qc_pass", "sum"))
            .reset_index())

    ok = tile_df[tile_df["qc_pass"] == 1]
    if len(ok) == 0:
        return base
    g = ok.groupby(keys, dropna=False)[feat_cols]
    parts = []
    qmap = {"p10": 0.10, "p25": 0.25, "median": 0.50, "p75": 0.75, "p90": 0.90}
    for s in stats:
        if s in ("mean", "std", "min", "max"):
            r = getattr(g, s)()
        elif s in qmap:
            r = g.quantile(qmap[s])
        else:
            raise ValueError(f"Unknown agg stat: {s}")
        r.columns = [f"{c}__{s}" for c in r.columns]
        parts.append(r)
    agg = pd.concat(parts, axis=1).reset_index()

    # pooled (sum-based) features: better than averaging tile densities
    pooled = ok.groupby(keys, dropna=False).agg(
        pooled_tissue_area_px=("qc_tissue_area_px", "sum"),
        pooled_nucleus_count=("nucleus_count", "sum"),
        pooled_lymph_like_count=("lymph_like_count", lambda x: x.sum(min_count=1)),
        mpp_mean=("mpp", "mean"),
    ).reset_index()
    a_mm2 = pooled["pooled_tissue_area_px"] * (pooled["mpp_mean"] / 1000.0) ** 2
    pooled["pooled_tissue_area_mm2"] = a_mm2
    pooled["pooled_nucleus_density_per_mm2"] = pooled["pooled_nucleus_count"] / a_mm2
    pooled["pooled_lymph_like_density_per_mm2"] = pooled["pooled_lymph_like_count"] / a_mm2
    pooled["pooled_lymph_like_fraction"] = (
        pooled["pooled_lymph_like_count"] / pooled["pooled_nucleus_count"].replace(0, np.nan)
    )
    pooled = pooled.drop(columns=["mpp_mean"])

    out = base.merge(pooled, on=keys, how="left").merge(agg, on=keys, how="left")
    return out


# =====================================================================
# Feature dictionary
# =====================================================================
GROUP_PREFIX = [
    ("qc_", "QC"), ("seg_", "QC"), ("lymph_like", "cell_heuristic"),
    ("nucleus_", "nuclear"), ("gray", "intensity"), ("hematoxylin", "intensity"),
    ("eosin", "intensity"), ("rgb_", "intensity"), ("glcm_", "texture_glcm"),
    ("lbp_", "texture_lbp"), ("gabor_", "texture_gabor"), ("wavelet_", "texture_wavelet"),
    ("fft_", "frequency"), ("nn_", "spatial"), ("delaunay_", "spatial"),
    ("radius_graph", "spatial"),
]


def feature_group(name):
    for pre, g in GROUP_PREFIX:
        if name.startswith(pre):
            return g
    return "meta" if name in NON_FEATURE else "other"


# =====================================================================
# Main
# =====================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="H&E tile feature extractor v2")
    ap.add_argument("--input", required=True, help="Root folder of tiles")
    ap.add_argument("--output", required=True, help="Output folder")
    ap.add_argument("--manifest", default=None, help="CSV: rel_path,patient_id,slide_id[,biopsy_date,mpp]")
    ap.add_argument("--mpp", type=float, default=None, help="Microns per pixel (e.g. 0.50)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="Skip tiles already in progress file")

    ap.add_argument("--stain_ref", default=None, help="Reference H&E image to fit Macenko target")
    ap.add_argument("--no_stain_norm", action="store_true")
    ap.add_argument("--save_normalized", action="store_true", help="Save normalized tiles (for image branch)")
    ap.add_argument("--no_side_outputs", action="store_true", help="Skip per-nucleus/overlay/mask files")

    ap.add_argument("--min_nucleus_area", type=int, default=25, help="px")
    ap.add_argument("--max_nucleus_area", type=int, default=2500, help="px")
    ap.add_argument("--min_distance", type=int, default=4, help="watershed peak distance (px)")
    ap.add_argument("--lumen_min_px", type=int, default=50)

    ap.add_argument("--lymph_min_um2", type=float, default=12.0)
    ap.add_argument("--lymph_max_um2", type=float, default=50.0)
    ap.add_argument("--lymph_min_circularity", type=float, default=0.80)
    ap.add_argument("--lymph_cluster_radius_um", type=float, default=20.0)

    ap.add_argument("--min_tissue_fraction", type=float, default=0.25)
    ap.add_argument("--min_blur_var", type=float, default=0.0,
                    help="Laplacian variance threshold; 0 = disabled (calibrate from qc distribution)")
    ap.add_argument("--max_dark_fraction", type=float, default=0.30)

    ap.add_argument("--fft_low_cutoff", type=float, default=0.05)
    ap.add_argument("--fft_mid_cutoff", type=float, default=0.15)
    ap.add_argument("--agg_stats", default="mean,std,p10,median,p90")
    args = ap.parse_args()

    if not (0.0 < args.fft_low_cutoff < args.fft_mid_cutoff):
        ap.error("0 < --fft_low_cutoff < --fft_mid_cutoff required")
    if args.mpp is not None and args.mpp <= 0:
        ap.error("--mpp must be > 0")
    if args.mpp is None and not args.manifest:
        print("[WARN] --mpp not given: um / mm2 features and lymph_like features will be NaN.")
    return args


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    he_ref, maxc_ref = None, None
    if args.stain_ref and not args.no_stain_norm:
        he, mc = fit_reference_stain(args.stain_ref)
        he_ref, maxc_ref = he.tolist(), mc.tolist()
        print(f"[INFO] Stain reference fitted from {args.stain_ref}")

    tasks = build_tasks(args, he_ref, maxc_ref)
    progress = out_dir / "_progress.jsonl"
    done_rows = []
    if args.resume and progress.exists():
        with open(progress, encoding="utf-8") as f:
            done_rows = [json.loads(l) for l in f if l.strip()]
        done = {r["tile_uid"] for r in done_rows}
        tasks = [t for t in tasks if t["tile_uid"] not in done]
        print(f"[INFO] Resume: {len(done)} done, {len(tasks)} remaining")
    elif progress.exists():
        progress.unlink()

    print(f"[INFO] Processing {len(tasks)} tiles with {args.workers} worker(s)")
    rows, failures = list(done_rows), []
    with open(progress, "a", encoding="utf-8") as pf:
        def handle(i, res):
            row, err = res
            if err:
                failures.append(err)
                print(f"  [ERROR] {err['rel_path']}: {err['error']}")
            else:
                rows.append(row)
                pf.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
                pf.flush()
            if i % 50 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] {time.time() - t0:.0f}s")

        if args.workers <= 1:
            for i, t in enumerate(tasks, 1):
                handle(i, process_tile(t))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(process_tile, t) for t in tasks]
                for i, fu in enumerate(as_completed(futs), 1):
                    handle(i, fu.result())

    if not rows:
        print("[ERROR] No tiles processed.")
        if failures:
            pd.DataFrame(failures).to_csv(out_dir / "failed_images.csv", index=False,
                                          encoding="utf-8-sig")
        sys.exit(1)

    tile_df = pd.DataFrame(rows)
    front = [c for c in META_COLS if c in tile_df.columns]
    tile_df = tile_df[front + [c for c in tile_df.columns if c not in front]]
    tile_df = tile_df.sort_values(["patient_id", "slide_id", "tile_uid"]).reset_index(drop=True)
    tile_df.to_csv(out_dir / "pathology_features_tile.csv", index=False, encoding="utf-8-sig")

    stats = [s.strip() for s in args.agg_stats.split(",") if s.strip()]
    slide_keys = ["patient_id", "slide_id"]
    slide_df = aggregate(tile_df, slide_keys, stats)
    if tile_df["biopsy_date"].astype(str).str.len().gt(0).any():
        dates = tile_df.groupby(slide_keys)["biopsy_date"].first().reset_index()
        slide_df = dates.merge(slide_df, on=slide_keys, how="right")
    slide_df.to_csv(out_dir / "pathology_features_slide.csv", index=False, encoding="utf-8-sig")

    patient_df = aggregate(tile_df, ["patient_id"], stats)
    patient_df.to_csv(out_dir / "pathology_features_patient.csv", index=False, encoding="utf-8-sig")

    man_cols = ["tile_uid", "rel_path", "normalized_path", "patient_id", "slide_id",
                "biopsy_date", "qc_pass", "qc_reason", "qc_tissue_fraction",
                "qc_blur_laplacian_var"]
    tile_df[man_cols].to_csv(out_dir / "tile_manifest.csv", index=False, encoding="utf-8-sig")

    fd = pd.DataFrame({"feature": tile_df.columns})
    fd["group"] = fd["feature"].map(feature_group)
    fd["missing_rate"] = [float(tile_df[c].isna().mean()) for c in tile_df.columns]
    fd.to_csv(out_dir / "feature_dictionary.csv", index=False, encoding="utf-8-sig")

    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "failed_images.csv", index=False,
                                      encoding="utf-8-sig")

    cfg = {
        "script": "pathology_feature_extractor_v2.py",
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 1),
        "args": {k: v for k, v in vars(args).items()},
        "stain_reference": {"he_ref": he_ref or HE_REF.tolist(),
                            "maxc_ref": maxc_ref or MAXC_REF.tolist()},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "scikit-image": skimage.__version__, "opencv": cv2.__version__,
                     "pywavelets": pywt.__version__},
        "n_tiles": int(len(tile_df)), "n_tiles_pass": int(tile_df["qc_pass"].sum()),
        "n_failed": len(failures), "n_slides": int(len(slide_df)),
        "n_patients": int(len(patient_df)),
        "n_tile_features": int((fd["group"] != "meta").sum()),
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2, default=str)

    qc = tile_df["qc_blur_laplacian_var"].describe(percentiles=[0.01, 0.05, 0.5]).round(1)
    print("\n[DONE]")
    print(f"Tiles   : {cfg['n_tiles']} (QC pass {cfg['n_tiles_pass']}, failed {cfg['n_failed']})")
    print(f"Slides  : {cfg['n_slides']}   Patients: {cfg['n_patients']}")
    print(f"Blur (Laplacian var) 1%/5%/50%: {qc.get('1%')}/{qc.get('5%')}/{qc.get('50%')}"
          "  -> use to set --min_blur_var")
    print(f"Outputs in {out_dir}")


if __name__ == "__main__":
    main()
