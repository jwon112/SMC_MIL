"""Opt-in execution-only optimizations. Original extraction files stay unchanged.

The full and smoke schemas, QC metrics and saved preview pixels are retained.
QC-failed texture values are already NA in the legacy implementation.
No patch sampling, resolution change, formula removal or learned model changes.
"""
from functools import lru_cache
from dataclasses import asdict
from types import SimpleNamespace
import math
import numpy as np
from PIL import Image, ImageDraw
from skimage import color, segmentation
from scipy import ndimage as ndi
import pathology_feature_extractor_v2 as v2
import pathomics_core as legacy
from pathomics_core import (Settings, validate_settings, make_he_tissue_mask,
                            optional_compartments, resize_mask, qc_preview)


class LazyPreview:
    """Materialize the identical preview only if a caller actually uses it."""
    def __init__(self, render):
        self.render = render
        self.array = None

    def materialize(self):
        if self.array is None:
            self.array = self.render()
            self.render = None
        return self.array

    @property
    def __array_interface__(self):
        array = self.materialize()
        interface = dict(array.__array_interface__)
        # Pillow uses tobytes for strided array-interface objects. Expose real
        # strides so this object need not implement Python's buffer protocol.
        interface["strides"] = array.strides
        return interface

    def tobytes(self):
        return self.materialize().tobytes()

    def __array__(self, dtype=None, copy=None):
        array = np.asarray(self.materialize(), dtype=dtype)
        return array.copy() if copy else array


@lru_cache(maxsize=64)
def gabor_filters(freq, theta):
    kernel = v2.gabor_kernel(freq, theta=theta)
    re = np.real(kernel).astype(np.float32)
    im = np.imag(kernel).astype(np.float32)
    re.flags.writeable = False
    im.flags.writeable = False
    return re, im


def gabor_cached(gray_f, tissue_mask):
    out = {}
    inner = ndi.binary_erosion(tissue_mask, structure=v2.disk(3))
    if inner.sum() < 100:
        inner = tissue_mask
    work = v2.fill_background(gray_f, tissue_mask).astype(np.float32)
    for freq in v2.GABOR_FREQS:
        for i, theta in enumerate(v2.GABOR_THETAS):
            prefix = f"gabor_f{freq:.2f}_theta{i}"
            if inner.sum() == 0:
                for stat in ["mean", "std", "energy"]:
                    out[f"{prefix}_{stat}"] = np.nan
                continue
            kr, ki = gabor_filters(freq, theta)
            re = v2.cv2.filter2D(work, v2.cv2.CV_32F, kr, borderType=v2.cv2.BORDER_REFLECT)
            im = v2.cv2.filter2D(work, v2.cv2.CV_32F, ki, borderType=v2.cv2.BORDER_REFLECT)
            mag = np.sqrt(re ** 2 + im ** 2)[inner]
            out[f"{prefix}_mean"] = float(np.mean(mag))
            out[f"{prefix}_std"] = float(np.std(mag))
            out[f"{prefix}_energy"] = float(np.mean(mag ** 2))
    return out


@lru_cache(maxsize=6)
def texture_schema(name):
    # Discover original fixed keys ONCE per process, from a tiny empty image.
    # No study image/outcome is consulted, and values are never reused.
    gray = np.zeros((16, 16), np.uint8)
    mask = np.zeros(gray.shape, bool)
    function = getattr(v2, name)
    if name == "extract_stain_intensity_features":
        out = function(np.zeros((16, 16, 3), np.uint8),
                       np.zeros((16, 16, 3), np.float32), mask)
    elif name in {"extract_gabor_features", "extract_wavelet_features", "extract_fft_features"}:
        out = function(gray.astype(np.float32), mask)
    else:
        out = function(gray, mask)
    return tuple(out)


def feature_call(name, skip, *args, **kwargs):
    if skip:
        return dict.fromkeys(texture_schema(name), np.nan)
    function = gabor_cached if name == "extract_gabor_features" else getattr(v2, name)
    return function(*args, **kwargs)


def describe_patch_fast(rgb_with_halo, core_box, mpp_x, mpp_y, settings, masks=None, diagnostics=None,
                   he_applicable=True):
    """Read RGB halo, segment on context, own nuclei by centroid in core.

    core_box=(left,top,width,height) in raw read pixels. Resample the entire
    window then crop, preserving a matched halo. No encoder resizing to 224.
    Returns numeric features and a compact QC overlay for optional saving.
    """
    if diagnostics is not None:
        return legacy.describe_patch(rgb_with_halo, core_box, mpp_x, mpp_y, settings,
                                     masks, diagnostics, he_applicable=he_applicable)
    validate_settings(settings)
    if any(not math.isfinite(v) or v <= 0 for v in [mpp_x, mpp_y]):
        raise ValueError("Physical features require verified mpp_x/y (never inferred from '40x')")
    rgb = v2.ensure_rgb_uint8(rgb_with_halo)
    left, top, width, height = core_box
    m = settings.target_mpp
    sx, sy = mpp_x / m, mpp_y / m
    w, h = int(round(rgb.shape[1]*sx)), int(round(rgb.shape[0]*sy))
    if min(w, h) < 16 or max(w, h) > 8192:
        raise ValueError("Unsupported standardized patch size")
    rgb = np.asarray(Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR))
    x0, y0 = int(round(left*sx)), int(round(top*sy))
    x1, y1 = int(round((left+width)*sx)), int(round((top+height)*sy))
    cy, cx = slice(y0, y1), slice(x0, x1)
    core = rgb[cy, cx]
    scale_error = max(abs((x1-x0)*m/(width*mpp_x)-1), abs((y1-y0)*m/(height*mpp_y)-1))
    if scale_error > .01:
        raise ValueError("Resampling physical size rounding exceeds 1%; use larger core")
    tissue_full, _, lumen_full = make_he_tissue_mask(rgb, settings, generic=not he_applicable)
    tissue, lumen = tissue_full[cy, cx], lumen_full[cy, cx]
    out = v2.extract_qc_features(core, tissue, lumen)
    out["qc_enclosed_white_fraction"] = out.pop("qc_lumen_fraction")
    out.update(analysis_mpp=m, analysis_width_px=x1-x0, analysis_height_px=y1-y0,
               resampling_size_relative_error=scale_error,
               tissue_area_mm2=float(tissue.sum()*(m/1000)**2))
    normalized = rgb
    ok = 1
    if settings.stain_normalization == "macenko_fixed" and he_applicable:
        normalized, ok = v2.macenko_normalize(rgb, tissue_full)
    reasons = []
    if out["qc_tissue_fraction"] < settings.min_tissue_fraction or tissue.sum() < 100:
        reasons.append("low_tissue")
    out['qc_tissue_area_criterion_pass']=int('low_tissue' not in reasons)
    if he_applicable and out["qc_dark_pixel_fraction"] > settings.max_dark_fraction:
        reasons.append("dark_artifact")
    if settings.min_blur_var > 0 and not out["qc_blur_laplacian_var"] >= settings.min_blur_var:
        reasons.append("blur")
    if not ok:
        reasons.append("stain_normalization_failed")
    out.update(qc_pass=int(not reasons), qc_reason=";".join(reasons), stain_norm_ok=ok)
    hed = color.rgb2hed(normalized.astype(np.float32)/255) if he_applicable else np.zeros(rgb.shape, np.float32)
    labels, threshold = v2.segment_nuclei(
        hed[..., 0], tissue_full,
        min_area=max(1, int(round(settings.nucleus_min_um2/m**2))),
        max_area=max(1, int(round(settings.nucleus_max_um2/m**2))),
        min_distance=max(1, int(round(settings.peak_distance_um/m)))) if he_applicable else (np.zeros(rgb.shape[:2], np.int32), np.nan)
    out["seg_h_threshold"] = threshold
    # Separate image QC from validity of morphology. PASS is not validation.
    gray_core = v2.cv2.cvtColor(core, v2.cv2.COLOR_RGB2GRAY)
    out["qc_gray_std_in_tissue"] = float(gray_core[tissue].std()) if tissue.any() else np.nan
    h_values = hed[cy,cx,0][tissue]
    out["qc_h_p90_minus_p10"] = float(np.diff(np.percentile(h_values,[10,90]))[0]) if len(h_values) else np.nan
    out["qc_measurement_status"] = "unvalidated" if not reasons else "excluded_basic_qc"
    out["qc_review_reasons"] = ""
    out["qc_candidate_nucleus_count"] = np.nan
    out["qc_candidate_support_added_fraction"] = np.nan
    out["qc_he_applicable"] = int(he_applicable)
    out["qc_he_unavailable_reason"] = "" if he_applicable else "not_confirmed_HE; HE_decomposition_and_cell_proxies_not_applied"
    out["qc_foreground_method"] = settings.tissue_mask_method if he_applicable else "generic_mean_OD_candidate_v1"
    all_nuclei = v2.per_nucleus_table(labels, hed[..., 0])
    border = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    owned = ((all_nuclei.centroid_x >= x0) & (all_nuclei.centroid_x < x1)
             & (all_nuclei.centroid_y >= y0) & (all_nuclei.centroid_y < y1))
    out["nuclei_context_edge_excluded"] = int((owned & all_nuclei.label.isin(border)).sum())
    nuc = all_nuclei.loc[owned & ~all_nuclei.label.isin(border)].copy()
    nuc["centroid_x"] -= x0
    nuc["centroid_y"] -= y0
    out["qc_detected_nuclei_before_exclusion"] = len(nuc)
    if not reasons and len(nuc)==0:
        out["qc_review_reasons"] = "no_detected_nuclei_not_proof_of_absence"
    out.update(v2.extract_nuclear_features(nuc, int(tissue.sum()), m))
    # Area occupancy clips all segmented nuclei to core, whereas counts are
    # centroid-owned. Do not sum the areas of halo-extended nuclei as occupancy.
    nuclear_area = int(((labels[cy, cx] > 0) & tissue).sum())
    out["nucleus_area_fraction"] = nuclear_area / tissue.sum() if tissue.sum() else np.nan
    out["nucleus_occupied_area_mm2"] = nuclear_area*(m/1000)**2
    lymph, is_lymph = v2.extract_lymph_like_features(nuc, int(tissue.sum()), m, SimpleNamespace(**asdict(settings)))
    out.update(lymph)
    core_n = normalized[cy, cx]
    gray_u8 = v2.cv2.cvtColor(core_n, v2.cv2.COLOR_RGB2GRAY)
    gray = gray_u8.astype(np.float32)/255
    out.update(feature_call("extract_stain_intensity_features", bool(reasons), core_n, hed[cy, cx], tissue))
    out.update(feature_call("extract_glcm_features", bool(reasons), gray_u8, tissue))
    out.update(feature_call("extract_lbp_features", bool(reasons), gray_u8, tissue))
    out.update(feature_call("extract_gabor_features", bool(reasons), gray, tissue))
    out.update(feature_call("extract_wavelet_features", bool(reasons), gray, tissue))
    out.update(feature_call("extract_fft_features", bool(reasons), gray, tissue, mpp=m))
    out.update(v2.extract_spatial_features(nuc, m))
    converted = {}
    for name, mask in (masks or {}).items():
        if mask.shape != (height, width):
            raise ValueError(f"{name}: mask must match unresized source core {(height,width)}")
        converted[name] = resize_mask(mask, tissue.shape)
    out.update(optional_compartments(converted, tissue, m))
    if not he_applicable:
        he_keys = set(v2.extract_nuclear_features(nuc, int(tissue.sum()), m))
        he_keys.update(lymph)
        he_keys.update(v2.extract_spatial_features(nuc, m))
        he_keys.update(k for k in out if k.startswith(("hematoxylin", "eosin")))
        he_keys.update(["nucleus_area_fraction", "nucleus_occupied_area_mm2", "nuclei_context_edge_excluded"])
        for key in he_keys: out[key] = np.nan
        out["qc_detected_nuclei_before_exclusion"] = np.nan
        out["qc_h_p90_minus_p10"] = np.nan
        out["qc_review_reasons"] = "generic_foreground_unvalidated"
        if out["qc_dark_pixel_fraction"] > settings.max_dark_fraction:
            out["qc_review_reasons"] += ";dark_pixels_may_be_stain_or_artifact"
    if reasons:
        # Invalid measurements are missing, NOT a normal/negative observation.
        for key in list(out):
            if key not in {"qc_pass", "qc_reason", "stain_norm_ok", "seg_h_threshold",
                            "analysis_mpp", "analysis_width_px", "analysis_height_px",
                            "resampling_size_relative_error", "tissue_area_mm2"} and not key.startswith("qc_"):
                out[key] = np.nan
    def render_preview():
        overlay = core.copy()
        overlay[segmentation.find_boundaries(labels[cy, cx])] = (0, 210, 255)
        ids = nuc.loc[is_lymph, "label"].to_numpy(int) if len(nuc) else []
        overlay[segmentation.find_boundaries(np.where(np.isin(labels[cy, cx], ids), labels[cy, cx], 0))] = (255, 200, 0)
        if diagnostics is not None and he_applicable:
            # Diagnostic ONLY: relax component-size cleanup in the segmentation
            # support, not in the area denominator or any current feature values.
            min_area=max(1,int(round(settings.nucleus_min_um2/m**2)))
            support,_,_=make_he_tissue_mask(rgb,settings,min_object=min_area)
            candidate,_=v2.segment_nuclei(hed[...,0],support,min_area=min_area,
                max_area=max(1,int(round(settings.nucleus_max_um2/m**2))),
                min_distance=max(1,int(round(settings.peak_distance_um/m))))
            cand_n=v2.per_nucleus_table(candidate,hed[...,0])
            cand_border=np.unique(np.r_[candidate[0],candidate[-1],candidate[:,0],candidate[:,-1]])
            cand_owned=((cand_n.centroid_x>=x0)&(cand_n.centroid_x<x1)&
                        (cand_n.centroid_y>=y0)&(cand_n.centroid_y<y1)&~cand_n.label.isin(cand_border))
            out["qc_candidate_nucleus_count"]=int(cand_owned.sum())
            out["qc_candidate_support_added_fraction"]=float((support[cy,cx]&~tissue).mean())
            current=core.copy();current[segmentation.find_boundaries(labels[cy,cx])]=(0,210,255)
            proposed=core.copy();proposed[segmentation.find_boundaries(candidate[cy,cx])]=(255,140,0)
            base=qc_preview(core,tissue,current,bool(out['qc_pass']),out['qc_reason'])
            panel=Image.new('RGB',(core.shape[1],base.shape[0]),'white')
            panel.paste(Image.fromarray(proposed),(0,28))
            ImageDraw.Draw(panel).text((4,7),'Candidate support: NOT validated',fill='black')
            diagnostics.update(comparison=np.concatenate([base,np.asarray(panel)],axis=1),
                candidate_count=int(cand_owned.sum()),current_count=len(nuc),
                candidate_support_min_pixels=min_area, current_support_min_pixels=500)
        preview = qc_preview(core, tissue, overlay, bool(out["qc_pass"]), out["qc_reason"])
        if not he_applicable:
            canvas = Image.fromarray(preview)
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((core.shape[1]*2, 0, core.shape[1]*3, 27), fill="white")
            draw.text((core.shape[1]*2+4, 7), "HE nuclei: not applicable", fill="black")
            preview = np.asarray(canvas)
            if diagnostics is not None:
                diagnostics.update(comparison=preview, candidate_count=None, current_count=None,
                                   candidate_support_min_pixels=None, current_support_min_pixels=None)
        return preview
    return out, LazyPreview(render_preview)
