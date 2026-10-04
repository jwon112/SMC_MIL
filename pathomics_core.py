"""Image-only research descriptors. No labels, fitting, or outcome access.

The user's v2 equations are reused without editing that file. New descriptors
are explicitly proxies unless an externally reviewed segmentation is supplied.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import color, measure, segmentation
import pathology_feature_extractor_v2 as v2


@dataclass(frozen=True)
class Settings:
    target_mpp: float = 0.25
    halo_um: float = 8.0
    min_tissue_fraction: float = 0.25
    max_dark_fraction: float = 0.30
    min_blur_var: float = 0.0
    tissue_mask_method: str = "he_od_saturation_v1"
    tissue_min_saturation: float = 0.04
    tissue_min_mean_od: float = 0.04
    nucleus_min_um2: float = 1.5625  # v2: 25 px at 0.25 MPP
    nucleus_max_um2: float = 156.25  # v2: 2500 px at 0.25 MPP
    peak_distance_um: float = 1.0
    lymph_min_um2: float = 12.0
    lymph_max_um2: float = 50.0
    lymph_min_circularity: float = 0.8
    lymph_cluster_radius_um: float = 20.0
    stain_normalization: str = "none"  # no cohort-fitted reference


CORE_FEATURES = [
    "nucleus_density_per_mm2", "nucleus_area_fraction",
    "nucleus_area_um2_median", "nucleus_area_um2_iqr",
    "nucleus_circularity_median", "nucleus_solidity_median",
    "nucleus_eccentricity_median", "nucleus_orientation_coherence",
    "lymph_like_density_per_mm2", "lymph_like_fraction",
    "lymph_like_nn_distance_um_mean", "lymph_like_cluster_fraction",
    "nn_distance_um_median", "glcm_contrast_mean", "glcm_homogeneity_mean",
    "glcm_entropy_mean", "gray_entropy", "fft_high_frequency_fraction",
    "compartment_stroma_fraction", "compartment_stroma_myocardium_ratio",
    "annotated_lymphocyte_density_per_mm2",
    "annotated_lymphocyte_in_myocardium_density_per_mm2",
]
OPTIONAL_FEATURES = [
    "compartment_myocardium_fraction", "compartment_stroma_fraction",
    "compartment_stroma_myocardium_ratio", "compartment_stroma_solidity_median",
    "compartment_stroma_eccentricity_median", "compartment_stroma_axial_coherence",
    "compartment_stroma_skeleton_area_over_length_um",
    "annotated_lymphocyte_count", "annotated_lymphocyte_density_per_mm2",
    "annotated_lymphocyte_area_fraction", "annotated_lymphocyte_in_myocardium_count",
    "annotated_lymphocyte_in_myocardium_density_per_mm2",
    "annotated_lymphocyte_myocardium_boundary_distance_um_mean",
    "annotated_lymphocyte_within_20um_myocardium_boundary_fraction",
]


def validate_settings(s):
    if s.target_mpp <= 0 or not math.isfinite(s.target_mpp) or s.halo_um < 0:
        raise ValueError("Invalid target MPP/halo")
    if not 0 <= s.min_tissue_fraction <= 1 or not 0 <= s.max_dark_fraction <= 1:
        raise ValueError("QC fractions must be in [0,1]")
    if s.stain_normalization not in {"none", "macenko_fixed"}:
        raise ValueError("Unknown stain normalization")
    if s.tissue_mask_method not in {"he_od_saturation_v1", "legacy_hsv_v"}:
        raise ValueError("Unknown tissue mask method")
    if not 0 <= s.tissue_min_saturation <= 1 or not math.isfinite(s.tissue_min_mean_od) or s.tissue_min_mean_od < 0:
        raise ValueError("Invalid tissue mask thresholds")
    if not math.isfinite(s.min_blur_var) or s.min_blur_var < 0:
        raise ValueError("Invalid blur threshold")


def make_he_tissue_mask(rgb, settings, min_object=500, generic=False):
    """Exploratory H&E mask; fixed thresholds, no cohort/outcome fitting.

    Saturation plus mean optical density does not require EVERY RGB channel
    to be below 250. Thus a bright red channel alone cannot remove pink tissue.
    Very dark pixels are retained for the downstream dark-artifact QC gate,
    not interpreted as confirmed tissue. Masks still require visual review.
    Component/hole cleanup matches the original v2 at analysis resolution.
    """
    if settings.tissue_mask_method == "legacy_hsv_v" and not generic:
        return v2.make_tissue_mask(rgb, min_object=min_object)
    rgb = v2.ensure_rgb_uint8(rgb)
    hsv = color.rgb2hsv(rgb.astype(np.float32) / 255.0)
    mean_od = -np.log((rgb.astype(np.float64) + 1.0) / 256.0).mean(axis=2)
    raw = ((hsv[..., 1] > settings.tissue_min_saturation) &
           (mean_od > settings.tissue_min_mean_od)) | (hsv[..., 2] < 0.20)
    if generic:
        # Neutral/black special stains must not require pink/purple saturation.
        # Foreground candidate only, not validated tissue segmentation.
        raw = mean_od > settings.tissue_min_mean_od
    raw = v2.remove_small(raw, min_object)
    filled_all = ndi.binary_fill_holes(raw)
    lumen = v2.remove_small(filled_all & ~raw, 50)
    tissue = v2.fill_small_holes(raw, 500)
    tissue = ndi.binary_closing(tissue, structure=v2.disk(3))
    tissue &= filled_all | raw
    return tissue, raw, lumen


def qc_preview(core, tissue, overlay, passed, reason):
    """Three labeled panels; green=tissue candidate, cyan/yellow=nuclear proxies."""
    masked = (core.astype(float) * .25).astype(np.uint8)
    masked[tissue] = np.rint(.65 * core[tissue] + .35 * np.array([0, 210, 70])).astype(np.uint8)
    panels = np.concatenate([core, masked, overlay], axis=1)
    height, width = core.shape[:2]
    canvas = Image.new("RGB", (width*3, height+28), "white")
    canvas.paste(Image.fromarray(panels), (0,28))
    draw = ImageDraw.Draw(canvas)
    titles = ["Original", f"Tissue mask: {tissue.mean():.1%}",
              "Nuclei: " + ("basic QC PASS" if passed else "FAIL " + reason)]
    for i, title in enumerate(titles): draw.text((i*width+4, 7), title, fill="black")
    return np.asarray(canvas)


def select_nonoverlapping(coords, size):
    """Deterministic original-index greedy selection; reject positive-area overlap.

    Adjacent cores can touch. Never call a sum of overlapping patch areas a WSI
    area. Discarded original indices remain in audit, not silently reordered.
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("coords must be finite [N,2]")
    if not np.equal(coords, np.rint(coords)).all() or size <= 0:
        raise ValueError("Coordinates must be integer source pixels; size positive")
    buckets, keep, dropped = {}, [], []
    for i, (x, y) in enumerate(coords):
        bx, by = int(x // size), int(y // size)
        neighbors = [j for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                     for j in buckets.get((bx + dx, by + dy), [])]
        if any(abs(x - coords[j, 0]) < size and abs(y - coords[j, 1]) < size for j in neighbors):
            dropped.append(i)
        else:
            keep.append(i)
            buckets.setdefault((bx, by), []).append(i)
    return np.asarray(keep, int), np.asarray(dropped, int)


def optional_compartments(masks, tissue, mpp):
    """Externally supplied masks ONLY; never derive myocardium from RGB color.

    All masks represent the core at standardized resolution. Annotated nuclei
    touching the core edge are excluded; these local features have edge bias.
    """
    out = dict.fromkeys(OPTIONAL_FEATURES, np.nan)
    if not masks:
        return out
    area = float(tissue.sum())
    if area == 0:
        return out
    myo = masks.get("myocardium")
    stroma = masks.get("stroma")
    if myo is not None and stroma is not None and np.any(myo.astype(bool) & stroma.astype(bool)):
        raise ValueError("myocardium/stroma masks overlap")
    for name, mask in [("myocardium", myo), ("stroma", stroma)]:
        if mask is not None:
            out[f"compartment_{name}_fraction"] = float((mask.astype(bool) & tissue).sum() / area)
    if stroma is not None:
        st = stroma.astype(bool) & tissue
        props = measure.regionprops(measure.label(st))
        if props:
            out["compartment_stroma_solidity_median"] = float(np.median([p.solidity for p in props]))
            out["compartment_stroma_eccentricity_median"] = float(np.median([p.eccentricity for p in props]))
            out["compartment_stroma_axial_coherence"] = float(abs(np.mean(np.exp(2j * np.array([p.orientation for p in props])))))
        # This is skeleton pixel length, an orientation-sensitive approximation,
        # not collagen thickness or a literal replication of a paper's fiber model.
        skel = v2.morphology.skeletonize(st)
        if skel.sum():
            out["compartment_stroma_skeleton_area_over_length_um"] = float(st.sum() / skel.sum() * mpp)
        if myo is not None and (myo.astype(bool) & tissue).sum():
            out["compartment_stroma_myocardium_ratio"] = float(st.sum() / (myo.astype(bool) & tissue).sum())
    lymph = masks.get("lymphocyte_instances")
    if lymph is not None:
        lymph = lymph.copy()
        border_ids = np.unique(np.r_[lymph[0], lymph[-1], lymph[:, 0], lymph[:, -1]])
        lymph[np.isin(lymph, border_ids[border_ids > 0])] = 0
        props = measure.regionprops(lymph)
        pts = np.array([p.centroid for p in props], float).reshape(-1, 2)
        keep = np.array([tissue[min(int(y), tissue.shape[0]-1), min(int(x), tissue.shape[1]-1)] for y, x in pts], bool)
        pts = pts[keep]
        out["annotated_lymphocyte_count"] = int(len(pts))
        out["annotated_lymphocyte_density_per_mm2"] = len(pts) / (area * (mpp / 1000)**2)
        valid_ids = [p.label for p, ok in zip(props, keep) if ok]
        out["annotated_lymphocyte_area_fraction"] = float((np.isin(lymph, valid_ids) & tissue).sum() / area)
        if myo is not None:
            mm = myo.astype(bool) & tissue
            n_inside = sum(bool(mm[int(y), int(x)]) for y, x in pts)
            out["annotated_lymphocyte_in_myocardium_count"] = int(n_inside)
            if mm.sum():
                out["annotated_lymphocyte_in_myocardium_density_per_mm2"] = n_inside / (mm.sum() * (mpp / 1000)**2)
            boundary = segmentation.find_boundaries(mm, mode="inner")
            if boundary.any() and len(pts):
                dist = ndi.distance_transform_edt(~boundary) * mpp
                distances = np.array([dist[int(y), int(x)] for y, x in pts])
                out["annotated_lymphocyte_myocardium_boundary_distance_um_mean"] = float(distances.mean())
                out["annotated_lymphocyte_within_20um_myocardium_boundary_fraction"] = float(np.mean(distances <= 20))
    return out


def resize_mask(mask, shape):
    if not np.isfinite(mask).all() or (mask < 0).any() or not np.equal(mask, np.rint(mask)).all():
        raise ValueError("Masks must contain nonnegative finite integer labels")
    return np.asarray(Image.fromarray(mask.astype(np.int32)).resize((shape[1], shape[0]), Image.Resampling.NEAREST))


def describe_patch(rgb_with_halo, core_box, mpp_x, mpp_y, settings, masks=None, diagnostics=None,
                   he_applicable=True):
    """Read RGB halo, segment on context, own nuclei by centroid in core.

    core_box=(left,top,width,height) in raw read pixels. Resample the entire
    window then crop, preserving a matched halo. No encoder resizing to 224.
    Returns numeric features and a compact QC overlay for optional saving.
    """
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
    out.update(v2.extract_stain_intensity_features(core_n, hed[cy, cx], tissue))
    out.update(v2.extract_glcm_features(gray_u8, tissue))
    out.update(v2.extract_lbp_features(gray_u8, tissue))
    out.update(v2.extract_gabor_features(gray, tissue))
    out.update(v2.extract_wavelet_features(gray, tissue))
    out.update(v2.extract_fft_features(gray, tissue, mpp=m))
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
    return out, preview


def rook_moran(frame, feature="lymph_like_density_per_mm2"):
    """Binary, symmetric, side-touching equal-size core graph; no gap bridging.

    A descriptive statistic only: no permutation p-value or disease claim.
    Full-grid source-coordinate adjacency, no global kNN across tissue pieces.
    """
    f = frame.loc[(frame.qc_pass == 1) & np.isfinite(frame[feature])]
    result = {"patch_moran_I_lymph_like_density": np.nan, "patch_moran_nodes": len(f), "patch_moran_edges": 0}
    if len(f) < 3 or f.patch_size.nunique() != 1:
        return result
    size = int(f.patch_size.iloc[0])
    positions = {(int(r.x_source), int(r.y_source)): i for i, r in enumerate(f.itertuples())}
    edges = [(i, positions[(x+dx, y+dy)]) for (x, y), i in positions.items()
             for dx, dy in [(size, 0), (0, size)] if (x+dx, y+dy) in positions]
    result["patch_moran_edges"] = len(edges)
    z = f[feature].to_numpy(dtype=float, copy=True)
    z -= z.mean()
    if not edges or np.dot(z, z) <= 0:
        return result
    # W=2*edges and numerator also double-counts undirected edges.
    result["patch_moran_I_lymph_like_density"] = float(len(z)/len(edges)*sum(z[i]*z[j] for i,j in edges)/np.dot(z,z))
    return result


def aggregate(frame, features):
    """Pooled patch distribution, not mean of slide medians. No patient pooling."""
    out = {"patches_read": len(frame), "patches_qc_pass": int(frame.qc_pass.sum())}
    valid = frame.loc[frame.qc_pass == 1]
    out["analyzed_core_tissue_mm2"] = float(valid.tissue_area_mm2.sum())
    for key in features:
        vals = pd.to_numeric(valid[key], errors="coerce").to_numpy(float)
        vals = vals[np.isfinite(vals)]
        out[f"{key}__valid_n"] = len(vals)
        for stat in ["mean", "median", "std", "min", "max", "p10", "p90"]:
            value = np.nan
            if len(vals):
                value = {"mean": np.mean, "median": np.median, "min": np.min, "max": np.max}.get(stat, lambda _: np.nan)(vals)
                if stat == "std": value = float(np.std(vals, ddof=1)) if len(vals)>1 else np.nan
                if stat in {"p10", "p90"}: value = float(np.percentile(vals, int(stat[1:])))
            out[f"{key}__{stat}"] = float(value)
    for stem, count_col in [("nucleus", "nucleus_count"), ("lymph_like", "lymph_like_count")]:
        good = valid[np.isfinite(valid[count_col]) & (valid.tissue_area_mm2 > 0)]
        denom = float(good.tissue_area_mm2.sum())
        out[f"{stem}_pooled_density_per_mm2"] = float(good[count_col].sum()/denom) if denom else np.nan
    occupancy = valid[np.isfinite(valid.nucleus_occupied_area_mm2) & (valid.tissue_area_mm2 > 0)]
    denom = float(occupancy.tissue_area_mm2.sum())
    out["nucleus_pooled_area_fraction"] = float(occupancy.nucleus_occupied_area_mm2.sum()/denom) if denom else np.nan
    return out
