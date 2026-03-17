import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def _maybe_smooth_diff_map(g: np.ndarray, blur_ksize: int) -> np.ndarray:
    """
    g: float32 [H,W]
    blur_ksize: 0이면 smoothing 안 함. >0이면 uniform filter 적용(가능하면).
    """
    if blur_ksize is None or blur_ksize <= 1:
        return g
    try:
        from scipy.ndimage import uniform_filter
    except Exception:
        # scipy가 없으면 그냥 반환
        return g
    return uniform_filter(g, size=int(blur_ksize))


def _morphology_cleanup(mask01: np.ndarray, open_iter: int, close_iter: int) -> np.ndarray:
    """
    mask01: bool/0-1 [H,W]
    scipy.ndimage가 있으면 opening/closing 수행, 없으면 그대로 반환.
    """
    if (open_iter is None or open_iter <= 0) and (close_iter is None or close_iter <= 0):
        return mask01
    try:
        from scipy.ndimage import binary_opening, binary_closing
    except Exception:
        return mask01

    out = mask01.astype(bool)
    if open_iter and open_iter > 0:
        out = binary_opening(out, iterations=int(open_iter))
    if close_iter and close_iter > 0:
        out = binary_closing(out, iterations=int(close_iter))
    return out


def _remove_small_components(mask01: np.ndarray, min_area: int) -> np.ndarray:
    """
    mask01: bool [H,W]
    min_area 이하 connected component 제거.
    scipy.ndimage가 없으면 동작 생략.
    """
    if min_area is None or min_area <= 0:
        return mask01
    try:
        from scipy.ndimage import label
    except Exception:
        return mask01

    lab, n = label(mask01)
    if n <= 0:
        return mask01
    counts = np.bincount(lab.ravel())
    keep = np.ones(n + 1, dtype=bool)
    keep[0] = False
    keep[counts < int(min_area)] = False
    return keep[lab]


def compute_mask_from_pair(
    ori_img: np.ndarray,
    art_img: np.ndarray,
    threshold: float,
    use_max_channel: bool = False,
    blur_ksize: int = 0,
    open_iter: int = 0,
    close_iter: int = 0,
    min_area: int = 0,
) -> np.ndarray:
    """
    ori_img, art_img: uint8 [H, W, 3]
    return: uint8 [H, W] in {0, 255}
    """
    diff = np.abs(ori_img.astype(np.float32) - art_img.astype(np.float32))
    # diff aggregation: mean(기본) 또는 max(artifact 경계/희미한 경우 더 잘 잡는 경우가 있음)
    g = diff.max(axis=2) if use_max_channel else diff.mean(axis=2)
    g = _maybe_smooth_diff_map(g, blur_ksize=blur_ksize)

    mask01 = (g > float(threshold))
    mask01 = _morphology_cleanup(mask01, open_iter=open_iter, close_iter=close_iter)
    mask01 = _remove_small_components(mask01, min_area=min_area)

    mask = mask01.astype(np.uint8) * 255
    return mask


def process_folder(
    root: str,
    threshold: float,
    overwrite: bool = False,
    use_max_channel: bool = False,
    blur_ksize: int = 0,
    open_iter: int = 0,
    close_iter: int = 0,
    min_area: int = 0,
):
    """
    root: path to one artifact type folder, e.g.
      /home/jupyter/data/image__team/ArtiDiffuser-Synth/marking
      containing ori/ and inpainted/ subfolders.
    """
    ori_dir = os.path.join(root, "ori")
    imp_dir = os.path.join(root, "inpainted")
    mask_dir = os.path.join(root, "masks")

    if not os.path.isdir(ori_dir) or not os.path.isdir(imp_dir):
        print(f"Skip {root}: ori/ or inpainted/ not found")
        return

    os.makedirs(mask_dir, exist_ok=True)

    ori_files = sorted([f for f in os.listdir(ori_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))])
    imp_files = sorted([f for f in os.listdir(imp_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))])

    # assume file names are aligned; use intersection
    common = sorted(set(ori_files).intersection(set(imp_files)))
    if not common:
        print(f"No common files in {ori_dir} and {imp_dir}")
        return

    for name in tqdm(common, desc=f"pairs in {root}"):
        ori_path = os.path.join(ori_dir, name)
        imp_path = os.path.join(imp_dir, name)
        mask_path = os.path.join(mask_dir, os.path.splitext(name)[0] + ".png")

        if not overwrite and os.path.isfile(mask_path):
            continue

        ori = np.array(Image.open(ori_path).convert("RGB"))
        art = np.array(Image.open(imp_path).convert("RGB"))

        if ori.shape != art.shape:
            print(f"Skip {name}: shape mismatch {ori.shape} vs {art.shape}")
            continue

        mask = compute_mask_from_pair(
            ori,
            art,
            threshold=threshold,
            use_max_channel=use_max_channel,
            blur_ksize=blur_ksize,
            open_iter=open_iter,
            close_iter=close_iter,
            min_area=min_area,
        )
        Image.fromarray(mask, mode="L").save(mask_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate binary artifact masks for ArtiDiffuser-Synth (ori vs inpainted differences)."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory of ArtiDiffuser-Synth, e.g. /home/jupyter/data/image__team/ArtiDiffuser-Synth",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=25.0,
        help="Intensity difference threshold for mask (default: 25.0).",
    )
    parser.add_argument(
        "--use_max_channel",
        action="store_true",
        help="Use max over RGB channels for diff aggregation (default: mean).",
    )
    parser.add_argument(
        "--blur_ksize",
        type=int,
        default=0,
        help="If >1, apply uniform blur to diff map before thresholding (requires scipy).",
    )
    parser.add_argument(
        "--open_iter",
        type=int,
        default=0,
        help="Binary opening iterations to remove speckles (requires scipy).",
    )
    parser.add_argument(
        "--close_iter",
        type=int,
        default=0,
        help="Binary closing iterations to fill small holes (requires scipy).",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=0,
        help="Remove connected components smaller than this area (requires scipy).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mask files.",
    )
    args = parser.parse_args()

    artifact_types = []
    for d in sorted(os.listdir(args.root)):
        full = os.path.join(args.root, d)
        if os.path.isdir(full):
            artifact_types.append(full)

    if not artifact_types:
        print(f"No subdirectories found under {args.root}")
        return

    print(f"Processing ArtiDiffuser-Synth under {args.root}")
    for sub in artifact_types:
        process_folder(
            sub,
            threshold=args.threshold,
            overwrite=args.overwrite,
            use_max_channel=args.use_max_channel,
            blur_ksize=args.blur_ksize,
            open_iter=args.open_iter,
            close_iter=args.close_iter,
            min_area=args.min_area,
        )


if __name__ == "__main__":
    main()

