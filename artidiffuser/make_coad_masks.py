import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def list_images(dir_path: str):
    return sorted([f for f in os.listdir(dir_path) if f.lower().endswith(EXTS)])


def hsv_marking_mask(rgb: np.ndarray, sat_thresh: float = 0.3, val_thresh: float = 0.2) -> np.ndarray:
    """
    Very rough heuristic: marking/tattoo tend to be more saturated / darker than background.
    rgb: uint8 [H,W,3]
    return: uint8 mask [H,W] in {0,255}
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    maxc = np.max(rgb_f, axis=2)
    minc = np.min(rgb_f, axis=2)
    v = maxc
    s = np.zeros_like(v)
    nonzero = maxc > 0
    s[nonzero] = (maxc[nonzero] - minc[nonzero]) / maxc[nonzero]

    mask = (s > sat_thresh) & (v < 1.0 - val_thresh)
    return (mask.astype(np.uint8) * 255)


def blur_like_mask(rgb: np.ndarray, ksize: int = 5, diff_thresh: float = 5.0) -> np.ndarray:
    """
    Rough 'out_of_focus' style mask: pixels whose local average differs little from neighbors
    (low-frequency region). Very crude.
    """
    from scipy.ndimage import uniform_filter

    gray = rgb.astype(np.float32).mean(axis=2)
    smooth = uniform_filter(gray, size=ksize)
    diff = np.abs(gray - smooth)
    mask = diff < diff_thresh
    return (mask.astype(np.uint8) * 255)


def edge_like_mask(rgb: np.ndarray, ksize: int = 5, grad_thresh: float = 20.0) -> np.ndarray:
    """
    Rough 'tissue_folding' style mask: pixels with higher gradient magnitude.
    """
    from scipy.ndimage import sobel

    gray = rgb.astype(np.float32).mean(axis=2)
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    grad = np.hypot(gx, gy)
    mask = grad > grad_thresh
    return (mask.astype(np.uint8) * 255)


def make_mask_for_image(path: str, kind: str) -> np.ndarray:
    img = np.array(Image.open(path).convert("RGB"))
    if kind in ("marking", "tattoo"):
        return hsv_marking_mask(img)
    elif kind == "out_of_focus":
        return blur_like_mask(img)
    elif kind == "tissue_folding":
        return edge_like_mask(img)
    else:
        # normal or unknown → empty mask
        h, w, _ = img.shape
        return np.zeros((h, w), dtype=np.uint8)


def process_coad_root(root: str, overwrite: bool = False):
    """
    root: /home/jupyter/data/image_team/COAD-Artifact
    Subdirs: normal/, marking/, out_of_focus/, tattoo/, tissue_folding/
    """
    if not os.path.isdir(root):
        print(f"COAD-Artifact root not found: {root}")
        return

    for kind in sorted(os.listdir(root)):
        kind_dir = os.path.join(root, kind)
        if not os.path.isdir(kind_dir):
            continue

        img_files = list_images(kind_dir)
        if not img_files:
            continue

        mask_dir = os.path.join(kind_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)

        print(f"Processing COAD-Artifact {kind} in {kind_dir}")
        for name in tqdm(img_files, desc=f"{kind}"):
            img_path = os.path.join(kind_dir, name)
            mask_name = os.path.splitext(name)[0] + ".png"
            mask_path = os.path.join(mask_dir, mask_name)

            if not overwrite and os.path.isfile(mask_path):
                continue

            mask = make_mask_for_image(img_path, kind)
            Image.fromarray(mask, mode="L").save(mask_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate rough artifact masks for COAD-Artifact dataset (heuristic-based)."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory of COAD-Artifact, e.g. /home/jupyter/data/image_team/COAD-Artifact",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mask files.",
    )
    args = parser.parse_args()

    process_coad_root(args.root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

