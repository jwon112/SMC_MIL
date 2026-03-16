import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_mask_from_pair(ori_img: np.ndarray, art_img: np.ndarray, threshold: float) -> np.ndarray:
    """
    ori_img, art_img: uint8 [H, W, 3]
    return: uint8 [H, W] in {0, 255}
    """
    diff = np.abs(ori_img.astype(np.float32) - art_img.astype(np.float32))
    # average over channels
    g = diff.mean(axis=2)
    mask = (g > threshold).astype(np.uint8) * 255
    return mask


def process_folder(root: str, threshold: float, overwrite: bool = False):
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

        mask = compute_mask_from_pair(ori, art, threshold=threshold)
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
        process_folder(sub, threshold=args.threshold, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

