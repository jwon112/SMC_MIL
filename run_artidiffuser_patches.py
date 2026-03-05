import os
import argparse
import tempfile
import shutil
import subprocess

import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image
import yaml

from create_noisy_patches import (
    get_patch_h5_subdir,
    get_slide_ids,
)


def _load_h5_imgs_and_coords(h5_path):
    if not os.path.isfile(h5_path):
        return None, None, None, "Patch H5 not found"

    with h5py.File(h5_path, "r") as f:
        if "imgs" not in f:
            return None, None, None, "No 'imgs' in H5 (coords-only H5 is not yet supported here)"
        imgs = f["imgs"][:]
        coords = f["coords"][:] if "coords" in f else None
        attrs = {}
        if coords is not None and hasattr(f["coords"], "attrs"):
            attrs = dict(f["coords"].attrs)

    return imgs, coords, attrs, "ok"


def _export_imgs_to_dir(imgs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n = imgs.shape[0]
    names = []
    for i in range(n):
        name = f"{i:06d}.png"
        path = os.path.join(out_dir, name)
        Image.fromarray(imgs[i]).save(path)
        names.append(name)
    # assume all patches share the same spatial size
    h, w = imgs.shape[1], imgs.shape[2]
    return names, (h, w)


def _export_dummy_masks(names, size, out_dir):
    """
    Create dummy all-ones masks (white RGB) for each GT image name.
    These masks indicate that the whole patch is to be processed.
    """
    os.makedirs(out_dir, exist_ok=True)
    h, w = size
    mask_arr = np.ones((h, w, 3), dtype=np.uint8) * 255
    for name in names:
        path = os.path.join(out_dir, name)
        Image.fromarray(mask_arr).save(path)


def _read_imgs_from_dir(names, in_dir):
    restored = []
    for name in names:
        path = os.path.join(in_dir, name)
        if not os.path.isfile(path):
            return None, f"missing restored file {path}"
        img = Image.open(path).convert("RGB")
        restored.append(np.array(img))
    return np.stack(restored, axis=0).astype(np.uint8), "ok"


def _prepare_slide_config(base_conf_path, gt_dir, mask_dir, out_dir, tmp_conf_path):
    """
    Load ArtiDiffuser YAML config, override eval input/output paths
    for a single-slide temporary run, and write to tmp_conf_path.

    Assumes there is exactly one entry under data.eval (same logic as Default_Conf.get_default_eval_name).
    Uses:
      - data.eval[eval_name].paths.gts  as input directory
      - data.eval[eval_name].paths.srs  as output directory
    """
    with open(base_conf_path, "r") as f:
        conf = yaml.safe_load(f)

    if "data" not in conf or "eval" not in conf["data"]:
        raise RuntimeError("Invalid ArtiDiffuser config: missing data.eval")

    eval_candidates = list(conf["data"]["eval"].keys())
    if len(eval_candidates) != 1:
        raise RuntimeError(f"ArtiDiffuser config must have exactly one data.eval entry, got {eval_candidates}")
    eval_name = eval_candidates[0]

    eval_conf = conf["data"]["eval"][eval_name]

    # Override input dirs used by the dataloader
    eval_conf["gt_path"] = gt_dir
    eval_conf["mask_path"] = mask_dir

    # Override output dirs used by eval_imswrite
    paths = eval_conf.get("paths", {})
    paths["gts"] = gt_dir
    paths["gt_keep_masks"] = mask_dir
    paths["srs"] = out_dir
    eval_conf["paths"] = paths

    os.makedirs(os.path.dirname(tmp_conf_path), exist_ok=True)
    with open(tmp_conf_path, "w") as f:
        yaml.safe_dump(conf, f)


def _run_artidiffuser(arti_root, conf_path, extra_args=None):
    """
    Call ArtiDiffuser's test.py with the given config.
    """
    if extra_args is None:
        extra_args = []

    test_py = os.path.join(arti_root, "test.py")
    if not os.path.isfile(test_py):
        raise FileNotFoundError(f"ArtiDiffuser test.py not found at {test_py}")

    cmd = ["python", "test.py", "--conf_path", conf_path] + list(extra_args)
    cwd = arti_root  # 예: ./artidiffuser/model_inference
    subprocess.run(cmd, check=True, cwd=cwd)


def process_slide_with_artidiffuser_from_imgs(
    slide_id,
    patch_h5_subdir,
    output_dir,
    arti_root,
    arti_conf,
    extra_args=None,
):
    in_path = os.path.join(patch_h5_subdir, slide_id + ".h5")
    out_sub = os.path.join(output_dir, "patches")
    os.makedirs(out_sub, exist_ok=True)
    out_path = os.path.join(out_sub, slide_id + ".h5")

    imgs, coords, attrs, msg = _load_h5_imgs_and_coords(in_path)
    if imgs is None:
        return False, 0, msg

    tmp_root = tempfile.mkdtemp(prefix=f"arti_{slide_id}_")
    gt_dir = os.path.join(tmp_root, "gt")
    mask_dir = os.path.join(tmp_root, "mask")
    out_dir = os.path.join(tmp_root, "out")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    try:
        # 1) Export H5 imgs to PNGs (GT) and create dummy full-field masks
        names, size = _export_imgs_to_dir(imgs, gt_dir)
        _export_dummy_masks(names, size, mask_dir)

        # 2) Prepare per-slide config and run ArtiDiffuser
        tmp_conf_path = os.path.join(tmp_root, "conf.yml")
        _prepare_slide_config(arti_conf, gt_dir, mask_dir, out_dir, tmp_conf_path)
        _run_artidiffuser(arti_root, tmp_conf_path, extra_args=extra_args)

        # 3) Read restored PNGs back
        restored, msg2 = _read_imgs_from_dir(names, out_dir)
        if restored is None:
            return False, 0, msg2

        # 4) Save restored imgs (and original coords/attrs) into new H5
        with h5py.File(out_path, "w") as f:
            f.create_dataset("imgs", data=restored, compression="gzip")
            if coords is not None:
                d = f.create_dataset("coords", data=coords)
                for k, v in attrs.items():
                    d.attrs[k] = v

        return True, restored.shape[0], "ok"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def process_slide_with_artidiffuser_from_wsi(
    slide_id,
    patch_h5_subdir,
    output_dir,
    data_slide_dir,
    slide_ext,
    arti_root,
    arti_conf,
    extra_args=None,
):
    """
    CLAM-style WSI mode:
      - Read coords-only patch H5 for a slide.
      - Use openslide to extract patches on the fly from the WSI.
      - Run ArtiDiffuser on these patches (with dummy full-field masks).
      - Save restored patches + original coords back to a new H5.
    """
    import openslide

    in_path = os.path.join(patch_h5_subdir, slide_id + ".h5")
    wsi_path = os.path.join(data_slide_dir, slide_id + slide_ext)
    out_sub = os.path.join(output_dir, "patches")
    os.makedirs(out_sub, exist_ok=True)
    out_path = os.path.join(out_sub, slide_id + ".h5")

    if not os.path.isfile(in_path):
        return False, 0, "Patch H5 not found"
    if not os.path.isfile(wsi_path):
        return False, 0, f"WSI not found: {wsi_path}"

    with h5py.File(in_path, "r") as f:
        if "coords" not in f:
            return False, 0, "No 'coords' in H5"
        coords = f["coords"][:]
        patch_level = int(f["coords"].attrs.get("patch_level", 0))
        patch_size = int(f["coords"].attrs.get("patch_size", 256))
        attrs = dict(f["coords"].attrs)

    wsi = openslide.open_slide(wsi_path)
    imgs = []
    for coord in coords:
        xy = tuple(int(x) for x in coord)
        pil = wsi.read_region(xy, patch_level, (patch_size, patch_size)).convert("RGB")
        imgs.append(np.array(pil))
    imgs = np.array(imgs, dtype=np.uint8)

    tmp_root = tempfile.mkdtemp(prefix=f"arti_{slide_id}_")
    gt_dir = os.path.join(tmp_root, "gt")
    mask_dir = os.path.join(tmp_root, "mask")
    out_dir = os.path.join(tmp_root, "out")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    try:
        # 1) Export WSI patches to PNGs (GT) and create dummy full-field masks
        names, size = _export_imgs_to_dir(imgs, gt_dir)
        _export_dummy_masks(names, size, mask_dir)

        # 2) Prepare per-slide config and run ArtiDiffuser
        tmp_conf_path = os.path.join(tmp_root, "conf.yml")
        _prepare_slide_config(arti_conf, gt_dir, mask_dir, out_dir, tmp_conf_path)
        _run_artidiffuser(arti_root, tmp_conf_path, extra_args=extra_args)

        # 3) Read restored PNGs back
        restored, msg2 = _read_imgs_from_dir(names, out_dir)
        if restored is None:
            return False, 0, msg2

        # 4) Save restored imgs (and original coords/attrs) into new H5
        with h5py.File(out_path, "w") as f:
            f.create_dataset("imgs", data=restored, compression="gzip")
            d = f.create_dataset("coords", data=coords)
            for k, v in attrs.items():
                d.attrs[k] = v

        return True, restored.shape[0], "ok"
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run ArtiDiffuser restoration on CLAM patch H5 files.\n"
            "Input (mode 1): --patch_h5_dir with patches/slide_id.h5 containing 'imgs'.\n"
            "Input (mode 2): --patch_h5_dir with coords-only H5 and --data_slide_dir with WSIs.\n"
            "Output: restored patches to --output_dir/patches/slide_id.h5 (same coords/attrs)."
        )
    )
    parser.add_argument(
        "--patch_h5_dir",
        type=str,
        required=True,
        help="Root dir containing CLAM patch H5s (patches/ or h5_files/ with slide_id.h5).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output root; restored patches will be written to output_dir/patches/slide_id.h5.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Optional CSV to restrict slides (same format as used in create_noisy_patches.py).",
    )
    parser.add_argument(
        "--data_slide_dir",
        type=str,
        default=None,
        help="(Not yet supported) If patch H5 has only coords, WSI directory to read patches from.",
    )
    parser.add_argument(
        "--slide_ext",
        type=str,
        default=".svs",
        help="WSI file extension (default .svs).",
    )

    parser.add_argument(
        "--arti_root",
        type=str,
        default=os.path.join("artidiffuser", "model_inference"),
        help="Path to ArtiDiffuser model_inference directory (default: ./artidiffuser/model_inference).",
    )
    parser.add_argument(
        "--arti_conf",
        type=str,
        default=os.path.join("artidiffuser", "model_inference", "confs", "example_classifier.yml"),
        help="Base ArtiDiffuser YAML config to use (will be copied & patched per slide).",
    )

    args, unknown = parser.parse_known_args()

    patch_h5_sub = get_patch_h5_subdir(args.patch_h5_dir)
    slide_ids = get_slide_ids(patch_h5_sub, args.csv_path)
    if not slide_ids:
        print("No slides to process.")
        return

    use_wsi = args.data_slide_dir is not None and os.path.isdir(args.data_slide_dir)
    if use_wsi:
        print("Mode: read patches from WSI using coords-only H5, then run ArtiDiffuser per slide.")
    else:
        print("Mode: read patches from H5 'imgs' and run ArtiDiffuser per slide.")

    ok, fail, total = 0, 0, 0
    for slide_id in tqdm(slide_ids, desc="slides"):
        try:
            if use_wsi:
                success, n, msg = process_slide_with_artidiffuser_from_wsi(
                    slide_id=slide_id,
                    patch_h5_subdir=patch_h5_sub,
                    output_dir=args.output_dir,
                    data_slide_dir=args.data_slide_dir,
                    slide_ext=args.slide_ext,
                    arti_root=args.arti_root,
                    arti_conf=args.arti_conf,
                    extra_args=unknown,
                )
            else:
                success, n, msg = process_slide_with_artidiffuser_from_imgs(
                    slide_id=slide_id,
                    patch_h5_subdir=patch_h5_sub,
                    output_dir=args.output_dir,
                    arti_root=args.arti_root,
                    arti_conf=args.arti_conf,
                    extra_args=unknown,
                )
        except Exception as e:
            success, n, msg = False, 0, f"Exception: {e}"

        if success:
            ok += 1
            total += n
        else:
            fail += 1
            if fail <= 3:
                print(f"Skip {slide_id}: {msg}")

    print(f"Done. OK={ok}, Failed={fail}, Total restored patches={total}")
    print(
        f"Restored patches saved under {os.path.join(args.output_dir, 'patches')}.\n"
        f"Use this directory as patch_h5_dir for CLAM feature extraction / training."
    )


if __name__ == "__main__":
    main()

