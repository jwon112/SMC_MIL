import os
import argparse
import tempfile
import shutil

import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel

from create_noisy_patches import (
    get_patch_h5_subdir,
    get_slide_ids,
)

from latent_artifusion.pipeline_latent_diffusion import LatentDiffusionImpaintingPipeline


def _load_coords_and_meta(h5_path):
    if not os.path.isfile(h5_path):
        return None, None, None, "Patch H5 not found"

    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            return None, None, None, "No 'coords' in H5"
        coords = f["coords"][:]
        patch_level = int(f["coords"].attrs.get("patch_level", 0))
        patch_size = int(f["coords"].attrs.get("patch_size", 256))
        attrs = dict(f["coords"].attrs)

    return coords, patch_level, (patch_size, patch_size), attrs


def _read_patches_from_wsi(wsi_path, coords, patch_level, patch_size_hw):
    import openslide

    if not os.path.isfile(wsi_path):
        return None, f"WSI not found: {wsi_path}"

    patch_h, patch_w = patch_size_hw
    wsi = openslide.open_slide(wsi_path)
    imgs = []
    for coord in coords:
        xy = tuple(int(x) for x in coord)
        pil = wsi.read_region(xy, patch_level, (patch_w, patch_h)).convert("RGB")
        imgs.append(np.array(pil, dtype=np.uint8))
    imgs = np.asarray(imgs, dtype=np.uint8)
    return imgs, "ok"


def _numpy_to_pil_batch(images):
    """
    images: uint8 [N,H,W,C] (RGB)
    """
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Expected images of shape [N,H,W,3] uint8")
    return [Image.fromarray(im) for im in images]


def _make_full_one_masks_like(images):
    """
    Create all-ones (white) masks for each image; same spatial size.
    """
    n, h, w, _ = images.shape
    masks = []
    arr = np.ones((h, w), dtype=np.uint8) * 255
    for _ in range(n):
        masks.append(Image.fromarray(arr, mode="L"))
    return masks


def build_latentartifusion_pipeline(vae_dir, unet_dir, scheduler_dir, device, torch_dtype=torch.float16):
    """
    Build LatentDiffusionImpaintingPipeline with pretrained VAE/UNet/scheduler.
    """
    vae = AutoencoderKL.from_pretrained(vae_dir, torch_dtype=torch_dtype)
    unet = UNet2DModel.from_pretrained(unet_dir, subfolder="unet", torch_dtype=torch_dtype)
    scheduler = DDPMScheduler.from_pretrained(scheduler_dir)

    vae.requires_grad_(False)
    unet.requires_grad_(False)

    pipe = LatentDiffusionImpaintingPipeline(
        vae=vae,
        unet=unet,
        scheduler=scheduler,
    )
    pipe.to(device)
    return pipe


def restore_patches_latent(
    pipe,
    patches_u8,
    num_inference_steps=50,
    generator=None,
):
    """
    Run LatentArtiFusion inpainting on a batch of patches using full-field masks.

    patches_u8: uint8 [N,H,W,C]
    return: uint8 [N,H,W,C]
    """
    pil_images = _numpy_to_pil_batch(patches_u8)
    masks = _make_full_one_masks_like(patches_u8)

    restored_list = []
    for img, m in zip(pil_images, masks):
        out = pipe(
            original_image=img,
            mask_image=m,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        # pipeline returns a list of PIL images in .images
        restored_pil = out.images[0]
        restored_list.append(np.array(restored_pil.convert("RGB"), dtype=np.uint8))

    restored = np.stack(restored_list, axis=0)
    return restored


def process_slide_with_latentartifusion_from_wsi(
    slide_id,
    patch_h5_subdir,
    output_dir,
    data_slide_dir,
    slide_ext,
    pipe,
    num_inference_steps=50,
    seed=42,
):
    in_path = os.path.join(patch_h5_subdir, slide_id + ".h5")
    out_sub = os.path.join(output_dir, "patches")
    os.makedirs(out_sub, exist_ok=True)
    out_path = os.path.join(out_sub, slide_id + ".h5")

    coords, patch_level, patch_size_hw, attrs = _load_coords_and_meta(in_path)
    if coords is None:
        return False, 0, patch_level  # patch_level carries error message here

    wsi_path = os.path.join(data_slide_dir, slide_id + slide_ext)
    imgs, msg = _read_patches_from_wsi(wsi_path, coords, patch_level, patch_size_hw)
    if imgs is None:
        return False, 0, msg

    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    restored = restore_patches_latent(
        pipe,
        imgs,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )

    # Save restored patches and coords to H5
    with h5py.File(out_path, "w") as f:
        f.create_dataset("imgs", data=restored, compression="gzip")
        d = f.create_dataset("coords", data=coords)
        for k, v in attrs.items():
            d.attrs[k] = v

    return True, restored.shape[0], "ok"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run LatentArtiFusion (LatentDiffusionImpaintingPipeline) restoration on CLAM patch H5 files.\n"
            "Input (WSI mode): --patch_h5_dir with coords-only H5 and --data_slide_dir with WSIs.\n"
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
        required=True,
        help="WSI directory to read patches from (coords-only H5).",
    )
    parser.add_argument(
        "--slide_ext",
        type=str,
        default=".svs",
        help="WSI file extension (default .svs).",
    )
    parser.add_argument(
        "--vae_dir",
        type=str,
        required=True,
        help="Path to pretrained VAE directory (Stable Diffusion VAE).",
    )
    parser.add_argument(
        "--unet_dir",
        type=str,
        required=True,
        help="Path to pretrained latent UNet directory (LatentArtiFusion UNet).",
    )
    parser.add_argument(
        "--scheduler_dir",
        type=str,
        default=os.path.join("latent_artifusion", "scheduler"),
        help="Path to scheduler config directory (default: ./latent_artifusion/scheduler).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run LatentArtiFusion on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="Torch dtype to use for VAE/UNet (default: fp16).",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of diffusion steps per patch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generator.",
    )

    args = parser.parse_args()

    patch_h5_sub = get_patch_h5_subdir(args.patch_h5_dir)
    slide_ids = get_slide_ids(patch_h5_sub, args.csv_path)
    if not slide_ids:
        print("No slides to process.")
        return

    if not os.path.isdir(args.data_slide_dir):
        print(f"data_slide_dir does not exist: {args.data_slide_dir}")
        return

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    pipe = build_latentartifusion_pipeline(
        vae_dir=args.vae_dir,
        unet_dir=args.unet_dir,
        scheduler_dir=args.scheduler_dir,
        device=args.device,
        torch_dtype=torch_dtype,
    )

    print("Mode: read patches from WSI using coords-only H5, then run LatentArtiFusion per slide.")

    ok, fail, total = 0, 0, 0
    for slide_id in tqdm(slide_ids, desc="slides"):
        try:
            success, n, msg = process_slide_with_latentartifusion_from_wsi(
                slide_id=slide_id,
                patch_h5_subdir=patch_h5_sub,
                output_dir=args.output_dir,
                data_slide_dir=args.data_slide_dir,
                slide_ext=args.slide_ext,
                pipe=pipe,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
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

