from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms as T
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pad as tv_pad


@dataclass(frozen=True)
class DataSpec:
    dataset: str = "voc"
    data_root: str = "./_data"
    year: str = "2012"
    image_set: str = "train"
    download: bool = True
    num_workers: int = 4
    batch_size: int = 8
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2

    # segmentation-friendly transforms (defaults: "standard" recipe)
    # - random scale jitter + random crop + hflip
    # - mild photometric jitter on image only
    crop_size: int = 512
    scale_min: float = 0.5
    scale_max: float = 2.0
    hflip: float = 0.5
    color_jitter: float = 0.2  # applied to image only
    gaussian_blur_p: float = 0.0  # keep off by default (many baselines don't use it)

    # normalization (ImageNet default)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def _voc_pair_transform(spec: DataSpec) -> Callable:
    """
    Apply *paired* transforms so image and mask stay aligned.
    Note: VOC masks use class indices and 255 for ignore.
    """

    def _tfm(img, mask):
        # random scale jitter (paired)
        if spec.scale_min and spec.scale_max and spec.scale_max > 0:
            s = float(torch.empty(1).uniform_(spec.scale_min, spec.scale_max).item())
            new_h = max(1, int(round(img.size[1] * s)))
            new_w = max(1, int(round(img.size[0] * s)))
            img = T.functional.resize(img, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)
            mask = T.functional.resize(mask, (new_h, new_w), interpolation=InterpolationMode.NEAREST)

        # pad to at least crop size (paired)
        th = int(spec.crop_size)
        tw = int(spec.crop_size)
        pad_h = max(0, th - img.size[1])
        pad_w = max(0, tw - img.size[0])
        if pad_h > 0 or pad_w > 0:
            # right/bottom padding only; mask padded with ignore_index=255
            img = tv_pad(img, padding=[0, 0, pad_w, pad_h], fill=0)
            mask = tv_pad(mask, padding=[0, 0, pad_w, pad_h], fill=255)

        # random crop (paired)
        i, j, h, w = T.RandomCrop.get_params(img, output_size=(th, tw))
        img = T.functional.crop(img, i, j, h, w)
        mask = T.functional.crop(mask, i, j, h, w)

        # hflip
        if spec.hflip and torch.rand(()) < spec.hflip:
            img = T.functional.hflip(img)
            mask = T.functional.hflip(mask)

        # photometric aug (image only)
        cj = float(spec.color_jitter or 0.0)
        if cj > 0:
            img = T.ColorJitter(brightness=cj, contrast=cj, saturation=cj, hue=min(0.1, cj / 2))(img)
        if spec.gaussian_blur_p and torch.rand(()) < float(spec.gaussian_blur_p):
            img = T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))(img)

        img = T.functional.to_tensor(img)  # float [0,1]
        img = T.functional.normalize(img, mean=spec.mean, std=spec.std)

        mask = torch.from_numpy(np.array(mask, dtype=np.int64))  # HxW, values in [0..20] or 255(ignore)
        return img, mask

    return _tfm


class PairedTransformDataset(Dataset):
    def __init__(self, base: Dataset, paired_transform: Callable):
        self.base = base
        self.paired_transform = paired_transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask = self.base[idx]
        return self.paired_transform(img, mask)


class ADE20KDataset(Dataset):
    """
    ADE20K (ADEChallengeData2016) dataset wrapper.
    Expected layout:
    - <root>/images/training/*.jpg
    - <root>/images/validation/*.jpg
    - <root>/annotations/training/*.png
    - <root>/annotations/validation/*.png
    """

    def __init__(self, root: str, image_set: str):
        root_path = Path(root).resolve()

        # Accept both:
        # - data_root=/.../ADEChallengeData2016
        # - data_root=/.../data   (contains ADEChallengeData2016 subfolder)
        candidates = [
            root_path,
            root_path / "ADEChallengeData2016",
            root_path / "ADEChallangeData2016",  # common typo seen in some paths
        ]
        base = None
        for c in candidates:
            if (c / "images").is_dir() and (c / "annotations").is_dir():
                base = c
                break
        if base is None:
            raise FileNotFoundError(
                "ADE20K root not found. Expected one of: "
                f"{', '.join(str(c) for c in candidates)} with images/ and annotations/."
            )

        split = (image_set or "train").lower()
        if split in {"train", "training"}:
            split_dir = "training"
        elif split in {"val", "validation"}:
            split_dir = "validation"
        else:
            raise ValueError(f"Unsupported image_set for ADE20K: {image_set}")

        self.img_dir = base / "images" / split_dir
        self.ann_dir = base / "annotations" / split_dir

        if not self.img_dir.is_dir() or not self.ann_dir.is_dir():
            raise FileNotFoundError(f"Missing ADE20K split dirs: {self.img_dir} and/or {self.ann_dir}")

        self.samples = []
        for p in sorted(self.img_dir.glob("*.jpg")):
            m = self.ann_dir / f"{p.stem}.png"
            if m.exists():
                self.samples.append((p, m))

        if len(self.samples) == 0:
            raise RuntimeError(f"No ADE20K samples found under {self.img_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        return img, mask


def voc_is_prepared(root: str, year: str = "2012") -> bool:
    root = os.path.abspath(root)
    voc_dir = os.path.join(root, "VOCdevkit", f"VOC{year}")
    # A lightweight check that mirrors torchvision's expected layout.
    return os.path.isdir(voc_dir) and os.path.isfile(os.path.join(voc_dir, "ImageSets", "Segmentation", "train.txt"))


def build_dataset(spec: DataSpec) -> Dataset:
    ds = (spec.dataset or "voc").lower()
    root = os.path.abspath(spec.data_root)
    if ds == "voc":
        if spec.download and voc_is_prepared(root, spec.year):
            # Avoid re-downloading when the dataset is already present.
            download = False
        else:
            download = spec.download
        base = VOCSegmentation(
            root=root,
            year=spec.year,
            image_set=spec.image_set,
            download=download,
            transforms=None,
            transform=None,
            target_transform=None,
        )
        return PairedTransformDataset(base, _voc_pair_transform(spec))

    if ds == "ade20k":
        if spec.download:
            raise ValueError("ADE20K auto-download is not supported. Set --no-download and place dataset manually.")
        base = ADE20KDataset(root=root, image_set=spec.image_set)
        return PairedTransformDataset(base, _voc_pair_transform(spec))

    raise ValueError(f"Unsupported dataset: {spec.dataset} (expected 'voc' or 'ade20k')")


def build_loader(spec: DataSpec, *, shuffle: bool) -> DataLoader:
    ds = build_dataset(spec)
    kwargs = {}
    if spec.num_workers and spec.num_workers > 0:
        kwargs["persistent_workers"] = bool(spec.persistent_workers)
        kwargs["prefetch_factor"] = int(spec.prefetch_factor)
    return DataLoader(
        ds,
        batch_size=spec.batch_size,
        shuffle=shuffle,
        num_workers=spec.num_workers,
        pin_memory=spec.pin_memory,
        drop_last=shuffle,
        **kwargs,
    )

