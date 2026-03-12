from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode


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

    # segmentation-friendly transforms
    resize: Optional[int] = 384
    crop_size: Optional[int] = 352
    hflip: float = 0.5


def _voc_pair_transform(spec: DataSpec) -> Callable:
    """
    Apply *paired* transforms so image and mask stay aligned.
    Note: VOC masks use class indices and 255 for ignore.
    """

    def _tfm(img, mask):
        # resize (optional)
        if spec.resize:
            img = T.functional.resize(img, spec.resize, interpolation=InterpolationMode.BILINEAR)
            mask = T.functional.resize(mask, spec.resize, interpolation=InterpolationMode.NEAREST)

        # random crop (optional)
        if spec.crop_size:
            i, j, h, w = T.RandomCrop.get_params(img, output_size=(spec.crop_size, spec.crop_size))
            img = T.functional.crop(img, i, j, h, w)
            mask = T.functional.crop(mask, i, j, h, w)

        # hflip
        if spec.hflip and torch.rand(()) < spec.hflip:
            img = T.functional.hflip(img)
            mask = T.functional.hflip(mask)

        img = T.functional.to_tensor(img)  # float [0,1]
        img = T.functional.normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

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


def build_dataset(spec: DataSpec) -> Dataset:
    ds = (spec.dataset or "voc").lower()
    if ds != "voc":
        raise ValueError(f"Only dataset='voc' is implemented right now (got {spec.dataset})")

    root = os.path.abspath(spec.data_root)
    base = VOCSegmentation(
        root=root,
        year=spec.year,
        image_set=spec.image_set,
        download=spec.download,
        transforms=None,
        transform=None,
        target_transform=None,
    )
    return PairedTransformDataset(base, _voc_pair_transform(spec))


def build_loader(spec: DataSpec, *, shuffle: bool) -> DataLoader:
    ds = build_dataset(spec)
    return DataLoader(
        ds,
        batch_size=spec.batch_size,
        shuffle=shuffle,
        num_workers=spec.num_workers,
        pin_memory=spec.pin_memory,
        drop_last=shuffle,
    )

