import os
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _list_images(dir_path: str) -> List[str]:
    return sorted(
        f for f in os.listdir(dir_path)
        if f.lower().endswith(EXTS)
    )


def _load_image(path: str) -> torch.Tensor:
    """[H,W,3] uint8 -> [3,H,W] float32 in [-1,1]."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # [3,H,W]
    tensor = torch.from_numpy(arr)
    tensor = tensor * 2.0 - 1.0
    return tensor


def _load_mask(path: str, shape_hw: Optional[tuple] = None) -> torch.Tensor:
    """[H,W] uint8 {0,255} -> [1,H,W] float32 {0,1}."""
    mask = Image.open(path).convert("L")
    if shape_hw is not None:
        mask = mask.resize((shape_hw[1], shape_hw[0]), Image.NEAREST)
    arr = (np.array(mask) > 0).astype(np.float32)
    arr = arr[None, ...]  # [1,H,W]
    return torch.from_numpy(arr)


class CoadDataset(Dataset):
    """
    COAD-Artifact용 Dataset.

    root:
      /home/jupyter/data/image_team/COAD-Artifact
        normal/
        marking/
        out_of_focus/
        tattoo/
        tissue_folding/
    각 타입 디렉터리 안에 이미지와 (선택적으로) masks/*.png 가 있다고 가정.
    """

    def __init__(self, root: str, cls_map: Dict[str, int]):
        super().__init__()
        self.root = root
        self.cls_map = cls_map
        self.items = []  # (img_path, mask_path or None, cls, src)

        for kind in os.listdir(root):
            kind_dir = os.path.join(root, kind)
            if not os.path.isdir(kind_dir):
                continue
            if kind not in cls_map:
                continue

            label = cls_map[kind]
            mask_dir = os.path.join(kind_dir, "masks")
            has_mask = os.path.isdir(mask_dir)

            for name in _list_images(kind_dir):
                img_path = os.path.join(kind_dir, name)
                mask_path = None
                if has_mask:
                    mname = os.path.splitext(name)[0] + ".png"
                    cand = os.path.join(mask_dir, mname)
                    if os.path.isfile(cand):
                        mask_path = cand
                self.items.append((img_path, mask_path, label, "coad"))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, mask_path, label, src = self.items[idx]
        img = _load_image(img_path)
        _, H, W = img.shape

        if mask_path is not None:
            mask = _load_mask(mask_path, shape_hw=(H, W))
        else:
            mask = torch.zeros(1, H, W, dtype=torch.float32)

        return {
            "img": img,
            "cls": torch.tensor(label, dtype=torch.long),
            "mask": mask,
            "src": src,
        }


class SynthDataset(Dataset):
    """
    ArtiDiffuser-Synth용 Dataset.

    root:
      /home/jupyter/data/image__team/ArtiDiffuser-Synth
        marking/
          inpainted/
          masks/
        out_of_focus/
        tattoo/
        tissue_folding/

    입력 이미지는 inpainted, 마스크는 masks 를 사용.
    (ori는 전략 B에서 별도 로더/파이프라인으로 활용 가능)
    """

    def __init__(self, root: str, cls_map: Dict[str, int]):
        super().__init__()
        self.root = root
        self.cls_map = cls_map
        self.items = []  # (img_path, mask_path, cls, src)

        for kind in os.listdir(root):
            kind_dir = os.path.join(root, kind)
            if not os.path.isdir(kind_dir):
                continue
            if kind not in cls_map:
                continue

            label = cls_map[kind]
            imp_dir = os.path.join(kind_dir, "inpainted")
            mask_dir = os.path.join(kind_dir, "masks")
            if not (os.path.isdir(imp_dir) and os.path.isdir(mask_dir)):
                continue

            for name in _list_images(imp_dir):
                img_path = os.path.join(imp_dir, name)
                mname = os.path.splitext(name)[0] + ".png"
                mask_path = os.path.join(mask_dir, mname)
                if not os.path.isfile(mask_path):
                    continue
                self.items.append((img_path, mask_path, label, "synth"))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, mask_path, label, src = self.items[idx]
        img = _load_image(img_path)
        _, H, W = img.shape
        mask = _load_mask(mask_path, shape_hw=(H, W))

        return {
            "img": img,
            "cls": torch.tensor(label, dtype=torch.long),
            "mask": mask,
            "src": src,
        }


class SynthEvalDataset(Dataset):
    """
    ArtiDiffuser-Synth (inpainted, ori) 페어용 eval Dataset.
    ori/ 와 inpainted/ 만 필요. masks 불필요.
    """

    def __init__(self, root: str, cls_map: Dict[str, int]):
        super().__init__()
        self.root = root
        self.cls_map = cls_map
        self.items = []  # (inpainted_path, ori_path, cls)

        for kind in os.listdir(root):
            kind_dir = os.path.join(root, kind)
            if not os.path.isdir(kind_dir):
                continue
            if kind not in cls_map:
                continue

            label = cls_map[kind]
            ori_dir = os.path.join(kind_dir, "ori")
            imp_dir = os.path.join(kind_dir, "inpainted")
            if not (os.path.isdir(ori_dir) and os.path.isdir(imp_dir)):
                continue

            ori_files = set(_list_images(ori_dir))
            imp_files = set(_list_images(imp_dir))
            common = sorted(ori_files & imp_files)

            for name in common:
                imp_path = os.path.join(imp_dir, name)
                ori_path = os.path.join(ori_dir, name)
                self.items.append((imp_path, ori_path, label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        imp_path, ori_path, label = self.items[idx]
        inpainted = _load_image(imp_path)
        ori = _load_image(ori_path)
        return {
            "inpainted": inpainted,
            "ori": ori,
            "cls": torch.tensor(label, dtype=torch.long),
        }


class CombinedDataset(Dataset):
    """
    COAD + Synth를 단순 concat으로 합친 Dataset 래퍼.
    """

    def __init__(self, coad_ds: CoadDataset, synth_ds: SynthDataset):
        super().__init__()
        self.coad_ds = coad_ds
        self.synth_ds = synth_ds
        self.n_coad = len(coad_ds)
        self.n_synth = len(synth_ds)

    def __len__(self) -> int:
        return self.n_coad + self.n_synth

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < self.n_coad:
            return self.coad_ds[idx]
        return self.synth_ds[idx - self.n_coad]


def build_dataloader(
    coad_root: str,
    synth_root: Optional[str],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    coad_only: bool = False,
) -> DataLoader:
    """
    COAD-only 또는 COAD+Synth 학습용 DataLoader 생성.

    반환되는 배치 딕셔너리 포맷:
      - img:  [B, 3, H, W] float32 in [-1,1]
      - cls:  [B] long (0~4)
      - mask: [B, 1, H, W] float32 in {0,1}
      - src:  "coad" 또는 "synth"
    """

    cls_map = {
        "normal": 0,
        "marking": 1,
        "out_of_focus": 2,
        "tattoo": 3,
        "tissue_folding": 4,
    }

    coad_ds = CoadDataset(coad_root, cls_map)

    if coad_only or synth_root is None:
        ds = coad_ds
    else:
        synth_ds = SynthDataset(synth_root, cls_map)
        ds = CombinedDataset(coad_ds, synth_ds)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

