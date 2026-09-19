"""Data helpers shared by the SMC stain-aware event MIL scripts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from models.stain_aware_event_mil import STAIN_GROUPS


def normalize_stain_group(value: object) -> str | None:
    text = str(value).strip()
    mapping = {"HE": "HE", "IHC": "IHC", "special_other": "other", "other": "other"}
    return mapping.get(text)


def load_event_tables(event_csv: Path, slide_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(event_csv, dtype={"event_id": str, "case_id": str})
    slides = pd.read_csv(slide_csv, dtype={"event_id": str, "slide_id": str})
    required_events = {"event_id", "case_id", "label"}
    required_slides = {"event_id", "slide_id", "stain_group"}
    if missing := required_events.difference(events.columns):
        raise ValueError(f"Event CSV missing columns: {sorted(missing)}")
    if missing := required_slides.difference(slides.columns):
        raise ValueError(f"Event slide CSV missing columns: {sorted(missing)}")
    if events["event_id"].duplicated().any():
        raise ValueError("event_id must be unique in event CSV")
    unknown = set(slides["event_id"]).difference(events["event_id"])
    if unknown:
        raise ValueError(f"Slides reference unknown events: {sorted(unknown)[:5]}")
    return events, slides


class EventFeatureDataset(Dataset):
    def __init__(
        self,
        events: pd.DataFrame,
        slides: pd.DataFrame,
        feature_dir: Path,
        max_patches_per_slide: int,
        training: bool,
    ) -> None:
        self.events = events.reset_index(drop=True).copy()
        self.feature_dir = feature_dir
        self.max_patches_per_slide = max_patches_per_slide
        self.training = training
        grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for row in slides.itertuples(index=False):
            stain = normalize_stain_group(row.stain_group)
            if stain is not None:
                grouped[str(row.event_id)].append((stain, str(row.slide_id)))
        self.slide_map = dict(grouped)
        missing = set(self.events["event_id"]).difference(self.slide_map)
        if missing:
            raise ValueError(f"Events without known-stain slides: {sorted(missing)[:5]}")

    def __len__(self) -> int:
        return len(self.events)

    def _sample(self, features: torch.Tensor) -> torch.Tensor:
        if len(features) <= self.max_patches_per_slide:
            return features
        if self.training:
            indices = torch.randperm(len(features))[: self.max_patches_per_slide]
        else:
            indices = torch.linspace(0, len(features) - 1, self.max_patches_per_slide).long()
        return features[indices]

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.events.iloc[index]
        event_id = str(row["event_id"])
        event_slides: list[tuple[str, torch.Tensor]] = []
        for stain_group, slide_id in self.slide_map[event_id]:
            path = self.feature_dir / "pt_files" / f"{slide_id}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing feature bag: {path}")
            features = torch.load(path, map_location="cpu", weights_only=True).float()
            event_slides.append((stain_group, self._sample(features)))
        return {
            "event_id": event_id,
            "case_id": str(row["case_id"]),
            "label": int(row["label"]),
            "slides": event_slides,
        }


def collate_event(batch: list[dict[str, object]]) -> dict[str, object]:
    if len(batch) != 1:
        raise ValueError("Event MIL currently requires batch_size=1")
    return batch[0]


def verify_feature_bags(slides: pd.DataFrame, feature_dir: Path) -> None:
    missing = [
        slide_id for slide_id in slides["slide_id"].astype(str)
        if not (feature_dir / "pt_files" / f"{slide_id}.pt").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} feature bags; first: {missing[:5]}")
