"""Patch sampling with its own RNG, independent of model and event sampling."""
import hashlib
import torch
from utils.event_mil import EventFeatureDataset


def sample_patches(features, cap, training, seed):
    if cap < 1 or features.ndim != 2 or len(features) == 0:
        raise ValueError('Expected a positive cap and a nonempty 2D feature bag')
    if len(features) <= cap:
        return features
    if training:
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(features), generator=rng)[:cap]
    else:
        indices = torch.linspace(0, len(features)-1, cap).long()
    return features[indices]


class ControlledPatchDataset(EventFeatureDataset):
    def __init__(self, *args, sampling_seed=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_seed = sampling_seed
        self.draw = 0

    def __getitem__(self, index):
        # One draw per event, including bags below the cap. Both conditions thus
        # share event/slide-specific seeds regardless of their patch counts.
        self.draw += 1
        self.slide_draw = 0
        self.event_index = index
        return super().__getitem__(index)

    def _sample(self, features):
        self.slide_draw += 1
        key = f'{self.sampling_seed}:{self.draw}:{self.event_index}:{self.slide_draw}'
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') % (2**63-1)
        return sample_patches(features, self.max_patches_per_slide, self.training, seed)
