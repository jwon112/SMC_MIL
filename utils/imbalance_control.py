"""Explicit event-level imbalance policies, including batch-one-safe weighted CE."""
import hashlib
import torch
from torch.nn import functional as F
from torch.utils.data import RandomSampler, WeightedRandomSampler

CONDITIONS = ('balanced_sampler', 'natural_ce', 'natural_weighted_ce')


def balanced_weights(labels):
    labels = torch.as_tensor(list(labels), dtype=torch.long)
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError('Nonbinary labels')
    counts = torch.bincount(labels, minlength=2)
    if (counts == 0).any():
        raise ValueError('Fit patients must contain both classes')
    return len(labels)/(2*counts.to(torch.float32))


def make_sampler(labels, condition, generator):
    if condition not in CONDITIONS:
        raise ValueError('Unknown imbalance condition')
    labels = torch.as_tensor(list(labels), dtype=torch.long)
    weights = balanced_weights(labels)
    if condition == 'balanced_sampler':
        return WeightedRandomSampler(weights[labels].double(), len(labels), replacement=True, generator=generator)
    return RandomSampler(range(len(labels)), replacement=False, generator=generator)


def training_loss(logits, labels, condition, weights):
    if condition not in CONDITIONS:
        raise ValueError('Unknown imbalance condition')
    losses = F.cross_entropy(logits, labels, reduction='none')
    if condition == 'natural_weighted_ce':
        # CE(weight=..., reduction='mean') divides by the selected weights,
        # which cancels class weighting for this project's batch_size=1.
        losses = losses*weights.to(logits.device)[labels]
    return losses.mean()


def model_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
