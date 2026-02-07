"""
Patch-level metrics computed from WSI patch images (RGB).
Used to augment feature H5 files without re-running feature extraction.
"""
import numpy as np
from PIL import Image
import cv2


def _pil_to_rgb(pil_img):
    """PIL Image (RGB or other) to numpy RGB uint8."""
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    return np.array(pil_img)


def stain_saturation(pil_img, downsample=1):
    """
    Mean saturation in HSV space (0~1). Higher = more vivid stain color.
    Args:
        pil_img: PIL Image (RGB)
        downsample: optional downscale factor for speed
    Returns:
        float
    """
    img = _pil_to_rgb(pil_img)
    if downsample > 1:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // downsample, h // downsample))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    return float(np.mean(s))


def color_entropy(pil_img, bins=32, downsample=1):
    """
    Entropy of the RGB color distribution (scaled to 0~1 range for typical images).
    Higher = more diverse colors in the patch.
    Args:
        pil_img: PIL Image (RGB)
        bins: number of bins per channel for 3D histogram
        downsample: optional downscale factor for speed
    Returns:
        float (entropy in bits, then normalized by log2(bins^3) so max ~ 1)
    """
    img = _pil_to_rgb(pil_img)
    if downsample > 1:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // downsample, h // downsample))
    hist, _ = np.histogramdd(
        img.reshape(-1, 3),
        bins=bins,
        range=((0, 256), (0, 256), (0, 256))
    )
    hist = hist.ravel()
    hist = hist[hist > 0]
    p = hist / hist.sum()
    entropy_bits = -np.sum(p * np.log2(p))
    max_entropy = np.log2(bins ** 3)
    return float(entropy_bits / max_entropy) if max_entropy > 0 else 0.0


def contrast(pil_img, downsample=1):
    """
    Contrast as standard deviation of luminance (grayscale). Higher = more contrast.
    Args:
        pil_img: PIL Image (RGB)
        downsample: optional downscale factor for speed
    Returns:
        float (normalized 0~1 range for typical histology; raw std can be scaled)
    """
    img = _pil_to_rgb(pil_img)
    if downsample > 1:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // downsample, h // downsample))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    std = float(np.std(gray))
    # optional: clip/normalize so values sit in a similar range as other metrics
    return min(1.0, std / 128.0) if std > 0 else 0.0


def compute_patch_metrics(pil_img, downsample=2, include_laplacian=False):
    """
    Compute stain_saturation, color_entropy, contrast (and optionally Laplacian blur) from one patch.
    Args:
        pil_img: PIL Image (RGB)
        downsample: used for saturation/entropy/contrast (and blur if include_laplacian)
        include_laplacian: if True, also compute blur_score_laplacian (requires blur_utils)
    Returns:
        dict with keys 'stain_saturation', 'color_entropy', 'contrast', and optionally 'laplacian'
    """
    out = {
        'stain_saturation': stain_saturation(pil_img, downsample=downsample),
        'color_entropy': color_entropy(pil_img, downsample=downsample),
        'contrast': contrast(pil_img, downsample=downsample),
    }
    if include_laplacian:
        from utils.blur_utils import blur_score_laplacian
        out['laplacian'] = blur_score_laplacian(pil_img, downsample=downsample)
    return out
