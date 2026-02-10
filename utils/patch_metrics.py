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


def tenengrad(pil_img, downsample=1, ksize=3):
    """
    Tenengrad: sum of squared Sobel gradient magnitudes. Higher = sharper.
    Args:
        pil_img: PIL Image (RGB)
        downsample: optional downscale factor for speed
        ksize: Sobel kernel size (1, 3, 5 or 7)
    Returns:
        float (raw sum of gx^2 + gy^2; scale may vary by image size)
    """
    img = _pil_to_rgb(pil_img)
    if downsample > 1:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // downsample, h // downsample))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag_sq = gx * gx + gy * gy
    return float(np.sum(mag_sq))


def vgm(pil_img, downsample=1, ksize=3):
    """
    Variance of Gradient Magnitude (VGM). Higher = sharper.
    Args:
        pil_img: PIL Image (RGB)
        downsample: optional downscale factor for speed
        ksize: Sobel kernel size
    Returns:
        float (variance of gradient magnitude)
    """
    img = _pil_to_rgb(pil_img)
    if downsample > 1:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // downsample, h // downsample))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.var(mag))


def wavelet_sharpness(pil_img, downsample=1, wavelet='haar'):
    """
    Wavelet-based sharpness: energy in high-frequency subbands (HH, HL, LH) after 2D DWT.
    Higher = sharper. Requires PyWavelets (pip install PyWavelets).
    Args:
        pil_img: PIL Image (RGB)
        downsample: optional downscale factor for speed
        wavelet: wavelet name for pywt.dwt2 (default 'haar')
    Returns:
        float (sum of squared coefficients in high subbands), or 0.0 if pywt not available
    """
    img = _pil_to_rgb(pil_img)
    if downsample > 1:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // downsample, h // downsample))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    try:
        import pywt
        cA, (cH, cV, cD) = pywt.dwt2(gray, wavelet)
        # high-frequency energy: HH (diagonal), HL (horizontal), LH (vertical)
        energy = float(np.sum(cH ** 2) + np.sum(cV ** 2) + np.sum(cD ** 2))
        return energy
    except ImportError:
        return 0.0


def compute_patch_metrics(pil_img, downsample=2, include_laplacian=False, include_tenengrad_vgm_wavelet=True):
    """
    Compute stain_saturation, color_entropy, contrast (and optionally Laplacian, Tenengrad, VGM, Wavelet) from one patch.
    Args:
        pil_img: PIL Image (RGB)
        downsample: used for saturation/entropy/contrast and sharpness metrics
        include_laplacian: if True, also compute blur_score_laplacian (requires blur_utils)
        include_tenengrad_vgm_wavelet: if True, also compute tenengrad, vgm, wavelet_sharpness (wavelet requires PyWavelets)
    Returns:
        dict with keys 'stain_saturation', 'color_entropy', 'contrast', and optionally 'laplacian', 'tenengrad', 'vgm', 'wavelet'
    """
    out = {
        'stain_saturation': stain_saturation(pil_img, downsample=downsample),
        'color_entropy': color_entropy(pil_img, downsample=downsample),
        'contrast': contrast(pil_img, downsample=downsample),
    }
    if include_laplacian:
        from utils.blur_utils import blur_score_laplacian
        out['laplacian'] = blur_score_laplacian(pil_img, downsample=downsample)
    if include_tenengrad_vgm_wavelet:
        out['tenengrad'] = tenengrad(pil_img, downsample=downsample)
        out['vgm'] = vgm(pil_img, downsample=downsample)
        out['wavelet'] = wavelet_sharpness(pil_img, downsample=downsample)
    return out
