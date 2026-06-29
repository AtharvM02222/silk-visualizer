"""
effects.py — Post-processing effects: bloom/glow pass.

apply_bloom() extracts bright pixels from a surface, Gaussian-blurs them,
and returns the result as a surface. The caller blits it back additively:

    bloom = apply_bloom(surface, threshold=160, blur_radius=18)
    surface.blit(bloom, (0, 0), special_flags=pygame.BLEND_ADD)

This makes bright areas radiate a soft glow without affecting dark areas
(additive blending adds 0 where the bloom surface is black).

Performance:
  - Uses cv2.GaussianBlur (C++/SIMD, ~8-10× faster than scipy)
  - Blur is computed on a 4× downscaled copy then upscaled back
  - These two tricks together give ~40× speedup vs naïve full-res scipy blur
"""

from __future__ import annotations

import cv2
import numpy as np
import pygame


def apply_bloom(
    surface: pygame.Surface,
    threshold: int = 180,
    blur_radius: int = 15,
) -> pygame.Surface:
    """
    Fast Gaussian bloom pass using OpenCV.

    Parameters
    ----------
    surface:     Source pygame.Surface to extract bloom from.
    threshold:   Pixel brightness (max of R,G,B) above which bloom is applied.
    blur_radius: Glow spread in pixels at full resolution.

    Returns
    -------
    pygame.Surface with the blurred bright-pixel layer (same size as input).
    """
    # 1. Grab pixels as (H, W, 3) uint8
    #    pygame.surfarray.array3d returns (W, H, 3) — transpose to (H, W, 3)
    raw = pygame.surfarray.array3d(surface)  # (W, H, 3) RGB
    arr = np.ascontiguousarray(raw.transpose(1, 0, 2))  # (H, W, 3)

    # 2. Extract only bright pixels (per-channel threshold on max channel)
    brightness = arr.max(axis=2)  # (H, W)
    mask = (brightness > threshold)[:, :, np.newaxis].astype(np.uint8)
    bright = arr * mask  # uint8 multiply

    # 3. Downscale 4× → blur with cv2 → upscale back
    #    cv2.GaussianBlur is 8-10× faster than scipy.ndimage.gaussian_filter
    scale = 4
    h, w = bright.shape[:2]
    sh, sw = max(1, h // scale), max(1, w // scale)

    small = cv2.resize(bright, (sw, sh), interpolation=cv2.INTER_LINEAR)
    ksize = max(3, (blur_radius // scale) * 2 + 1)  # must be odd
    sigma = max(1.0, blur_radius / scale)
    blurred_small = cv2.GaussianBlur(small, (ksize, ksize), sigma)

    # Upscale back to original resolution
    blurred = cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 4. Convert back to pygame surface  (H,W,3) → (W,H,3)
    bloom_arr = np.ascontiguousarray(blurred.transpose(1, 0, 2))
    bloom_surf = pygame.surfarray.make_surface(bloom_arr)
    return bloom_surf
