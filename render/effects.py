"""
effects.py — Post-processing effects: bloom/glow pass.

apply_bloom() extracts bright pixels from a surface, Gaussian-blurs them,
and returns the result as a surface. The caller blits it back additively:

    bloom = apply_bloom(surface, threshold=160, blur_radius=18)
    surface.blit(bloom, (0, 0), special_flags=pygame.BLEND_ADD)

This makes bright areas radiate a soft glow without affecting dark areas
(additive blending adds 0 where the bloom surface is black).

Performance note: the blur is computed on a 4× downscaled copy and then
scaled back up. This gives ≈16× speed improvement with barely visible
quality loss on the bloom effect.
"""

from __future__ import annotations

import numpy as np
import pygame
from scipy.ndimage import gaussian_filter


def apply_bloom(
    surface: pygame.Surface,
    threshold: int = 180,
    blur_radius: int = 15,
) -> pygame.Surface:
    """
    Gaussian bloom pass.

    Parameters
    ----------
    surface:     Source pygame.Surface to extract bloom from.
    threshold:   Pixel brightness (max of R,G,B) above which bloom is applied.
    blur_radius: Gaussian sigma in pixels (applied at full resolution).

    Returns
    -------
    pygame.Surface with the blurred bright-pixel layer (same size as input).
    """
    # 1. Grab pixel array as (H, W, 3) float32
    #    pygame.surfarray.array3d returns (W, H, 3) — transpose to (H, W, 3)
    raw = pygame.surfarray.array3d(surface)  # (W, H, 3)
    arr = raw.transpose(1, 0, 2).astype(np.float32)  # (H, W, 3)

    # 2. Extract only bright pixels (above threshold)
    brightness = arr.max(axis=2)  # (H, W)
    mask = (brightness > threshold)[:, :, np.newaxis]
    bright = arr * mask  # zero-out dim pixels

    # 3. Downscale 4× for fast blur, then blur, then upscale back
    scale = 4
    h, w = bright.shape[:2]
    small = bright[::scale, ::scale, :]  # crude 4× downscale
    sigma = max(1.0, blur_radius / scale)
    blurred_small = gaussian_filter(small, sigma=[sigma, sigma, 0])

    # Upscale with np.repeat (nearest-neighbour, fast)
    blurred = np.repeat(np.repeat(blurred_small, scale, axis=0), scale, axis=1)
    # Trim to original dimensions (repeat may overshoot by scale-1 pixels)
    blurred = blurred[:h, :w, :]

    # 4. Convert back to uint8 pygame surface
    blurred_u8 = np.clip(blurred, 0, 255).astype(np.uint8)
    bloom_arr = blurred_u8.transpose(1, 0, 2)  # back to (W, H, 3)
    bloom_surf = pygame.surfarray.make_surface(bloom_arr)
    return bloom_surf
