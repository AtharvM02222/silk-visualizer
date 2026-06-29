"""
effects.py — Post-processing effects: bloom/glow pass.

apply_bloom() extracts bright pixels, Gaussian-blurs them, and returns
the result as a surface.  The caller blits it back additively:

    bloom = apply_bloom(surface, threshold=145, blur_radius=22)
    surface.blit(bloom, (0, 0), special_flags=pygame.BLEND_ADD)

Performance strategy:
  - Downscale the *input surface* first with pygame.transform.scale
    (hardware-assisted, much faster than numpy copying 6 MB of pixels)
  - All numpy work is done on the small (¼-res) array
  - Upscale the result back to full size with pygame.transform.scale
  - Only one ascontiguousarray call instead of two
  - cv2.GaussianBlur runs on ~1/16th the pixel count
"""

from __future__ import annotations

import cv2
import numpy as np
import pygame


def apply_bloom(
    surface: pygame.Surface,
    threshold: int = 145,
    blur_radius: int = 22,
) -> pygame.Surface:
    """
    Fast Gaussian bloom pass.

    Parameters
    ----------
    surface:      Source pygame.Surface.
    threshold:    Pixel brightness (max of R,G,B) above which bloom applies.
    blur_radius:  Glow spread in pixels at *full* resolution.

    Returns
    -------
    pygame.Surface — blurred bright-pixel layer at full input size.
    """
    w, h = surface.get_size()

    # ── 1. Downscale input ────────────────────────────────────────────────────
    #   pygame.transform.scale uses hardware/SDL2 scaling — much faster than
    #   a full numpy copy of the 1920×1080 surface.
    scale = 4
    sw = max(1, w // scale)
    sh = max(1, h // scale)
    small_surf = pygame.transform.scale(surface, (sw, sh))

    # ── 2. Read small array ────────────────────────────────────────────────────
    #   pygame returns (W, H, 3).  OpenCV expects (H, W, 3).
    raw = pygame.surfarray.array3d(small_surf)  # (sw, sh, 3)
    arr = np.ascontiguousarray(raw.transpose(1, 0, 2))  # (sh, sw, 3)

    # ── 3. Threshold bright pixels ────────────────────────────────────────────
    brightness = arr.max(axis=2)  # (sh, sw)
    mask = (brightness > threshold)[:, :, np.newaxis].astype(np.uint8)
    bright = arr * mask  # zero out dark pixels

    # ── 4. Gaussian blur on small image ───────────────────────────────────────
    ksize = max(3, (blur_radius // scale) * 2 + 1)
    if ksize % 2 == 0:
        ksize += 1
    sigma = max(1.0, blur_radius / scale)
    blurred = cv2.GaussianBlur(bright, (ksize, ksize), sigma)

    # ── 5. Make small bloom surface ───────────────────────────────────────────
    #   (sh, sw, 3) → (sw, sh, 3) for pygame
    bloom_raw = np.ascontiguousarray(blurred.transpose(1, 0, 2))
    bloom_small = pygame.surfarray.make_surface(bloom_raw)

    # ── 6. Upscale to full resolution ─────────────────────────────────────────
    return pygame.transform.scale(bloom_small, (w, h))
