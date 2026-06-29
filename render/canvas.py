"""
Canvas — Cloth mesh renderer.

draw_frame() composites one frame in four steps:
  1. Trail fade    — in-place numpy multiply (pre-allocated buffer, no temp alloc)
  2. Cloth meshes  — horizontal strip polygons + vectorised colour computation
                     • One polygon per row-pair (21 draws vs 693 per cloth)
                     • Specular highlight band computed in batch with numpy
                     • Cloth 0: drawn normally; extra cloths: additive blend
  3. Centre sphere — optional pulsing glow (disabled by default)
  4. Bloom pass    — see effects.py (operates at 1/4 resolution)
"""

from __future__ import annotations

import colorsys
import math
from typing import List, Tuple

import numpy as np
import pygame

from physics.world import World

from .effects import apply_bloom


class Canvas:
    """Offscreen pygame Surface compositor."""

    def __init__(self, width: int, height: int, config: dict):
        self.width = width
        self.height = height
        self.config = config

        self.surface = pygame.Surface((width, height))
        bg = config.get("background_color", [3, 3, 8])
        self.surface.fill(tuple(bg))

        # Pre-allocated float32 buffer for the trail fade so we never allocate
        # a 50 MB float64 temporary array every frame.
        self._fade_f32 = np.empty((width, height, 3), dtype=np.float32)

        # Secondary surface for additive cloth layers
        self._aux = pygame.Surface((width, height))

    # ── Main entry ───────────────────────────────────────────────────────────

    def draw_frame(self, world: "World", features: dict) -> pygame.Surface:
        """Composite a full frame and return it as a pygame.Surface copy."""
        cfg = self.config

        # ── Step 1: Trail / motion-blur fade ────────────────────────────────
        trail_decay = cfg.get("trail_decay", 0.82)
        arr = pygame.surfarray.pixels3d(self.surface)  # direct lock, no copy
        np.multiply(arr, trail_decay, out=self._fade_f32)
        np.copyto(arr, self._fade_f32, casting="unsafe")  # float32 → uint8
        del arr  # release the surface lock

        # ── Step 2: Cloth meshes ─────────────────────────────────────────────
        for idx, cloth in enumerate(world.cloths):
            if idx == 0:
                _draw_cloth_strips(self.surface, cloth, features, world.global_hue)
            else:
                self._aux.fill((0, 0, 0))
                _draw_cloth_strips(
                    self._aux,
                    cloth,
                    features,
                    world.global_hue,
                    brightness_scale=0.65,
                )
                self.surface.blit(
                    self._aux,
                    (0, 0),
                    special_flags=pygame.BLEND_ADD,
                )

        # ── Step 3: Optional centre sphere ──────────────────────────────────
        sphere_base = cfg.get("sphere_base_radius", 0.0)
        if sphere_base > 0:
            radius = world.get_sphere_radius(features, sphere_base)
            _draw_sphere(
                self.surface,
                float(world.cx),
                float(world.cy),
                radius,
                features,
                world.global_hue,
            )

        # ── Step 4: Bloom pass ───────────────────────────────────────────────
        bloom_surf = apply_bloom(
            self.surface,
            threshold=cfg.get("bloom_threshold", 145),
            blur_radius=cfg.get("bloom_blur", 22),
        )
        self.surface.blit(bloom_surf, (0, 0), special_flags=pygame.BLEND_ADD)

        return self.surface.copy()


# ── Cloth rendering (horizontal strips) ──────────────────────────────────────


def _draw_cloth_strips(
    surface: pygame.Surface,
    cloth,
    features: dict,
    global_hue: float,
    brightness_scale: float = 1.0,
) -> None:
    """
    Render a cloth mesh as horizontal strip polygons.

    One polygon per row-pair (rows-1 draws per cloth) instead of one per quad.
    Colours are computed in a single vectorised numpy pass before the draw loop.

    Each strip polygon:
      top edge  = row r   particles, left → right
      bottom edge = row r+1 particles, right → left
    This captures the cloth's deformation faithfully.
    """
    pos = cloth.pos  # (rows, cols, 2) float64
    rows = cloth.rows
    cols = cloth.cols

    # Pre-convert all particle positions to Python int tuples (fast .tolist())
    int_pos = pos.astype(np.int32)  # (rows, cols, 2)

    # Batch-compute one colour per row-strip (vectorised, no Python loop)
    strip_rgb = _compute_strip_colors(cloth, features, global_hue, brightness_scale)
    # strip_rgb: list of (r, g, b) uint8 tuples, length = rows - 1

    for r in range(rows - 1):
        # Top edge: particles in row r, left to right
        top_xy = int_pos[r, :, :]  # (cols, 2)
        # Bottom edge: particles in row r+1, right to left (close the polygon)
        bot_xy = int_pos[r + 1, ::-1, :]  # (cols, 2), reversed

        # Convert to list-of-tuples expected by pygame.draw.polygon
        pts = np.concatenate([top_xy, bot_xy], axis=0)  # (2*cols, 2)
        poly_pts = list(map(tuple, pts.tolist()))

        pygame.draw.polygon(surface, strip_rgb[r], poly_pts)

    # ── Highlight thread lines ───────────────────────────────────────────────
    # Draw a few bright narrow lines along cloth rows to simulate silk threads
    # catching the light.  Very cheap (only a handful of polyline calls).
    hl_strength = min(1.0, features["bass"] * 0.5 + 0.15) * brightness_scale
    hl_lum = int(hl_strength * 180)
    hl_color = (hl_lum, hl_lum, min(255, int(hl_lum * 1.2)))  # slightly bluer

    step = max(2, rows // 6)  # draw ~6 highlight lines across the cloth height
    for r in range(0, rows, step):
        pts_r = list(map(tuple, int_pos[r, :, :].tolist()))
        if len(pts_r) >= 2:
            pygame.draw.lines(surface, hl_color, False, pts_r, 1)


# ── Vectorised colour computation ────────────────────────────────────────────


def _compute_strip_colors(
    cloth,
    features: dict,
    global_hue: float,
    brightness_scale: float = 1.0,
) -> List[Tuple[int, int, int]]:
    """
    Compute one RGB colour per horizontal strip (rows-1 values) in one numpy pass.

    The colour model:
      Hue:  cloth base + global drift + slight centre-column spread
      Sat:  rich silk colour, boosted by mids
      Val:  brightness envelope peaking at ~28% down the cloth height,
            plus a moving specular highlight band
    """
    rows_1 = cloth.rows - 1  # number of strips

    # Normalised row position for each strip midpoint: 0 = top, 1 = bottom
    t_row = np.linspace(0.0, 1.0, rows_1, dtype=np.float32)

    # ── Hue ──────────────────────────────────────────────────────────────────
    h = float((cloth.hue + global_hue + 0.05) % 1.0)  # single hue per cloth

    # ── Saturation ───────────────────────────────────────────────────────────
    s_base = float(min(1.0, 0.72 + features["mids"] * 0.28))

    # ── Value envelope ────────────────────────────────────────────────────────
    brightness_env = np.maximum(0.0, 1.0 - np.abs(t_row - 0.28) * 2.2)
    v = np.minimum(
        1.0,
        0.12 + brightness_env * 0.60 + features["bass"] * 0.20 + features["rms"] * 0.08,
    )

    # ── Specular highlight ────────────────────────────────────────────────────
    hl_y = 0.20 + 0.16 * math.sin(cloth.time * 1.4 + cloth.phase_offset)
    hl_dist = np.abs(t_row - hl_y)
    strength = np.maximum(0.0, (0.12 - hl_dist) / 0.12)
    v = np.minimum(1.0, v + strength * 0.55)
    s_arr = np.maximum(0.0, s_base - strength * 0.45)

    # ── Hue variation: slight tilt across rows for iridescent look ──────────
    h_arr = (h + t_row * 0.06) % 1.0

    # ── Vectorised HSV → RGB ──────────────────────────────────────────────────
    r_arr, g_arr, b_arr = _hsv_to_rgb_batch(h_arr, s_arr, v)

    if brightness_scale < 1.0:
        r_arr = (r_arr * brightness_scale).clip(0, 255).astype(np.uint8)
        g_arr = (g_arr * brightness_scale).clip(0, 255).astype(np.uint8)
        b_arr = (b_arr * brightness_scale).clip(0, 255).astype(np.uint8)

    return list(zip(r_arr.tolist(), g_arr.tolist(), b_arr.tolist()))


def _hsv_to_rgb_batch(
    h: np.ndarray,
    s: np.ndarray,
    v: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fully-vectorised HSV → RGB conversion.

    All inputs are float arrays with the same broadcast-compatible shape.
    Returns three uint8 arrays (R, G, B) in the range 0–255.
    """
    h6 = (h * 6.0) % 6.0
    i = h6.astype(np.int32) % 6
    f = h6 - i.astype(np.float32)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4], [v, q, p, p, t], default=v)
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4], [t, v, v, q, p], default=p)
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4], [p, p, t, v, v], default=q)

    return (
        (r * 255).astype(np.uint8),
        (g * 255).astype(np.uint8),
        (b * 255).astype(np.uint8),
    )


# ── Sphere (optional overlay) ─────────────────────────────────────────────────


def _draw_sphere(
    surface: pygame.Surface,
    cx: float,
    cy: float,
    radius: float,
    features: dict,
    global_hue: float,
) -> None:
    """Pulsing centre sphere with concentric glow rings."""
    bass = features["bass"]
    onset = features["onset"]

    if features["is_beat"] and onset > 0.6:
        sphere_color: Tuple[int, int, int] = (240, 245, 255)
    else:
        h = global_hue
        s = max(0.0, 0.30 + bass * 0.30)
        vv = min(1.0, 0.80 + bass * 0.20)
        r_, g_, b_ = colorsys.hsv_to_rgb(h, s, vv)
        sphere_color = (int(r_ * 255), int(g_ * 255), int(b_ * 255))

    for i in range(4, 0, -1):
        glow_r = int(radius * (1.0 + i * 0.40))
        alpha = max(0, int(35.0 * bass * (1.0 - i * 0.20) + 5))
        diam = glow_r * 2 + 1
        glow_s = pygame.Surface((diam, diam), pygame.SRCALPHA)
        pygame.draw.circle(glow_s, (*sphere_color, alpha), (glow_r, glow_r), glow_r)
        surface.blit(
            glow_s, (int(cx) - glow_r, int(cy) - glow_r), special_flags=pygame.BLEND_ADD
        )

    pygame.draw.circle(surface, sphere_color, (int(cx), int(cy)), max(2, int(radius)))
