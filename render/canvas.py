"""
Canvas — Cloth mesh renderer.

draw_frame() composites one frame in four steps:
  1. Trail fade    — numpy-multiply existing pixels by trail_decay (motion blur)
  2. Cloth meshes  — filled quads per grid cell with silk sheen colouring
                     Cloth 0 is drawn normally; additional cloths are
                     blended additively for a layered translucent-fabric look
  3. Centre sphere — optional pulsing glow (disabled by default)
  4. Bloom pass    — extract bright pixels, blur, additively blit back

Returns a copy of the internal pygame.Surface each frame.
"""

from __future__ import annotations

import colorsys
import math
from typing import Tuple

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

        # Secondary surface for additive cloth layers
        self._aux = pygame.Surface((width, height))

    # ── Main entry ───────────────────────────────────────────────────────────

    def draw_frame(self, world: "World", features: dict) -> pygame.Surface:
        """Composite a full frame and return it as a pygame.Surface copy."""
        cfg = self.config

        # ── Step 1: Trail / motion-blur fade ────────────────────────────────
        trail_decay = cfg.get("trail_decay", 0.82)
        arr = pygame.surfarray.pixels3d(self.surface)
        arr[:] = (arr * trail_decay).astype(np.uint8)
        del arr  # release the surface lock

        # ── Step 2: Cloth meshes ─────────────────────────────────────────────
        for idx, cloth in enumerate(world.cloths):
            if idx == 0:
                # Primary cloth: draw directly onto the main surface
                _draw_cloth(self.surface, cloth, features, world.global_hue)
            else:
                # Additional cloths: draw onto aux surface and blit additively.
                # This creates a glowing layered-fabric effect without full occlusion.
                self._aux.fill((0, 0, 0))
                _draw_cloth(
                    self._aux,
                    cloth,
                    features,
                    world.global_hue,
                    brightness_scale=0.65,  # softer so additive blend doesn't saturate
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


# ── Drawing helpers ───────────────────────────────────────────────────────────


def _draw_cloth(
    surface: pygame.Surface,
    cloth,
    features: dict,
    global_hue: float,
    brightness_scale: float = 1.0,
) -> None:
    """
    Draw a cloth mesh as a grid of filled quads.

    Each cell (r, c) → (r+1, c+1) is rendered as a filled quadrilateral
    whose colour encodes depth, audio energy, and a moving silk highlight.

    brightness_scale:  overall brightness multiplier (use < 1 for additive layers).
    """
    pos = cloth.pos  # numpy (rows, cols, 2)
    rows = cloth.rows
    cols = cloth.cols

    for r in range(rows - 1):
        for c in range(cols - 1):
            p00 = (int(pos[r, c, 0]), int(pos[r, c, 1]))
            p01 = (int(pos[r, c + 1, 0]), int(pos[r, c + 1, 1]))
            p11 = (int(pos[r + 1, c + 1, 0]), int(pos[r + 1, c + 1, 1]))
            p10 = (int(pos[r + 1, c, 0]), int(pos[r + 1, c, 1]))

            cr, cg, cb = cloth.get_quad_color(r, c, features, global_hue)
            if brightness_scale < 1.0:
                cr = int(cr * brightness_scale)
                cg = int(cg * brightness_scale)
                cb = int(cb * brightness_scale)

            pygame.draw.polygon(
                surface,
                (cr, cg, cb),
                [p00, p01, p11, p10],
            )


def _draw_sphere(
    surface: pygame.Surface,
    cx: float,
    cy: float,
    radius: float,
    features: dict,
    global_hue: float,
) -> None:
    """
    Draw the optional pulsing centre sphere with concentric glow rings.

    On strong beats the sphere briefly flashes near-white.
    """
    bass = features["bass"]
    onset = features["onset"]

    if features["is_beat"] and onset > 0.6:
        sphere_color: Tuple[int, int, int] = (240, 245, 255)
    else:
        h = global_hue
        s = max(0.0, 0.30 + bass * 0.30)
        v = min(1.0, 0.80 + bass * 0.20)
        r_, g_, b_ = colorsys.hsv_to_rgb(h, s, v)
        sphere_color = (int(r_ * 255), int(g_ * 255), int(b_ * 255))

    # Outer glow rings
    for i in range(4, 0, -1):
        glow_r = int(radius * (1.0 + i * 0.40))
        alpha = max(0, int(35.0 * bass * (1.0 - i * 0.20) + 5))
        diam = glow_r * 2 + 1
        glow_s = pygame.Surface((diam, diam), pygame.SRCALPHA)
        pygame.draw.circle(glow_s, (*sphere_color, alpha), (glow_r, glow_r), glow_r)
        surface.blit(
            glow_s,
            (int(cx) - glow_r, int(cy) - glow_r),
            special_flags=pygame.BLEND_ADD,
        )

    # Solid core
    pygame.draw.circle(surface, sphere_color, (int(cx), int(cy)), max(2, int(radius)))
