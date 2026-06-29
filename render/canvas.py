"""
Canvas — All drawing logic.

draw_frame() produces one composited frame in four steps:
  1. Trail fade   — numpy-multiply existing pixels by trail_decay (motion blur)
  2. Silk strands — Catmull-Rom smoothed curves with tapering width
  3. Centre sphere — pulsing glow rings + solid core
  4. Bloom pass  — extract bright pixels, blur, additively blit back

Returns a copy of the internal pygame.Surface.
"""

from __future__ import annotations

import colorsys
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

        # Offscreen surface (no display required)
        self.surface = pygame.Surface((width, height))
        bg = config.get("background_color", [5, 5, 10])
        self.surface.fill(tuple(bg))

    # ── Main entry ───────────────────────────────────────────────────────────

    def draw_frame(self, world: "World", features: dict) -> pygame.Surface:
        """Composite a full frame and return it as a pygame.Surface copy."""
        cfg = self.config

        # ── Step 1: Trail / motion-blur fade ────────────────────────────────
        #   Multiply every pixel by trail_decay each frame.
        #   This leaves fading echoes of past strands → silk trail effect.
        trail_decay = cfg.get("trail_decay", 0.88)
        arr = pygame.surfarray.pixels3d(self.surface)
        arr[:] = (arr * trail_decay).astype(np.uint8)
        del arr  # release pygame surface lock

        # ── Step 2: Draw silk strands ────────────────────────────────────────
        base_width = cfg.get("strand_base_width", 3)
        for strand in world.strands:
            points = strand.get_points()
            color = strand.get_color(features, world.global_hue)
            # Smooth the raw particle positions with Catmull-Rom
            if len(points) >= 4:
                smooth = catmull_rom_points(points, subdivisions=4)
            else:
                smooth = list(points)
            _draw_strand(self.surface, smooth, color, base_width)

        # ── Step 3: Centre sphere ────────────────────────────────────────────
        sphere_radius = world.get_sphere_radius(
            features, cfg.get("sphere_base_radius", 28.0)
        )
        _draw_sphere(
            self.surface,
            float(world.cx),
            float(world.cy),
            sphere_radius,
            features,
            world.global_hue,
        )

        # ── Step 4: Bloom pass ───────────────────────────────────────────────
        bloom_surf = apply_bloom(
            self.surface,
            threshold=cfg.get("bloom_threshold", 160),
            blur_radius=cfg.get("bloom_blur", 18),
        )
        self.surface.blit(bloom_surf, (0, 0), special_flags=pygame.BLEND_ADD)

        return self.surface.copy()


# ── Drawing helpers (module-level for clarity) ────────────────────────────────


def _draw_strand(
    surface: pygame.Surface,
    points: List[Tuple[float, float]],
    color: Tuple[int, int, int],
    base_width: int = 3,
) -> None:
    """
    Draw a strand as a series of line segments with tapered width.

    Width at segment i: base_width → 1 px, interpolated by position along chain.
    """
    n = len(points)
    if n < 2:
        return
    r, g, b = color
    for i in range(n - 1):
        t = i / max(n - 1, 1)  # 0.0 at root, 1.0 at tip
        width = max(1, int(base_width * (1.0 - t * 0.70)))
        pygame.draw.line(
            surface,
            (r, g, b),
            (int(points[i][0]), int(points[i][1])),
            (int(points[i + 1][0]), int(points[i + 1][1])),
            width,
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
    Draw the pulsing centre sphere with concentric glow rings.

    On strong beats the sphere briefly flashes near-white.
    """
    bass = features["bass"]
    onset = features["onset"]

    # Sphere colour: near-white flash on strong beats, warm glow otherwise
    if features["is_beat"] and onset > 0.6:
        sphere_color: Tuple[int, int, int] = (240, 245, 255)
    else:
        h = global_hue
        s = max(0.0, 0.30 + bass * 0.30)
        v = min(1.0, 0.80 + bass * 0.20)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        sphere_color = (int(r * 255), int(g * 255), int(b * 255))

    # Outer glow rings (additive blend — bright areas accumulate)
    for i in range(4, 0, -1):
        glow_r = int(radius * (1.0 + i * 0.40))
        alpha = max(0, int(35.0 * bass * (1.0 - i * 0.20) + 5))
        diam = glow_r * 2 + 1
        glow_s = pygame.Surface((diam, diam), pygame.SRCALPHA)
        pygame.draw.circle(
            glow_s,
            (*sphere_color, alpha),
            (glow_r, glow_r),
            glow_r,
        )
        surface.blit(
            glow_s,
            (int(cx) - glow_r, int(cy) - glow_r),
            special_flags=pygame.BLEND_ADD,
        )

    # Solid core
    r_int = max(2, int(radius))
    pygame.draw.circle(surface, sphere_color, (int(cx), int(cy)), r_int)


def catmull_rom_points(
    points: List[Tuple[float, float]],
    subdivisions: int = 4,
) -> List[Tuple[float, float]]:
    """
    Catmull-Rom spline interpolation.

    Takes a list of control points (the raw particle positions) and returns a
    denser list of smooth interpolated points. Makes strands look like flowing
    silk instead of a jagged polyline.

    Requires at least 4 input points (i-1, i, i+1, i+2 for each segment).
    """
    result: List[Tuple[float, float]] = []
    n = len(points)

    # Always include the root particle
    result.append(points[0])

    for i in range(1, n - 2):
        p0 = points[i - 1]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[i + 2]

        for j in range(subdivisions):
            t = j / subdivisions
            t2 = t * t
            t3 = t2 * t

            # Catmull-Rom formula (alpha = 0.5 centripetal is omitted here;
            # standard uniform CR is fine for this use case)
            x = 0.5 * (
                (2.0 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2.0 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3
            )
            result.append((x, y))

    # Append the last control point so the strand reaches its tip
    if n >= 2:
        result.append(points[-1])

    return result
