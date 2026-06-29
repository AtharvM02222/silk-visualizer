"""
Strand — A chain of Particle objects forming one silk ribbon.

The root particle (index 0) is pinned at the canvas centre.
Audio forces are applied to the free particles every frame, propagating
cloth-like waves through the spring chain.
"""

from __future__ import annotations

import colorsys
import math
import random
from typing import List, Tuple

from .particle import Particle


class Strand:
    """One silk strand: a linked chain of particles radiating from the origin."""

    def __init__(
        self,
        origin: Tuple[float, float],
        angle: float,  # Direction this strand points (radians)
        n_particles: int = 20,
        segment_length: float = 8.0,  # Rest length of each spring (pixels)
        stiffness: float = 0.85,  # Spring stiffness 0–1
        damping: float = 0.975,  # Per-particle Verlet damping
        hue: float = 0.55,  # Base colour hue 0–1
        config: dict | None = None,
    ):
        self.origin = origin
        self.angle = angle
        self.n_particles = n_particles
        self.segment_length = segment_length
        self.stiffness = stiffness
        self.hue = hue
        self.config = config or {}

        # Build particle chain from origin outward along `angle`
        self.particles: List[Particle] = []
        ox, oy = origin
        for i in range(n_particles):
            px = ox + math.cos(angle) * segment_length * i
            py = oy + math.sin(angle) * segment_length * i
            p = Particle(
                x=px,
                y=py,
                prev_x=px,
                prev_y=py,
                pinned=(i == 0),
                damping=damping,
            )
            self.particles.append(p)

    # ── Audio forces ─────────────────────────────────────────────────────────

    def apply_audio_force(self, features: dict, dt: float) -> None:
        """
        Translate per-frame audio features into physics forces on the particles.

        Forces applied:
        - Bass  → large perpendicular wind force on the tip (~last 40%)
        - Mids  → adds to wind magnitude
        - Highs → random Gaussian jitter on every free particle
        - Beat  → radial burst outward from centre (stronger at tip)
        """
        bass = features["bass"]
        mids = features["mids"]
        highs = features["highs"]
        onset = features["onset"]
        is_beat = features["is_beat"]

        cfg = self.config
        bass_mult = cfg.get("bass_force_mult", 900.0)
        mids_mult = cfg.get("mids_force_mult", 250.0)
        burst_mult = cfg.get("beat_burst_mult", 1400.0)
        jitter_std = cfg.get("jitter_mult", 55.0)
        min_force = cfg.get("min_force_baseline", 50.0)

        # ── Perpendicular wind (makes the strand wave sideways) ──────────────
        perp_angle = self.angle + math.pi / 2.0
        wind_x = math.cos(perp_angle)
        wind_y = math.sin(perp_angle)

        force_mag = bass * bass_mult + mids * mids_mult
        force_mag = max(force_mag, min_force)

        # ── Beat burst: radial push outward along the strand's axis ──────────
        if is_beat:
            burst = onset * burst_mult
            for i in range(1, self.n_particles):
                weight = i / self.n_particles  # tip gets most force
                self.particles[i].apply_force(
                    math.cos(self.angle) * burst * weight,
                    math.sin(self.angle) * burst * weight,
                )

        # ── Continuous wind on the tip portion (last 40% of particles) ───────
        start = int(self.n_particles * 0.6)
        tip_count = max(1, self.n_particles - start)
        for i in range(start, self.n_particles):
            weight = (i - start) / tip_count  # tapers 0→1 toward tip
            self.particles[i].apply_force(
                wind_x * force_mag * weight,
                wind_y * force_mag * weight,
            )

        # ── Highs: random jitter on all free particles (sparkle / texture) ───
        jitter = highs * jitter_std + 1.0  # +1 keeps motion during silence
        for p in self.particles[1:]:
            p.apply_force(
                random.gauss(0.0, jitter),
                random.gauss(0.0, jitter),
            )

    # ── Update ───────────────────────────────────────────────────────────────

    def update(
        self,
        dt: float,
        constraint_iterations: int = 6,
        canvas_w: int = 1920,
        canvas_h: int = 1080,
    ) -> None:
        """
        1. Verlet-integrate all particle positions.
        2. Iteratively enforce spring constraints (more passes = stiffer cloth).
        3. Re-pin root to origin (prevents drift).
        4. Clamp free particles to canvas bounds.
        """
        # Step 1: integrate
        for p in self.particles:
            p.update(dt)

        # Step 2 + 3: constraint solving passes
        margin = 30
        for _ in range(constraint_iterations):
            # Forward sweep: root → tip
            for i in range(len(self.particles) - 1):
                self.particles[i].constrain_distance(
                    self.particles[i + 1],
                    rest_length=self.segment_length,
                    stiffness=self.stiffness,
                )
            # Always re-pin root to its anchor position
            self.particles[0].x = self.origin[0]
            self.particles[0].y = self.origin[1]

        # Step 4: keep particles on screen
        for p in self.particles[1:]:
            p.clamp(margin, canvas_w - margin, margin, canvas_h - margin)

    # ── Accessors ────────────────────────────────────────────────────────────

    def get_points(self) -> List[Tuple[float, float]]:
        """Return (x, y) for every particle in the chain."""
        return [(p.x, p.y) for p in self.particles]

    def get_color(self, features: dict, global_hue: float) -> Tuple[int, int, int]:
        """
        Compute HSV → RGB strand colour for this frame.

        Hue:        strand's own hue + global drift + small mids nudge
        Saturation: base 0.65, boosted by mids
        Value:      base 0.50, boosted by RMS and bass
        """
        palette = self.config.get("palette", "cool")

        if palette == "mono":
            # All strands share the same slowly-drifting hue
            h = global_hue % 1.0
        else:
            h = (self.hue + global_hue + features["mids"] * 0.05) % 1.0
            h = _apply_palette(h, palette)

        s = min(1.0, 0.65 + features["mids"] * 0.35)
        if palette == "neon":
            s = min(1.0, s + 0.15)

        v = min(1.0, 0.50 + features["rms"] * 0.30 + features["bass"] * 0.20)

        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))


# ── Palette helpers ───────────────────────────────────────────────────────────


def _apply_palette(h: float, palette: str) -> float:
    """Map normalised hue 0–1 into the palette's hue band."""
    if palette == "warm":
        # Reds, oranges, yellows  →  hue 0.00 – 0.15
        return (h % 1.0) * 0.15
    elif palette == "cool":
        # Blues, cyans, purples   →  hue 0.55 – 0.75
        return 0.55 + (h % 1.0) * 0.20
    # "neon" and unknown: full spectrum, no mapping
    return h
