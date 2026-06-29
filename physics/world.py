"""
World — Container and coordinator for all silk strands.

Manages:
- Creating N strands evenly spaced around the canvas centre
- Stepping physics each frame with per-frame audio data
- Global hue drift across all strands
- Sphere radius calculation based on bass / onset
"""

from __future__ import annotations

import math
from typing import List

from .strand import Strand


class World:
    """Physics world: holds all strands and drives them with audio features."""

    def __init__(self, config: dict, canvas_w: int, canvas_h: int):
        self.cx = canvas_w // 2
        self.cy = canvas_h // 2
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.config = config

        self.strands: List[Strand] = []
        self.global_hue: float = 0.0
        self.hue_drift_speed: float = config.get("hue_drift_speed", 0.0002)

        self._build_strands(config)

    # ── Initialisation ───────────────────────────────────────────────────────

    def _build_strands(self, config: dict) -> None:
        """Create N strands at evenly spaced angles with spread hues."""
        n_strands = config.get("n_strands", 10)
        n_particles = config.get("n_particles", 24)
        segment_length = config.get("segment_length", 9.0)
        stiffness = config.get("stiffness", 0.82)
        damping = config.get("damping", 0.975)

        for i in range(n_strands):
            angle = i * (2.0 * math.pi / n_strands)
            hue = i / n_strands  # spread hues evenly around colour wheel
            strand = Strand(
                origin=(float(self.cx), float(self.cy)),
                angle=angle,
                n_particles=n_particles,
                segment_length=segment_length,
                stiffness=stiffness,
                damping=damping,
                hue=hue,
                config=config,
            )
            self.strands.append(strand)

    # ── Per-frame update ─────────────────────────────────────────────────────

    def step(self, features: dict, dt: float) -> None:
        """Advance the world by one frame.

        - Drift global hue
        - Apply audio forces and integrate each strand
        """
        self.global_hue = (self.global_hue + self.hue_drift_speed) % 1.0
        constraint_iters = self.config.get("constraint_iterations", 6)

        for strand in self.strands:
            strand.apply_audio_force(features, dt)
            strand.update(
                dt,
                constraint_iterations=constraint_iters,
                canvas_w=self.canvas_w,
                canvas_h=self.canvas_h,
            )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def get_sphere_radius(self, features: dict, base_radius: float) -> float:
        """Centre sphere radius pulsing with bass and beat onset."""
        pulse_scale = self.config.get("sphere_pulse_scale", 0.45)
        pulse = features["bass"] * pulse_scale + features["onset"] * 0.20
        return base_radius * (1.0 + pulse)
