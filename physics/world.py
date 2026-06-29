"""
World — Container and coordinator for all cloth meshes.

Manages:
- Creating N cloth meshes spanning the full canvas
- Stepping physics each frame with per-frame audio data
- Global hue drift across all cloths
- Sphere radius calculation (kept for optional overlay use)
"""

from __future__ import annotations

import math
from typing import List

from .cloth import ClothMesh


class World:
    """Physics world: holds cloth meshes and drives them with audio features."""

    def __init__(self, config: dict, canvas_w: int, canvas_h: int):
        self.cx = canvas_w // 2
        self.cy = canvas_h // 2
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.config = config

        self.cloths: List[ClothMesh] = []
        self.global_hue: float = 0.0
        self.hue_drift_speed: float = config.get("hue_drift_speed", 0.00015)

        self._build_cloths(config)

    # ── Initialisation ───────────────────────────────────────────────────────

    def _build_cloths(self, config: dict) -> None:
        """Create N cloth meshes with spread hues and staggered phases."""
        n = config.get("n_cloths", 2)
        rows = config.get("cloth_rows", 22)
        cols = config.get("cloth_cols", 34)
        stiffness = config.get("cloth_stiffness", 0.70)
        damping = config.get("cloth_damping", 0.972)

        # Hue sequence: blue → violet → teal-blue → magenta
        hues = [0.63, 0.77, 0.57, 0.85]

        for i in range(min(n, len(hues))):
            # Each cloth gets a different phase so their waves are out of sync,
            # producing complex layered interference patterns.
            phase = i * (2.0 * math.pi / max(n, 1))

            cloth = ClothMesh(
                x=0.0,
                y=0.0,
                width=float(self.canvas_w),
                height=float(self.canvas_h),
                rows=rows,
                cols=cols,
                stiffness=stiffness,
                damping=damping,
                hue=hues[i],
                phase_offset=phase,
                config=config,
            )
            self.cloths.append(cloth)

    # ── Per-frame update ─────────────────────────────────────────────────────

    def step(self, features: dict, dt: float) -> None:
        """Advance every cloth mesh by one frame."""
        self.global_hue = (self.global_hue + self.hue_drift_speed) % 1.0
        constraint_iters = self.config.get("constraint_iterations", 5)

        for cloth in self.cloths:
            cloth.apply_audio_force(features, dt)
            cloth.update(
                dt,
                constraint_iters,
                self.canvas_w,
                self.canvas_h,
            )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def get_sphere_radius(self, features: dict, base_radius: float) -> float:
        """Centre sphere radius pulsing with bass and beat onset."""
        pulse_scale = self.config.get("sphere_pulse_scale", 0.45)
        pulse = features["bass"] * pulse_scale + features["onset"] * 0.20
        return base_radius * (1.0 + pulse)
