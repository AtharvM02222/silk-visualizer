"""
Particle — A single 2D point mass that moves via Verlet integration.

Verlet integration stores position history instead of explicit velocity:
    velocity ≈ (current_pos - prev_pos)
This is naturally stable for spring constraints and cloth simulation.
See: Jakobsen 2001 "Advanced Character Physics"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Particle:
    x: float
    y: float
    prev_x: float
    prev_y: float
    pinned: bool = False  # Pinned particles never move (e.g. strand root)
    mass: float = 1.0
    damping: float = 0.975  # Velocity damping per frame (0.95–0.99)

    # Accumulated force for this frame — reset to 0 after each update()
    acc_x: float = field(default=0.0, repr=False)
    acc_y: float = field(default=0.0, repr=False)

    # ── Force accumulation ───────────────────────────────────────────────

    def apply_force(self, fx: float, fy: float) -> None:
        """Add (fx, fy) to this frame's force accumulator. Ignored if pinned."""
        if self.pinned:
            return
        self.acc_x += fx / self.mass
        self.acc_y += fy / self.mass

    # ── Integration ─────────────────────────────────────────────────────────

    def update(self, dt: float) -> None:
        """
        Verlet integration step.

        new_pos = pos + (pos - prev_pos) * damping + acc * dt²
        """
        if self.pinned:
            return

        vx = (self.x - self.prev_x) * self.damping
        vy = (self.y - self.prev_y) * self.damping

        new_x = self.x + vx + self.acc_x * dt * dt
        new_y = self.y + vy + self.acc_y * dt * dt

        self.prev_x, self.prev_y = self.x, self.y
        self.x, self.y = new_x, new_y

        # Reset force accumulator
        self.acc_x = self.acc_y = 0.0

    # ── Spring constraint ────────────────────────────────────────────────────

    def constrain_distance(
        self,
        other: "Particle",
        rest_length: float,
        stiffness: float,
    ) -> None:
        """
        XPBD-style iterative distance constraint.

        Pushes self and other apart/together until their distance
        equals rest_length, scaled by stiffness (0–1).
        Call multiple times per frame for stiffer cloth.
        """
        dx = other.x - self.x
        dy = other.y - self.y
        dist = math.sqrt(dx * dx + dy * dy) + 1e-8
        diff = (dist - rest_length) / dist * stiffness

        if not self.pinned:
            self.x += dx * 0.5 * diff
            self.y += dy * 0.5 * diff
        if not other.pinned:
            other.x -= dx * 0.5 * diff
            other.y -= dy * 0.5 * diff

    # ── Boundary enforcement ─────────────────────────────────────────────────

    def clamp(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        """Hard-clamp position into canvas bounds. Pinned particles are unaffected."""
        if self.pinned:
            return
        self.x = max(x_min, min(x_max, self.x))
        self.y = max(y_min, min(y_max, self.y))
