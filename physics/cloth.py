"""
ClothMesh — NumPy-vectorised Verlet cloth simulation.

A ROWS × COLS grid of particles connected by:
  • Structural springs  (horizontal + vertical neighbours)
  • Shear springs       (diagonal neighbours)

The top row is pinned at its initial positions.
Audio energy drives lateral travelling waves, vertical ripples,
and beat-snap impulses through the mesh each frame.
"""

from __future__ import annotations

import colorsys
import math
from typing import Tuple

import numpy as np


class ClothMesh:
    """A rectangular cloth mesh, anchored along its top edge."""

    def __init__(
        self,
        x: float,  # left edge of rest-position grid
        y: float,  # top edge of rest-position grid
        width: float,  # total width in pixels
        height: float,  # total height in pixels
        rows: int,  # number of particle rows  (vertical)
        cols: int,  # number of particle columns (horizontal)
        stiffness: float,  # spring stiffness scalar  0–1
        damping: float,  # Verlet velocity damping  (0.90–0.99)
        hue: float,  # base colour hue for this cloth (0–1)
        phase_offset: float,  # phase of audio wave (radians) — creates layering
        config: dict,
    ):
        self.rows = rows
        self.cols = cols
        self.stiffness = stiffness
        self.damping = damping
        self.hue = hue
        self.phase_offset = phase_offset
        self.config = config
        self.time: float = 0.0

        # Rest cell dimensions
        self.rest_dx = width / max(cols - 1, 1)
        self.rest_dy = height / max(rows - 1, 1)
        self.rest_diag = math.sqrt(self.rest_dx**2 + self.rest_dy**2)

        # ── Particle state arrays: shape (rows, cols, 2) ─────────────────────
        self.pos = np.empty((rows, cols, 2), dtype=np.float64)
        self.prev_pos = np.empty((rows, cols, 2), dtype=np.float64)
        self.forces = np.zeros((rows, cols, 2), dtype=np.float64)

        # Lay out particles in a regular grid
        xs = np.linspace(x, x + width, cols)  # shape (cols,)
        ys = np.linspace(y, y + height, rows)  # shape (rows,)
        self.pos[:, :, 0] = xs[np.newaxis, :]  # broadcast across rows
        self.pos[:, :, 1] = ys[:, np.newaxis]  # broadcast across cols
        self.prev_pos[:] = self.pos

        # ── Pinned mask ──────────────────────────────────────────────────────
        self.pinned = np.zeros((rows, cols), dtype=bool)
        self.pinned[0, :] = True  # anchor entire top row

        # Store rest positions of pinned particles so constraints can re-anchor
        self.pin_pos = self.pos.copy()

    # ─────────────────────────────────────────────────────────────────────────
    # Audio forces
    # ─────────────────────────────────────────────────────────────────────────

    def apply_audio_force(self, features: dict, dt: float) -> None:
        """Accumulate audio-driven forces into self.forces for this frame."""
        bass = features["bass"]
        mids = features["mids"]
        highs = features["highs"]
        onset = features["onset"]
        is_beat = features["is_beat"]

        cfg = self.config
        wave_amp = cfg.get("bass_wave_amp", 130.0)
        beat_f = cfg.get("beat_snap_force", 220.0)
        turb = cfg.get("turbulence_scale", 35.0)
        min_f = cfg.get("cloth_min_force", 12.0)

        self.time += dt

        # Pre-build column and row index arrays for vectorised force application
        col_idx = np.arange(self.cols, dtype=np.float64)[np.newaxis, :]  # (1, cols)
        row_w = np.linspace(0.0, 1.0, self.rows)[:, np.newaxis]  # (rows, 1)
        # row_w == 0 at top (pinned), 1 at bottom → more force toward free edge

        # ── 1. Bass: lateral travelling wave ─────────────────────────────────
        #   Sinusoidal force sweeping across columns and advancing with time.
        wave_freq = 0.7 + bass * 1.6  # Hz; faster when bass is stronger
        wave_k = 2.0 * math.pi / self.cols * 2.0
        amp = max(bass * wave_amp + mids * 45.0, min_f)

        phase = (
            wave_k * col_idx + self.time * wave_freq * 2.0 * math.pi + self.phase_offset
        )
        fx = np.sin(phase) * amp * row_w  # (rows, cols)
        self.forces[:, :, 0] += fx

        # ── 2. Mids: vertical ripple ──────────────────────────────────────────
        v_phase = self.time * 1.8 + col_idx * 0.35 + self.phase_offset * 0.5
        fy = np.sin(v_phase) * mids * 55.0 * row_w
        self.forces[:, :, 1] += fy

        # ── 3. Highs: random turbulence ───────────────────────────────────────
        if highs > 0.04:
            noise = np.random.normal(0.0, highs * turb, (self.rows, self.cols, 2))
            self.forces += noise

        # ── 4. Beat: sharp directional snap ──────────────────────────────────
        if is_beat and onset > 0.20:
            direction = math.sin(self.time * 0.6 + self.phase_offset)
            snap = math.copysign(onset * beat_f, direction)
            self.forces[:, :, 0] += snap * row_w * 2.5

        # ── 5. Gravity (always on) ────────────────────────────────────────────
        gravity = 180.0  # pixels / s²
        self.forces[:, :, 1] += gravity

        # Pinned particles never move — zero any accumulated force on them
        self.forces[self.pinned] = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Physics update
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        dt: float,
        constraint_iterations: int,
        canvas_w: int,
        canvas_h: int,
    ) -> None:
        """Integrate positions, satisfy spring constraints, clamp to canvas."""

        # ── 1. Verlet integration ─────────────────────────────────────────────
        #   x_new = x + (x − x_prev) * damping + F * dt²
        vel = (self.pos - self.prev_pos) * self.damping
        new_pos = self.pos + vel + self.forces * (dt * dt)

        # Restore pinned particles to their rest positions
        new_pos[self.pinned] = self.pin_pos[self.pinned]

        self.prev_pos[:] = self.pos
        self.pos[:] = new_pos
        self.forces[:] = 0.0  # reset for next frame

        # ── 2. Spring constraint solving ──────────────────────────────────────
        # Pinned mask broadcast to 3-D for direct use with position slices
        pin3 = self.pinned[:, :, np.newaxis]  # (rows, cols, 1)

        for _ in range(constraint_iterations):
            # — Structural horizontal: (r,c) ↔ (r,c+1) ——————————————————
            pa = self.pos[:, :-1]
            pb = self.pos[:, 1:]
            pin_a = pin3[:, :-1]
            pin_b = pin3[:, 1:]
            delta = pb - pa
            dist = np.sqrt((delta * delta).sum(axis=2, keepdims=True)) + 1e-8
            corr = delta * (1.0 - self.rest_dx / dist) * (0.5 * self.stiffness)
            self.pos[:, :-1] += np.where(pin_a, 0.0, corr)
            self.pos[:, 1:] -= np.where(pin_b, 0.0, corr)

            # — Structural vertical: (r,c) ↔ (r+1,c) ————————————————————
            pa = self.pos[:-1]
            pb = self.pos[1:]
            pin_a = pin3[:-1]
            pin_b = pin3[1:]
            delta = pb - pa
            dist = np.sqrt((delta * delta).sum(axis=2, keepdims=True)) + 1e-8
            corr = delta * (1.0 - self.rest_dy / dist) * (0.5 * self.stiffness)
            self.pos[:-1] += np.where(pin_a, 0.0, corr)
            self.pos[1:] -= np.where(pin_b, 0.0, corr)

            # — Shear diagonal: (r,c) ↔ (r+1,c+1) ——————————————————————
            pa = self.pos[:-1, :-1]
            pb = self.pos[1:, 1:]
            pin_a = pin3[:-1, :-1]
            pin_b = pin3[1:, 1:]
            delta = pb - pa
            dist = np.sqrt((delta * delta).sum(axis=2, keepdims=True)) + 1e-8
            corr = delta * (1.0 - self.rest_diag / dist) * (0.5 * self.stiffness * 0.5)
            self.pos[:-1, :-1] += np.where(pin_a, 0.0, corr)
            self.pos[1:, 1:] -= np.where(pin_b, 0.0, corr)

            # — Shear anti-diagonal: (r,c+1) ↔ (r+1,c) ————————————————
            pa = self.pos[:-1, 1:]
            pb = self.pos[1:, :-1]
            pin_a = pin3[:-1, 1:]
            pin_b = pin3[1:, :-1]
            delta = pb - pa
            dist = np.sqrt((delta * delta).sum(axis=2, keepdims=True)) + 1e-8
            corr = delta * (1.0 - self.rest_diag / dist) * (0.5 * self.stiffness * 0.5)
            self.pos[:-1, 1:] += np.where(pin_a, 0.0, corr)
            self.pos[1:, :-1] -= np.where(pin_b, 0.0, corr)

            # Re-anchor pinned particles after each pass (prevents drift)
            self.pos[self.pinned] = self.pin_pos[self.pinned]

        # ── 3. Loose canvas bounds (allow slight overshoot for drama) ─────────
        self.pos[~self.pinned, 0] = np.clip(
            self.pos[~self.pinned, 0], -canvas_w * 0.25, canvas_w * 1.25
        )
        self.pos[~self.pinned, 1] = np.clip(
            self.pos[~self.pinned, 1], -20.0, canvas_h * 1.15
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Colour
    # ─────────────────────────────────────────────────────────────────────────

    def get_quad_color(
        self,
        r: int,
        c: int,
        features: dict,
        global_hue: float,
    ) -> Tuple[int, int, int]:
        """
        Compute the silk colour for the quad at grid cell (r, c).

        Combines:
          - Base hue (cloth's own + global drift + subtle width spread)
          - Saturation driven by mids energy
          - Value envelope: bright band in the upper-middle zone, fading up/down
          - Moving specular highlight that creates the silk sheen
        """
        t_col = c / max(self.cols - 1, 1)  # 0 = left,  1 = right
        t_row = r / max(self.rows - 1, 1)  # 0 = top,   1 = bottom

        # Hue: cloth base + slow global drift + slight spread across width
        h = (self.hue + global_hue + t_col * 0.10) % 1.0

        # Saturation: rich silk colour, boosted by mids
        s = min(1.0, 0.72 + features["mids"] * 0.28)

        # Value: peaks around 30% down the cloth (upper-lit area)
        brightness_env = max(0.0, 1.0 - abs(t_row - 0.28) * 2.2)
        v = min(
            1.0,
            0.12
            + brightness_env * 0.60
            + features["bass"] * 0.20
            + features["rms"] * 0.08,
        )

        # ── Specular highlight: bright near-white band sweeping up-down ──────
        # Simulates a light source reflecting off the fabric surface
        hl_y = 0.20 + 0.16 * math.sin(self.time * 1.4 + self.phase_offset)
        hl_x = 0.50 + 0.20 * math.sin(self.time * 0.8 + self.phase_offset + 1.1)
        dist_y = abs(t_row - hl_y)
        dist_x = abs(t_col - hl_x)
        hl_dist = math.sqrt(dist_y**2 + (dist_x * 0.5) ** 2)
        if hl_dist < 0.12:
            strength = (0.12 - hl_dist) / 0.12  # 0 at edge → 1 at centre
            v = min(1.0, v + strength * 0.55)
            s = max(0.0, s - strength * 0.45)  # wash toward white at centre

        r_, g_, b_ = colorsys.hsv_to_rgb(h, s, v)
        return (int(r_ * 255), int(g_ * 255), int(b_ * 255))
