"""
Tests for physics/strand.py (and implicitly physics/particle.py).
"""

from __future__ import annotations

import math

import pytest

from physics.particle import Particle
from physics.strand import Strand, _apply_palette

# ── Fixtures ──────────────────────────────────────────────────────────────────

ORIGIN = (960.0, 540.0)
N = 12
SEG_LEN = 9.0
FEATURES_ZERO = {
    "bass": 0.0,
    "mids": 0.0,
    "highs": 0.0,
    "rms": 0.0,
    "onset": 0.0,
    "is_beat": False,
}
FEATURES_LOUD = {
    "bass": 1.0,
    "mids": 1.0,
    "highs": 1.0,
    "rms": 1.0,
    "onset": 1.0,
    "is_beat": True,
}
DT = 1.0 / 30.0


@pytest.fixture
def strand():
    return Strand(
        origin=ORIGIN,
        angle=0.0,
        n_particles=N,
        segment_length=SEG_LEN,
        stiffness=0.82,
        damping=0.975,
        hue=0.5,
        config={},
    )


# ── Construction ──────────────────────────────────────────────────────────────


def test_particle_count(strand):
    assert len(strand.particles) == N


def test_root_is_pinned(strand):
    assert strand.particles[0].pinned is True


def test_free_particles_not_pinned(strand):
    for p in strand.particles[1:]:
        assert p.pinned is False


def test_initial_positions_along_angle(strand):
    """Particles should be spaced segment_length apart along angle=0."""
    for i, p in enumerate(strand.particles):
        expected_x = ORIGIN[0] + i * SEG_LEN
        expected_y = ORIGIN[1]
        assert abs(p.x - expected_x) < 0.1, f"Particle {i} x off"
        assert abs(p.y - expected_y) < 0.1, f"Particle {i} y off"


# ── get_points ────────────────────────────────────────────────────────────────


def test_get_points_length(strand):
    pts = strand.get_points()
    assert len(pts) == N


def test_get_points_tuples(strand):
    for pt in strand.get_points():
        assert len(pt) == 2
        assert all(isinstance(v, float) for v in pt)


# ── Physics ────────────────────────────────────────────────────────────────────


def test_root_stays_pinned_after_update(strand):
    """Root particle must not move no matter what forces are applied."""
    for _ in range(30):
        strand.apply_audio_force(FEATURES_LOUD, DT)
        strand.update(DT)
    assert abs(strand.particles[0].x - ORIGIN[0]) < 1e-6
    assert abs(strand.particles[0].y - ORIGIN[1]) < 1e-6


def test_free_particles_move_under_force(strand):
    """After 30 frames with loud audio, the tip should have moved."""
    tip_before = (strand.particles[-1].x, strand.particles[-1].y)
    for _ in range(30):
        strand.apply_audio_force(FEATURES_LOUD, DT)
        strand.update(DT)
    tip_after = (strand.particles[-1].x, strand.particles[-1].y)
    dist = math.dist(tip_before, tip_after)
    assert dist > 1.0, "Tip did not move under loud audio forces"


def test_particles_stay_on_canvas(strand):
    """Particles should stay within canvas bounds after many loud frames."""
    W, H = 1920, 1080
    for _ in range(60):
        strand.apply_audio_force(FEATURES_LOUD, DT)
        strand.update(DT, canvas_w=W, canvas_h=H)
    for p in strand.particles:
        assert 0 <= p.x <= W, f"x={p.x} out of bounds"
        assert 0 <= p.y <= H, f"y={p.y} out of bounds"


# ── Color ─────────────────────────────────────────────────────────────────────


def test_get_color_returns_rgb_tuple(strand):
    color = strand.get_color(FEATURES_ZERO, global_hue=0.0)
    assert len(color) == 3
    for ch in color:
        assert 0 <= ch <= 255


@pytest.mark.parametrize(
    "palette,expected_min,expected_max",
    [
        ("warm", 0.0, 0.15),
        ("cool", 0.55, 0.75),
    ],
)
def test_palette_hue_range(palette, expected_min, expected_max):
    """_apply_palette should map hue into the expected band."""
    for raw_h in [0.0, 0.3, 0.7, 0.99]:
        h = _apply_palette(raw_h, palette)
        assert expected_min <= h <= expected_max + 1e-9, (
            f"{palette}: h={h:.4f} outside [{expected_min}, {expected_max}]"
        )


# ── Particle unit tests ────────────────────────────────────────────────────────


class TestParticle:
    def test_pinned_does_not_move(self):
        p = Particle(x=100.0, y=100.0, prev_x=100.0, prev_y=100.0, pinned=True)
        p.apply_force(9999.0, 9999.0)
        p.update(DT)
        assert p.x == 100.0 and p.y == 100.0

    def test_force_changes_position(self):
        p = Particle(x=0.0, y=0.0, prev_x=0.0, prev_y=0.0)
        p.apply_force(1000.0, 0.0)
        p.update(DT)
        assert p.x > 0.0

    def test_verlet_velocity(self):
        """Particle with prev_pos != pos should continue moving (velocity)."""
        p = Particle(x=10.0, y=0.0, prev_x=0.0, prev_y=0.0)
        p.update(DT)
        assert p.x > 10.0  # momentum carries it forward

    def test_damping_reduces_velocity(self):
        """Damping should cause velocity to decay each frame."""
        p = Particle(x=10.0, y=0.0, prev_x=0.0, prev_y=0.0, damping=0.5)
        v_prev = 10.0
        for _ in range(5):
            before = p.x
            p.update(DT)
            v_now = p.x - before
            assert v_now <= v_prev + 1e-6, "Velocity did not decay with damping"
            v_prev = v_now

    def test_constrain_distance(self):
        """Two particles at rest_length distance should not move."""
        a = Particle(x=0.0, y=0.0, prev_x=0.0, prev_y=0.0)
        b = Particle(x=10.0, y=0.0, prev_x=10.0, prev_y=0.0)
        a.constrain_distance(b, rest_length=10.0, stiffness=1.0)
        assert abs(a.x - 0.0) < 1e-5
        assert abs(b.x - 10.0) < 1e-5

    def test_constrain_corrects_stretched(self):
        """Stretched spring should move particles closer together."""
        a = Particle(x=0.0, y=0.0, prev_x=0.0, prev_y=0.0)
        b = Particle(x=20.0, y=0.0, prev_x=20.0, prev_y=0.0)
        a.constrain_distance(b, rest_length=10.0, stiffness=1.0)
        dist = abs(b.x - a.x)
        assert abs(dist - 10.0) < 1e-4
