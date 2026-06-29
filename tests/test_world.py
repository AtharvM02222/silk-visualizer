"""
Tests for physics/world.py
"""

from __future__ import annotations

import math

import pytest

from physics.world import World

# ── Fixtures ──────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "n_strands": 8,
    "n_particles": 12,
    "segment_length": 9.0,
    "stiffness": 0.82,
    "damping": 0.975,
    "constraint_iterations": 4,
    "bass_force_mult": 900.0,
    "mids_force_mult": 250.0,
    "beat_burst_mult": 1400.0,
    "jitter_mult": 55.0,
    "min_force_baseline": 50.0,
    "sphere_base_radius": 28.0,
    "sphere_pulse_scale": 0.45,
    "hue_drift_speed": 0.0002,
    "palette": "cool",
}

FEATURES_SILENT = {
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

W, H = 1920, 1080
DT = 1.0 / 30.0


@pytest.fixture
def world():
    return World(DEFAULT_CONFIG, W, H)


# ── Construction ──────────────────────────────────────────────────────────────


def test_correct_strand_count(world):
    assert len(world.strands) == DEFAULT_CONFIG["n_strands"]


def test_strands_centred(world):
    """Root particle of every strand should be at the canvas centre."""
    cx, cy = W // 2, H // 2
    for strand in world.strands:
        root = strand.particles[0]
        assert abs(root.x - cx) < 1e-6
        assert abs(root.y - cy) < 1e-6


def test_strand_angles_evenly_spaced(world):
    """Consecutive strand angles should differ by 2π/n_strands."""
    n = DEFAULT_CONFIG["n_strands"]
    delta = 2 * math.pi / n
    for i in range(len(world.strands) - 1):
        diff = world.strands[i + 1].angle - world.strands[i].angle
        assert abs(diff - delta) < 1e-9


def test_initial_hues_spread(world):
    """Strands should have hues spread evenly across 0–1."""
    n = DEFAULT_CONFIG["n_strands"]
    for i, strand in enumerate(world.strands):
        expected = i / n
        assert abs(strand.hue - expected) < 1e-9


# ── Step ─────────────────────────────────────────────────────────────────────


def test_step_does_not_crash(world):
    for _ in range(10):
        world.step(FEATURES_LOUD, DT)


def test_global_hue_drifts(world):
    initial_hue = world.global_hue
    for _ in range(10):
        world.step(FEATURES_SILENT, DT)
    assert world.global_hue != initial_hue


def test_global_hue_wraps(world):
    """global_hue should always stay in [0, 1)."""
    world.global_hue = 0.9999
    for _ in range(50):
        world.step(FEATURES_SILENT, DT)
    assert 0.0 <= world.global_hue < 1.0


# ── Sphere radius ─────────────────────────────────────────────────────────────


def test_sphere_radius_at_silence(world):
    r = world.get_sphere_radius(FEATURES_SILENT, base_radius=28.0)
    assert abs(r - 28.0) < 1e-6


def test_sphere_radius_pulses_with_bass(world):
    r_loud = world.get_sphere_radius(FEATURES_LOUD, base_radius=28.0)
    r_silent = world.get_sphere_radius(FEATURES_SILENT, base_radius=28.0)
    assert r_loud > r_silent


def test_sphere_radius_minimum(world):
    """Radius should never drop below base even at silence."""
    r = world.get_sphere_radius(FEATURES_SILENT, base_radius=28.0)
    assert r >= 28.0


# ── Canvas centre ─────────────────────────────────────────────────────────────


def test_world_centre(world):
    assert world.cx == W // 2
    assert world.cy == H // 2
