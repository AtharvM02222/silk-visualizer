"""
Tests for config loading and defaults.

Verifies that config.yaml can be parsed and that all documented parameters
have the expected types and value ranges.
"""

from __future__ import annotations

import os

import pytest
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


@pytest.fixture(scope="module")
def cfg() -> dict:
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


# ── Structure tests ────────────────────────────────────────────────────────────


def test_config_loads(cfg):
    assert isinstance(cfg, dict)
    assert len(cfg) > 0


def test_background_color(cfg):
    bg = cfg["background_color"]
    assert isinstance(bg, list) and len(bg) == 3
    for ch in bg:
        assert 0 <= ch <= 255


# ── Strand physics ─────────────────────────────────────────────────────────────


def test_n_strands(cfg):
    assert 1 <= cfg["n_strands"] <= 20


def test_n_particles(cfg):
    assert 5 <= cfg["n_particles"] <= 50


def test_segment_length(cfg):
    assert 1.0 <= cfg["segment_length"] <= 30.0


def test_stiffness(cfg):
    assert 0.0 <= cfg["stiffness"] <= 1.0


def test_damping(cfg):
    assert 0.0 < cfg["damping"] < 1.0


def test_constraint_iterations(cfg):
    assert 1 <= cfg["constraint_iterations"] <= 20


# ── Audio forces ───────────────────────────────────────────────────────────────


def test_bass_force_mult(cfg):
    assert cfg["bass_force_mult"] > 0


def test_beat_burst_mult(cfg):
    assert cfg["beat_burst_mult"] > 0


# ── Sphere ─────────────────────────────────────────────────────────────────────


def test_sphere_base_radius(cfg):
    assert cfg["sphere_base_radius"] > 0


# ── Colour ─────────────────────────────────────────────────────────────────────


def test_palette_valid(cfg):
    assert cfg["palette"] in {"warm", "cool", "neon", "mono"}


def test_hue_drift_speed(cfg):
    assert 0.0 <= cfg["hue_drift_speed"] <= 0.01


# ── Rendering ─────────────────────────────────────────────────────────────────


def test_trail_decay(cfg):
    assert 0.5 <= cfg["trail_decay"] < 1.0


def test_bloom_threshold(cfg):
    assert 0 <= cfg["bloom_threshold"] <= 255


def test_bloom_blur(cfg):
    assert cfg["bloom_blur"] >= 1


# ── Smoothing ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "key", ["smoothing_bass", "smoothing_mids", "smoothing_highs", "smoothing_rms"]
)
def test_smoothing_range(cfg, key):
    assert 0.0 <= cfg[key] < 1.0
