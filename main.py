#!/usr/bin/env python3
"""
Silk Audio Visualizer — CLI entry point and main render loop.

Usage:
    python main.py --input track.mp3
    python main.py --input track.mp3 --output my_video.mp4
    python main.py --input track.mp3 --preview
    python main.py --input track.mp3 --width 1080 --height 1080 --fps 60
    python main.py --input track.mp3 --config my_config.yaml

See README.md for full documentation.
"""

from __future__ import annotations

import os
import sys
import time

import click
import pygame
import yaml

from audio.analyzer import AudioAnalyzer
from export.recorder import Recorder
from physics.world import World
from render.canvas import Canvas


@click.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    help="Input audio file path (MP3/WAV/FLAC/OGG).",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    default="silk_output.mp4",
    show_default=True,
    help="Output video file.",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    default="config.yaml",
    show_default=True,
    help="YAML config path.",
)
@click.option(
    "--preview", "-p", is_flag=True, help="Show real-time pygame window (no export)."
)
@click.option(
    "--width", default=1920, show_default=True, help="Output width in pixels."
)
@click.option(
    "--height", default=1080, show_default=True, help="Output height in pixels."
)
@click.option("--fps", default=30, show_default=True, help="Frames per second.")
def main(
    input_path: str,
    output_path: str,
    config_path: str,
    preview: bool,
    width: int,
    height: int,
    fps: int,
) -> None:
    """Silk Audio Visualizer — physics-based cloth simulation synced to music."""

    # ── 1. Validate input file ────────────────────────────────────────────────
    if not os.path.isfile(input_path):
        print(f"[ERROR] Audio file not found: {input_path}")
        sys.exit(1)

    # ── 2. Load config ────────────────────────────────────────────────────────
    if not os.path.isfile(config_path):
        print(f"[ERROR] Configuration file not found: {config_path}")
        sys.exit(1)
    try:
        with open(config_path) as fh:
            cfg: dict = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        print(f"[ERROR] Invalid YAML syntax in configuration file:\n  {exc}")
        sys.exit(1)

    # Inject CLI overrides
    cfg.update({"width": width, "height": height, "fps": fps})

    # ── 3. Set up pygame display mode BEFORE init ─────────────────────────────
    # In export (non-preview) mode we run headless — no display server needed.
    if not preview:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    pygame.init()

    # ── 4. Analyse audio ──────────────────────────────────────────────────────
    print(f"[silk] Analysing audio: {input_path}")
    try:
        analyzer = AudioAnalyzer(input_path, fps=fps, config=cfg)
    except Exception as exc:
        print(f"[ERROR] Audio analysis failed: {exc}")
        pygame.quit()
        sys.exit(1)
    print(f"[silk] {analyzer.n_frames} frames | {analyzer.duration:.1f}s | {fps} fps")

    # ── 5. Build world + canvas ───────────────────────────────────────────────
    world = World(cfg, width, height)
    canvas = Canvas(width, height, cfg)

    # ── 6. Set up display / recorder ─────────────────────────────────────────
    if preview:
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Silk Visualizer — Preview")
        clock = pygame.time.Clock()
        recorder: Recorder | None = None
    else:
        screen = None
        clock = None
        try:
            recorder = Recorder(output_path, width, height, fps)
        except RuntimeError as exc:
            print(str(exc))
            pygame.quit()
            sys.exit(1)

    # ── 7. Main render loop ───────────────────────────────────────────────────
    dt = 1.0 / fps
    start_wall = time.time()
    log_every = max(1, fps * 5)  # progress print every 5 s of source content

    print("[silk] Rendering…")
    for frame_idx in range(analyzer.n_frames):
        # Handle window close in preview mode
        if preview:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("[silk] Preview closed.")
                    pygame.quit()
                    return

        # Audio features for this frame
        features = analyzer.get_frame(frame_idx)

        # Step physics simulation
        world.step(features, dt)

        # Draw composite frame
        frame_surf = canvas.draw_frame(world, features)

        # Output: either show in window or write to video
        if preview:
            screen.blit(frame_surf, (0, 0))
            pygame.display.flip()
            clock.tick(fps)
        else:
            recorder.write_frame(frame_surf)
            if frame_idx % log_every == 0:
                elapsed_src = frame_idx / fps
                elapsed_wall = time.time() - start_wall
                print(
                    f"[silk] {elapsed_src:5.0f}s / {analyzer.duration:.0f}s  "
                    f"({elapsed_wall:.0f}s elapsed)"
                )

    # ── 8. Finalise export ────────────────────────────────────────────────────
    if recorder is not None:
        print("[silk] Muxing audio…")
        recorder.finalize(input_path)
        total_wall = time.time() - start_wall
        print(f"[silk] Done → {output_path}  ({total_wall:.1f}s total)")

    pygame.quit()


if __name__ == "__main__":
    main()
