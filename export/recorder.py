"""
Recorder — Write pygame frames to a video file and mux original audio in.

Two-step process:
1. write_frame() → OpenCV VideoWriter → temporary silent MP4 (_noaudio.mp4)
2. finalize()    → ffmpeg mux audio into final output MP4 (via subprocess)

Falls back to copying the silent video if ffmpeg is unavailable.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import cv2
import numpy as np
import pygame


class Recorder:
    """Record pygame Surface frames to an H.264 MP4 with audio."""

    def __init__(self, output_path: str, width: int, height: int, fps: int):
        self.output_path = output_path
        self.fps = fps
        # Temporary file for the silent video pass
        base, ext = os.path.splitext(output_path)
        self.temp_path = base + "_noaudio" + (ext or ".mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.temp_path, fourcc, fps, (width, height))

        if not self.writer.isOpened():
            raise RuntimeError(
                f"[ERROR] Failed to initialize video writer at: {self.temp_path}\n"
                "Check that the output directory exists and is writable."
            )

    # ── Per-frame write ──────────────────────────────────────────────────────

    def write_frame(self, surface: pygame.Surface) -> None:
        """Convert a pygame RGB surface to BGR and write to the video stream."""
        arr = pygame.surfarray.array3d(surface)  # (W, H, 3) RGB
        arr = arr.transpose(1, 0, 2)  # → (H, W, 3) RGB
        arr = arr[:, :, ::-1]  # RGB → BGR (OpenCV expects BGR)
        self.writer.write(arr.astype(np.uint8))

    # ── Finalise ─────────────────────────────────────────────────────────────

    def finalize(self, audio_path: str) -> None:
        """
        Release the video writer, then mux audio into the final output file.

        Tries ffmpeg subprocess first (fastest, no Python dependency version
        issues). Falls back to a warning + silent-video copy if ffmpeg is
        not on PATH.
        """
        self.writer.release()

        muxed = self._mux_with_ffmpeg(audio_path)
        if not muxed:
            # Try moviepy as secondary fallback
            muxed = self._mux_with_moviepy(audio_path)

        if not muxed:
            print("[WARN] Audio mux skipped — output file has no audio.")
            if self.temp_path != self.output_path:
                shutil.copy(self.temp_path, self.output_path)

        # Clean up temp silent video
        if os.path.exists(self.temp_path) and self.temp_path != self.output_path:
            try:
                os.remove(self.temp_path)
            except OSError:
                pass

    def _mux_with_ffmpeg(self, audio_path: str) -> bool:
        """Use ffmpeg subprocess to mux audio. Returns True on success."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            self.temp_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",  # re-encode video would be slow; just copy stream
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",  # trim to shorter stream (video may be slightly shorter)
            self.output_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                return True
            print(f"[WARN] ffmpeg mux returned non-zero: {result.stderr[-300:]}")
            return False
        except FileNotFoundError:
            print("[WARN] ffmpeg not found on PATH — trying moviepy...")
            return False
        except subprocess.TimeoutExpired:
            print("[WARN] ffmpeg timed out during audio mux.")
            return False

    def _mux_with_moviepy(self, audio_path: str) -> bool:
        """Fallback: use moviepy to mux audio. Returns True on success."""
        try:
            # Support both moviepy 1.x and 2.x import paths
            try:
                from moviepy.editor import AudioFileClip, VideoFileClip  # type: ignore
            except ImportError:
                from moviepy import AudioFileClip, VideoFileClip  # type: ignore

            video = VideoFileClip(self.temp_path)
            audio = AudioFileClip(audio_path)
            final = video.set_audio(audio.subclip(0, video.duration))
            final.write_videofile(
                self.output_path,
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )
            video.close()
            audio.close()
            return True
        except Exception as exc:
            print(f"[WARN] moviepy mux failed: {exc}")
            return False
