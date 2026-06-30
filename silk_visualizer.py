#!/usr/bin/env python3
"""
silk_visualizer.py — Audio-reactive silk cloth sphere visualizer

Uses the bundled `main.mp4` as the silk cloth animation source (pre-rendered
physics simulation), loops it with audio-driven speed variation, and applies
per-frame audio-reactive effects (brightness flash, saturation, warmth, zoom)
synchronized to any input audio file.

Usage
-----
    python silk_visualizer.py input.mp3 [output.mp4] [options]

Options
-------
    --source    Path to silk source video  (default: main.mp4 in script dir)
    --width     Output width               (default: match source video)
    --height    Output height              (default: match source video)
    --fps       Output frame rate          (default: 30)

Requirements
------------
    numpy, scipy, Pillow   →  pip3 install numpy scipy pillow
    ffmpeg                 →  must be in PATH
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfilt

# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO ANALYSIS  —  pure ffmpeg + numpy, no librosa/numba needed
# ══════════════════════════════════════════════════════════════════════════════


def load_audio_ffmpeg(path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Decode any audio format to float64 mono PCM via ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        path,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-ac",
        "1",
        "pipe:1",
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(f"ffmpeg could not decode '{path}'")
    y = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float64) / 32768.0
    return y, sr


def analyze_audio(path: str, fps: int, sr: int = 22050):
    """
    Compute per-frame RMS energy and beat/onset strength from an audio file.

    Returns
    -------
    energy   : float64 (n_frames,)  — smoothed loudness [0, 1]
    beats    : float64 (n_frames,)  — onset strength    [0, 1]
    duration : float
    n_frames : int
    """
    print(f"[audio] Loading '{path}'", flush=True)
    y, sr = load_audio_ffmpeg(path, sr)
    duration = len(y) / sr
    n_frames = max(1, int(duration * fps))
    hop = max(1, sr // fps)
    print(f"[audio] {duration:.2f}s  →  {n_frames} frames @ {fps} fps", flush=True)

    n_hops = len(y) // hop
    chunks = y[: n_hops * hop].reshape(n_hops, hop)

    # RMS energy (overall loudness per frame)
    rms = np.sqrt(np.mean(chunks**2, axis=1) + 1e-12)

    # Beat / onset: derivative of high-pass-filtered RMS
    sos = butter(4, 150.0 / (sr / 2.0), btype="high", output="sos")
    y_hp = sosfilt(sos, y)
    chunks_hp = y_hp[: n_hops * hop].reshape(n_hops, hop)
    rms_hp = np.sqrt(np.mean(chunks_hp**2, axis=1) + 1e-12)
    onset = np.maximum(0.0, np.diff(rms_hp, prepend=rms_hp[0]))

    # Normalise to [0, 1]
    rms_n = rms / (rms.max() + 1e-12)
    onset_n = onset / (onset.max() + 1e-12)

    # Smooth
    energy = gaussian_filter1d(rms_n, sigma=3.0)
    beats = gaussian_filter1d(onset_n, sigma=0.8)

    # Re-normalise after smoothing so peaks still hit 1
    energy = energy / (energy.max() + 1e-12)
    beats = beats / (beats.max() + 1e-12)

    # Interpolate to exact frame count
    def to_frames(data: np.ndarray) -> np.ndarray:
        src_t = np.linspace(0.0, duration, len(data))
        dst_t = np.linspace(0.0, duration, n_frames)
        return np.interp(dst_t, src_t, data)

    return to_frames(energy), to_frames(beats), duration, n_frames


# ══════════════════════════════════════════════════════════════════════════════
#  SILK SOURCE — load main.mp4 frames into RAM
# ══════════════════════════════════════════════════════════════════════════════


def load_source_frames(video_path: str) -> tuple[np.ndarray, int, int, float]:
    """
    Extract every frame of the silk source video as an (N, H, W, 3) uint8 array.

    Returns
    -------
    frames   : uint8 array (N, H, W, 3)
    W, H     : frame dimensions
    src_fps  : source frame rate
    """
    print(f"[silk]  Loading source: '{video_path}'", flush=True)

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    info = json.loads(probe.stdout)
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    W = int(vs["width"])
    H = int(vs["height"])
    num, den = map(int, vs["r_frame_rate"].split("/"))
    src_fps = num / den

    # Decode all frames to raw RGB24
    raw = subprocess.run(
        ["ffmpeg", "-i", video_path, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    data = np.frombuffer(raw.stdout, dtype=np.uint8)
    n = len(data) // (H * W * 3)
    frames = data[: n * H * W * 3].reshape(n, H, W, 3).copy()

    print(f"[silk]  {n} frames loaded  ({W}×{H} @ {src_fps:.1f}fps)", flush=True)
    return frames, W, H, src_fps


def build_pingpong(n: int) -> list[int]:
    """
    Build a ping-pong index list: 0,1,...,n-1, n-2,...,1  (seamless loop).
    """
    fwd = list(range(n))
    bwd = list(range(n - 2, 0, -1))
    return fwd + bwd


# ══════════════════════════════════════════════════════════════════════════════
#  PER-FRAME AUDIO EFFECTS
# ══════════════════════════════════════════════════════════════════════════════


def resize_frames(frames: np.ndarray, out_W: int, out_H: int) -> np.ndarray:
    """
    Resize all source frames to output resolution once (letterbox if needed).
    Returns uint8 array (N, out_H, out_W, 3).
    """
    N, src_H, src_W = frames.shape[:3]
    if src_W == out_W and src_H == out_H:
        return frames

    print(f"[silk]  Resizing {src_W}×{src_H} → {out_W}×{out_H}...", flush=True)
    src_aspect = src_W / src_H
    out_aspect = out_W / out_H

    if abs(src_aspect - out_aspect) < 0.01:
        fit_W, fit_H = out_W, out_H
    elif src_aspect > out_aspect:
        fit_W, fit_H = out_W, int(out_W / src_aspect)
    else:
        fit_H, fit_W = out_H, int(out_H * src_aspect)

    off_x = (out_W - fit_W) // 2
    off_y = (out_H - fit_H) // 2

    out = np.zeros((N, out_H, out_W, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        pil = Image.fromarray(f).resize((fit_W, fit_H), Image.LANCZOS)
        out[i, off_y : off_y + fit_H, off_x : off_x + fit_W] = np.array(pil)

    return out


# Pre-computed LUT: zoom crops stored as (crop_y, crop_x) per zoom level step
_ZOOM_CACHE: dict = {}


def apply_audio_fx(
    frame: np.ndarray,
    energy: float,
    beat: float,
) -> np.ndarray:
    """
    Apply audio-reactive visual effects to a single (already-resized) silk frame.

    Effects
    -------
    1. Beat flash      — brief brightness spike on transients
    2. Energy lift     — overall brightness tracks loudness
    3. Saturation lift — colours richer when louder
    4. Warm tint       — more amber/orange on beats
    5. Zoom pulse      — subtle centre-zoom on strong beats (numpy, no PIL)

    All operations are fully vectorised numpy — no Pillow inside the hot loop.
    """
    # Work in float32
    img = frame.astype(np.float32) * (1.0 / 255.0)

    # ── 1+2. Brightness ──────────────────────────────────────────────────────
    bright = 1.0 + 0.12 * energy + 0.42 * beat
    img *= bright

    # ── 3. Saturation lift ───────────────────────────────────────────────────
    lum = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])[
        ..., np.newaxis
    ]
    sat = 1.0 + 0.28 * energy + 0.10 * beat
    img = lum + (img - lum) * sat

    # ── 4. Warm tint (amber push on beats) ───────────────────────────────────
    w = beat * 0.08
    img[:, :, 0] *= 1.0 + w  # R up
    img[:, :, 2] *= 1.0 - w * 0.5  # B slightly down

    np.clip(img, 0.0, 1.0, out=img)

    # ── 5. Zoom pulse (pure numpy nearest-neighbour crop + resize) ───────────
    zoom = beat * 0.06 + energy * 0.015  # 0 → ~7.5% zoom-in
    if zoom > 0.005:
        H, W = img.shape[:2]
        cy = int(H * zoom * 0.5)  # pixels to remove top/bottom
        cx = int(W * zoom * 0.5)  # pixels to remove left/right
        cy = max(1, min(cy, H // 6))
        cx = max(1, min(cx, W // 6))
        cropped = img[cy : H - cy, cx : W - cx]  # smaller region
        # Nearest-neighbour upsample back to full size (fast)
        y_idx = np.linspace(0, cropped.shape[0] - 1, H).astype(np.int32)
        x_idx = np.linspace(0, cropped.shape[1] - 1, W).astype(np.int32)
        img = cropped[np.ix_(y_idx, x_idx)]

    return (img * 255.0).clip(0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    # ── Script directory (to locate main.mp4 by default) ─────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_source = os.path.join(script_dir, "main.mp4")

    parser = argparse.ArgumentParser(
        description="Silk audio visualizer — music-reactive silk cloth sphere",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Input audio file (MP3, WAV, FLAC, AAC, …)")
    parser.add_argument(
        "output", nargs="?", default="output.mp4", help="Output MP4 file"
    )
    parser.add_argument(
        "--source",
        default=default_source,
        help="Silk cloth source video (default: main.mp4)",
    )
    parser.add_argument(
        "--width", type=int, default=0, help="Output width  (0 = match source)"
    )
    parser.add_argument(
        "--height", type=int, default=0, help="Output height (0 = match source)"
    )
    parser.add_argument("--fps", type=int, default=30, help="Output frame rate")
    args = parser.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────────
    for path, label in [(args.input, "audio"), (args.source, "silk source")]:
        if not os.path.isfile(path):
            sys.exit(f"[error] {label} file not found: '{path}'")

    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0:
        sys.exit("[error] ffmpeg not found in PATH.")

    # ── Load silk source frames ───────────────────────────────────────────────
    silk_frames, src_W, src_H, src_fps = load_source_frames(args.source)
    n_silk = len(silk_frames)

    out_W = args.width if args.width > 0 else src_W
    out_H = args.height if args.height > 0 else src_H
    fps = args.fps

    # ── Pre-resize all silk frames to output resolution (done once) ───────────
    silk_frames = resize_frames(silk_frames, out_W, out_H)

    # ── Ping-pong index list for seamless looping ───────────────────────────────
    pingpong = build_pingpong(n_silk)
    n_pp = len(pingpong)

    # ── Analyze audio ─────────────────────────────────────────────────────────
    energy, beats, duration, n_frames = analyze_audio(args.input, fps)

    # ── FFmpeg output pipe ────────────────────────────────────────────────────
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        # Raw video from stdin
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{out_W}x{out_H}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        # Original audio
        "-i",
        args.input,
        # Video encode
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        # Audio encode
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        args.output,
    ]

    print(f"\n[render] {n_frames} frames  |  {out_W}×{out_H}  |  {fps}fps", flush=True)
    print(f"[render] Output → {args.output}\n", flush=True)

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    t0 = time.perf_counter()

    # ── Fractional ping-pong position (advances with audio-driven speed) ──────
    pp_pos = 0.0

    # Base speed: advance through the ping-pong once every ~5 seconds normally.
    # The ping-pong has n_pp frames; at output fps we want a full cycle in T_base s.
    T_base = 5.0  # seconds per ping-pong cycle at energy=0
    speed_base = n_pp / (T_base * fps)  # pp-frames per output-frame at base speed

    try:
        for i in range(n_frames):
            e = float(energy[i]) if i < len(energy) else 0.0
            bv = float(beats[i]) if i < len(beats) else 0.0

            # ── Speed: slow at silence, fast at peaks/beats ───────────────────
            # Range: ~0.4× base (quiet) to ~2.5× base (loud + beat)
            speed = speed_base * (0.4 + 1.4 * e + 0.7 * bv)
            pp_pos += speed

            silk_idx = pingpong[int(pp_pos) % n_pp]
            source_frame = silk_frames[silk_idx]

            # ── Apply audio effects ──────────────────────────────────────────────
            frame = apply_audio_fx(source_frame, e, bv)

            # ── Send to ffmpeg ────────────────────────────────────────────────
            proc.stdin.write(frame.tobytes())

            # ── Progress ──────────────────────────────────────────────────────
            if (i + 1) % fps == 0 or i == n_frames - 1:
                elapsed = time.perf_counter() - t0
                pct = (i + 1) / n_frames
                fps_r = (i + 1) / elapsed if elapsed > 0 else 0.0
                eta_s = elapsed / pct * (1.0 - pct) if pct > 0 else 0.0
                filled = int(30 * pct)
                bar = "█" * filled + "░" * (30 - filled)
                print(
                    f"  [{bar}] {pct * 100:5.1f}%  "
                    f"{fps_r:6.1f} fps render  "
                    f"ETA {eta_s / 60:4.1f}min",
                    flush=True,
                )

    except BrokenPipeError:
        print("[error] ffmpeg pipe closed unexpectedly.", flush=True)
    finally:
        proc.stdin.close()
        ret = proc.wait()
        if ret not in (0, None):
            print(f"[warn] ffmpeg exited with code {ret}", flush=True)

    total = time.perf_counter() - t0
    m, s = divmod(int(total), 60)
    print(f"\n[done] Rendered in {m}m {s}s  →  {args.output}", flush=True)


if __name__ == "__main__":
    main()
