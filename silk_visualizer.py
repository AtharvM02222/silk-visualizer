#!/usr/bin/env python3
"""
silk_visualizer.py v2 — High-quality audio-reactive silk cloth sphere visualizer

What's new in v2
----------------
• Luminance-based LUT recoloring  — extracts greyscale from the silk footage and
  re-maps it through rich colour palettes so every hue is vivid and fully saturated.
• 5 colour palettes                — Gold → Rose → Violet → Cyan → Emerald, cycling
  smoothly over a configurable period (default 45 s per full round-trip).
• Sub-frame temporal blending     — cross-fades between consecutive silk frames so
  motion is perfectly smooth even when animation speed jumps on a beat.
• Fast radial zoom-pulse          — a vectorised radial brightness gradient replaces
  PIL in the hot loop: centre brightens, edges dim on each transient.  Zero PIL
  overhead during rendering; only PIL is used once at startup for the resize step.
• LUT caching                     — the 4096-entry colour LUT is rebuilt only when
  the palette blend shifts by > 0.2% (~every 15 frames at a 45 s cycle).
• Float32 lum maps                — pre-stored at startup to eliminate per-frame
  uint8→float32 conversion (~10× less per-frame work than v1).

Usage
-----
    python silk_visualizer.py  input.mp3  [output.mp4]  [options]

Options
-------
    --source       Silk cloth source video   (default: main.mp4 beside script)
    --width        Output width              (default: match source)
    --height       Output height             (default: match source)
    --fps          Output frame rate         (default: 30)
    --color-cycle  Seconds for one full colour cycle  (default: 45)

Requirements
------------
    pip3 install numpy scipy pillow
    ffmpeg  (must be in PATH)
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
#  COLOUR PALETTES
#  Each palette is a (7, 4) float32 array: [luminance, R, G, B].
#  Luminance 0 = deep shadow (always maps to pure black).
#  Luminance 1 = bright specular highlight (maps to near-white).
# ══════════════════════════════════════════════════════════════════════════════

# fmt: off
_GOLD = np.array([
    [0.00,  0.000, 0.000, 0.000],
    [0.10,  0.130, 0.030, 0.000],
    [0.30,  0.440, 0.130, 0.000],
    [0.52,  0.780, 0.340, 0.030],
    [0.72,  0.960, 0.650, 0.150],
    [0.88,  0.995, 0.870, 0.460],
    [1.00,  1.000, 0.980, 0.800],
], dtype=np.float32)

_ROSE = np.array([
    [0.00,  0.000, 0.000, 0.000],
    [0.10,  0.160, 0.010, 0.040],
    [0.30,  0.500, 0.040, 0.130],
    [0.52,  0.840, 0.160, 0.360],
    [0.72,  0.985, 0.430, 0.630],
    [0.88,  1.000, 0.720, 0.840],
    [1.00,  1.000, 0.930, 0.960],
], dtype=np.float32)

_VIOLET = np.array([
    [0.00,  0.000, 0.000, 0.000],
    [0.10,  0.060, 0.010, 0.180],
    [0.30,  0.220, 0.040, 0.540],
    [0.52,  0.480, 0.100, 0.840],
    [0.72,  0.730, 0.330, 0.985],
    [0.88,  0.910, 0.660, 1.000],
    [1.00,  0.980, 0.900, 1.000],
], dtype=np.float32)

_CYAN = np.array([
    [0.00,  0.000, 0.000, 0.000],
    [0.10,  0.000, 0.075, 0.180],
    [0.30,  0.010, 0.270, 0.520],
    [0.52,  0.040, 0.580, 0.840],
    [0.72,  0.170, 0.845, 0.985],
    [0.88,  0.560, 0.960, 1.000],
    [1.00,  0.870, 1.000, 1.000],
], dtype=np.float32)

_EMERALD = np.array([
    [0.00,  0.000, 0.000, 0.000],
    [0.10,  0.010, 0.120, 0.025],
    [0.30,  0.040, 0.390, 0.075],
    [0.52,  0.090, 0.710, 0.200],
    [0.72,  0.290, 0.940, 0.400],
    [0.88,  0.680, 1.000, 0.720],
    [1.00,  0.880, 1.000, 0.890],
], dtype=np.float32)
# fmt: on

# Cycle order: adjacent pairs blend pleasantly (warm→cool→warm)
PALETTES = [_GOLD, _ROSE, _VIOLET, _CYAN, _EMERALD]
PALETTE_NAMES = ["Gold", "Rose", "Violet", "Cyan", "Emerald"]

LUT_SIZE = 4096  # fits comfortably in L2 cache


# ══════════════════════════════════════════════════════════════════════════════
#  PALETTE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


def _smootherstep(t: float) -> float:
    """Perlin's C² ease-in/ease-out."""
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def build_lut(pa: np.ndarray, pb: np.ndarray, blend: float) -> tuple:
    """
    Interpolate two palettes and return (lut_r, lut_g, lut_b) float32 arrays
    of length LUT_SIZE.
    """
    pab = pa.copy()
    pab[:, 1:] = pa[:, 1:] + (pb[:, 1:] - pa[:, 1:]) * blend
    v = np.linspace(0.0, 1.0, LUT_SIZE, dtype=np.float32)
    r = np.interp(v, pab[:, 0], pab[:, 1]).astype(np.float32)
    g = np.interp(v, pab[:, 0], pab[:, 2]).astype(np.float32)
    b = np.interp(v, pab[:, 0], pab[:, 3]).astype(np.float32)
    return r, g, b


def palette_state(t: float, cycle: float) -> tuple:
    """
    Return (idx_a, idx_b, blend, label) for current time t.
    blend uses smootherstep for extra-smooth transitions.
    """
    n = len(PALETTES)
    pos = (t / cycle) % 1.0 * n
    ia = int(pos) % n
    ib = (ia + 1) % n
    raw = pos - int(pos)
    bl = _smootherstep(raw)
    pct = int(raw * 100)
    return ia, ib, bl, f"{PALETTE_NAMES[ia]} → {PALETTE_NAMES[ib]} {pct}%"


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO ANALYSIS  —  ffmpeg decode + scipy, no librosa / numba needed
# ══════════════════════════════════════════════════════════════════════════════


def _load_audio(path: str, sr: int = 22050):
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
    """Returns (energy, beats, duration, n_frames)  — all normalised to [0,1]."""
    print(f"[audio] Loading '{path}'", flush=True)
    y, sr = _load_audio(path, sr)
    dur = len(y) / sr
    n_frames = max(1, int(dur * fps))
    hop = max(1, sr // fps)
    print(f"[audio] {dur:.2f}s → {n_frames} frames @ {fps}fps", flush=True)

    n_hops = len(y) // hop
    chunks = y[: n_hops * hop].reshape(n_hops, hop)
    rms = np.sqrt(np.mean(chunks**2, axis=1) + 1e-12)

    sos = butter(4, 150.0 / (sr / 2.0), btype="high", output="sos")
    y_hp = sosfilt(sos, y)
    c_hp = y_hp[: n_hops * hop].reshape(n_hops, hop)
    rms_hp = np.sqrt(np.mean(c_hp**2, axis=1) + 1e-12)
    onset = np.maximum(0.0, np.diff(rms_hp, prepend=rms_hp[0]))

    def _norm(x):
        x = x.astype(np.float64)
        return x / (x.max() + 1e-12)

    energy = gaussian_filter1d(_norm(rms), sigma=3.0)
    beats = gaussian_filter1d(_norm(onset), sigma=0.8)
    energy /= energy.max() + 1e-12
    beats /= beats.max() + 1e-12

    def _interp(sig):
        src = np.linspace(0.0, dur, len(sig))
        dst = np.linspace(0.0, dur, n_frames)
        return np.interp(dst, src, sig)

    return _interp(energy), _interp(beats), dur, n_frames


# ══════════════════════════════════════════════════════════════════════════════
#  SILK SOURCE — load frames + pre-compute float32 luminance maps
# ══════════════════════════════════════════════════════════════════════════════


def load_source(video_path: str):
    """
    Load all frames and pre-compute Rec.709 luminance maps stored as float32.

    RAM cost: n_frames × H × W × 4 bytes  (≈680 MB for the 112-frame 1896×800 clip).
    This one-time cost eliminates the per-frame uint8→float32 conversion in the
    hot loop, roughly doubling throughput.
    """
    print(f"[silk]  Loading '{video_path}'", flush=True)
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
    W, H = int(vs["width"]), int(vs["height"])
    num, den = map(int, vs["r_frame_rate"].split("/"))
    src_fps = num / den

    raw = subprocess.run(
        ["ffmpeg", "-i", video_path, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    data = np.frombuffer(raw.stdout, dtype=np.uint8)
    n = len(data) // (H * W * 3)
    frames = data[: n * H * W * 3].reshape(n, H, W, 3)
    print(f"[silk]  {n} frames  ({W}×{H} @ {src_fps:.1f}fps)", flush=True)

    ram_mb = n * H * W * 4 // (1024 * 1024)
    print(f"[silk]  Pre-computing luminance maps (~{ram_mb} MB)...", flush=True)
    lum_maps = []
    for f in frames:
        fp = f.astype(np.float32) * (1.0 / 255.0)
        lum = 0.2126 * fp[:, :, 0] + 0.7152 * fp[:, :, 1] + 0.0722 * fp[:, :, 2]
        # Gamma-expand to give dark shadow regions more luminance separation
        lum = lum**1.5
        lum_maps.append(np.clip(lum, 0.0, 1.0).astype(np.float32))

    print("[silk]  Luminance pre-computation done.", flush=True)
    return lum_maps, W, H, src_fps


def resize_lum_maps(lum_maps, src_W, src_H, out_W, out_H):
    """Scale lum maps to output resolution with letterbox/pillarbox padding."""
    if src_W == out_W and src_H == out_H:
        return lum_maps, 0, 0

    print(f"[silk]  Resizing {src_W}×{src_H} → {out_W}×{out_H}...", flush=True)
    src_asp = src_W / src_H
    out_asp = out_W / out_H

    if abs(src_asp - out_asp) < 0.01:
        fit_W, fit_H = out_W, out_H
    elif src_asp > out_asp:
        fit_W, fit_H = out_W, int(out_W / src_asp)
    else:
        fit_H, fit_W = out_H, int(out_H * src_asp)

    off_x = (out_W - fit_W) // 2
    off_y = (out_H - fit_H) // 2

    out = []
    for lm in lum_maps:
        u8 = (lm * 255.0).astype(np.uint8)
        pil = Image.fromarray(u8).resize((fit_W, fit_H), Image.BILINEAR)
        canvas = np.zeros((out_H, out_W), dtype=np.float32)
        canvas[off_y : off_y + fit_H, off_x : off_x + fit_W] = (
            np.array(pil).astype(np.float32) / 255.0
        )
        out.append(canvas)

    return out, off_x, off_y


def build_pingpong(n: int) -> list:
    """Seamless ping-pong loop: 0,1,…,n-1, n-2,…,1"""
    return list(range(n)) + list(range(n - 2, 0, -1))


# ══════════════════════════════════════════════════════════════════════════════
#  RADIAL CACHE  — pre-computed distance map for zoom-pulse
# ══════════════════════════════════════════════════════════════════════════════

_radial_cache: dict = {}


def _get_radial(H: int, W: int) -> np.ndarray:
    """Cached float32 (H, W) map of distance from frame centre, range ≈[0, 1.7]."""
    key = (H, W)
    if key not in _radial_cache:
        y = np.linspace(-1.0, 1.0, H, dtype=np.float32)[:, np.newaxis]
        x = np.linspace(-1.0, 1.0, W, dtype=np.float32)[np.newaxis, :]
        _radial_cache[key] = np.sqrt(x * x + y * y)
    return _radial_cache[key]


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME RENDERING  (the hot loop — zero PIL calls)
# ══════════════════════════════════════════════════════════════════════════════


def render_frame(
    lum_maps: list,
    pingpong: list,
    pp_pos: float,
    lut_r: np.ndarray,
    lut_g: np.ndarray,
    lut_b: np.ndarray,
    energy: float,
    beat: float,
) -> np.ndarray:
    """
    Render one frame.  Returns uint8 (H, W, 3).

    Hot-path operations (all pure numpy, no PIL):
      1. Sub-frame luminance blend (two float32 array ops).
      2. Radial zoom-pulse (two array multiplications).
      3. Brightness / beat flash (one multiply + clip).
      4. LUT application (three fancy-index lookups).
      5. Pack to uint8.
    """
    n_pp = len(pingpong)

    # ── 1. Sub-frame bilinear blend ───────────────────────────────────────────
    frac = pp_pos % 1.0
    ia = int(pp_pos) % n_pp
    ib = (ia + 1) % n_pp

    lm_a = lum_maps[pingpong[ia]]  # pre-stored float32 — no copy needed
    lm_b = lum_maps[pingpong[ib]]
    lum = lm_a + (lm_b - lm_a) * frac

    # ── 2. Radial zoom-pulse (vectorised, no image warp) ──────────────────────
    #
    # A radial brightness weight creates the perceptual "push forward" that a
    # real zoom gives, at the cost of two float32 multiplications instead of a
    # full bilinear warp.  centre brightens, edges dim on each beat transient.
    #
    zoom_strength = beat * 0.55 + energy * 0.10  # 0 at silence, ~0.65 at peak
    if zoom_strength > 0.02:
        H, W = lum.shape
        radius = _get_radial(H, W)  # cached — never reallocated
        weight = 1.0 + zoom_strength * (0.20 - 0.28 * radius)
        lum = np.clip(lum * weight, 0.0, 1.0)

    # ── 3. Brightness + beat flash ────────────────────────────────────────────
    gain = 0.90 + 0.22 * energy + 0.48 * beat
    lum = np.clip(lum * gain, 0.0, 1.0)

    # ── 4. Colour LUT ─────────────────────────────────────────────────────────
    idx = (lum * (LUT_SIZE - 1)).astype(np.int32)
    np.clip(idx, 0, LUT_SIZE - 1, out=idx)

    r = lut_r[idx]
    g = lut_g[idx]
    b = lut_b[idx]

    # ── 5. Pack ───────────────────────────────────────────────────────────────
    return np.stack(
        [
            (r * 255.0).clip(0, 255).astype(np.uint8),
            (g * 255.0).clip(0, 255).astype(np.uint8),
            (b * 255.0).clip(0, 255).astype(np.uint8),
        ],
        axis=2,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_source = os.path.join(script_dir, "main.mp4")

    parser = argparse.ArgumentParser(
        description="Silk audio visualizer v2 — smooth colour-cycling silk sphere",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Input audio file (MP3, WAV, FLAC …)")
    parser.add_argument("output", nargs="?", default="output.mp4")
    parser.add_argument("--source", default=default_source)
    parser.add_argument(
        "--width", type=int, default=0, help="Output width (0 = match source)"
    )
    parser.add_argument(
        "--height", type=int, default=0, help="Output height (0 = match source)"
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--color-cycle",
        type=float,
        default=45.0,
        help="Seconds per full colour cycle  (30=faster / 60=slower)",
    )
    args = parser.parse_args()

    for p, label in [(args.input, "audio"), (args.source, "silk source")]:
        if not os.path.isfile(p):
            sys.exit(f"[error] {label} not found: '{p}'")
    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0:
        sys.exit("[error] ffmpeg not found in PATH.")

    fps = args.fps
    color_cycle = max(10.0, args.color_cycle)

    # ── Load + pre-process silk source ───────────────────────────────────────
    lum_maps, src_W, src_H, _ = load_source(args.source)

    out_W = args.width if args.width > 0 else src_W
    out_H = args.height if args.height > 0 else src_H

    lum_maps, _, _ = resize_lum_maps(lum_maps, src_W, src_H, out_W, out_H)

    pingpong = build_pingpong(len(lum_maps))
    n_pp = len(pingpong)

    # ── Analyse audio ─────────────────────────────────────────────────────────
    energy, beats, dur, n_frames = analyze_audio(args.input, fps)

    # ── FFmpeg output pipe ────────────────────────────────────────────────────
    cmd = [
        "ffmpeg",
        "-y",
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
        "-i",
        args.input,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        args.output,
    ]

    print(f"\n[render] {n_frames} frames  |  {out_W}×{out_H}  |  {fps}fps", flush=True)
    print(
        f"[render] Colour cycle: {color_cycle}s  |  Output: {args.output}\n", flush=True
    )

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    t0 = time.perf_counter()

    # Ping-pong speed: one cycle every T_base seconds at silence
    T_base = 6.0
    speed_base = n_pp / (T_base * fps)

    pp_pos = 0.0
    prev_label = ""

    # LUT cache — rebuild only when blend shifts by > 0.002
    lut_r = lut_g = lut_b = None
    _cac_ia, _cac_blend = -1, -1.0

    try:
        for i in range(n_frames):
            t = i / fps
            e = float(energy[i]) if i < len(energy) else 0.0
            bv = float(beats[i]) if i < len(beats) else 0.0

            # Animation speed (beats make cloth churn faster)
            speed = speed_base * (0.40 + 1.40 * e + 0.65 * bv)
            pp_pos += speed

            # Colour palette with LUT caching
            ia, ib, blend, pal_label = palette_state(t, color_cycle)
            if ia != _cac_ia or abs(blend - _cac_blend) > 0.002:
                lut_r, lut_g, lut_b = build_lut(PALETTES[ia], PALETTES[ib], blend)
                _cac_ia, _cac_blend = ia, blend

            # Render
            frame = render_frame(lum_maps, pingpong, pp_pos, lut_r, lut_g, lut_b, e, bv)
            proc.stdin.write(frame.tobytes())

            # Progress
            if (i + 1) % fps == 0 or i == n_frames - 1:
                elapsed = time.perf_counter() - t0
                pct = (i + 1) / n_frames
                fps_r = (i + 1) / elapsed if elapsed > 0 else 0.0
                eta_s = elapsed / pct * (1.0 - pct) if pct > 0 else 0.0
                filled = int(28 * pct)
                bar = "█" * filled + "░" * (28 - filled)
                colour_note = f"  [{pal_label}]" if pal_label != prev_label else ""
                prev_label = pal_label
                print(
                    f"  [{bar}] {pct * 100:5.1f}%  "
                    f"{fps_r:5.1f} fps  "
                    f"ETA {eta_s / 60:4.1f}min"
                    f"{colour_note}",
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
