# 🎵 Silk Audio Visualizer — Python Build Spec

> Physics-based cloth simulation that reacts to audio. Give it an MP3/WAV, get back a video (or real-time preview).

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Visual Design — What "Silk" Actually Is](#visual-design)
3. [Architecture Overview](#architecture-overview)
4. [Project Structure](#project-structure)
5. [Dependencies](#dependencies)
6. [Module-by-Module Spec](#module-by-module-spec)
   - [audio/analyzer.py](#audioanalyzerpy)
   - [physics/particle.py](#physicsparticlepy)
   - [physics/strand.py](#physicsstrandpy)
   - [physics/world.py](#physicsworldpy)
   - [render/canvas.py](#rendercanvaspy)
   - [render/effects.py](#rendereffectspy)
   - [export/recorder.py](#exportrecorderpy)
   - [main.py](#mainpy)
7. [Physics Deep Dive](#physics-deep-dive)
8. [Audio → Visual Mapping](#audio--visual-mapping)
9. [Color System](#color-system)
10. [CLI Interface](#cli-interface)
11. [Config File](#config-file)
12. [Performance Notes](#performance-notes)
13. [Execution Flow (Full Pipeline)](#execution-flow-full-pipeline)
14. [Edge Cases & Gotchas](#edge-cases--gotchas)

---

## What It Does

```
python main.py --input track.mp3 --output silk_video.mp4
```

Takes any audio file → analyzes it frame-by-frame → drives a cloth physics sim → renders silk strands with bloom/glow → exports MP4 with the original audio muxed back in.

Also has a `--preview` flag for real-time pygame window.

---

## Visual Design

The silk visualizer is made of **N symmetrical ribbon strands** that originate from a **glowing center sphere**. Each strand is a chain of physics particles connected by springs. Audio energy pushes forces into the strand tips, which propagates through the chain like cloth in wind.

### Core Visual Elements

| Element | Description |
|---|---|
| **Center Sphere** | Glowing orb at canvas center. Pulses with bass. |
| **Silk Strands** | 6–12 ribbon-like particle chains, radially symmetric. |
| **Symmetry** | Rotational symmetry (strands are mirrored). Optional XY mirror. |
| **Color** | HSV-based. Hue drifts slowly. Saturation/brightness driven by audio energy. |
| **Trails** | Previous frame blended with low alpha (motion blur / fade effect). |
| **Bloom** | Gaussian blur pass on bright pixels, additively composited back. |
| **Background** | Near-black, with very subtle color tint. Trails fade to it over time. |

### What Audio Does

| Audio Feature | Visual Effect |
|---|---|
| Bass (20–200 Hz) | Sphere pulse size + amplitude of strand displacement forces |
| Mids (200–2000 Hz) | Strand tension (spring stiffness varies) + color saturation |
| Highs (2000–16000 Hz) | Bloom intensity + strand tip jitter |
| Beat onset | Sudden burst force on all strands from center outward |
| Amplitude RMS | Overall brightness/alpha of strands |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                     main.py                          │
│          CLI → Config → Pipeline orchestrator        │
└──────────────┬──────────────────────┬────────────────┘
               │                      │
    ┌──────────▼──────────┐  ┌────────▼──────────────┐
    │   audio/analyzer.py  │  │   export/recorder.py   │
    │  librosa FFT, beats, │  │  OpenCV VideoWriter    │
    │  per-frame features  │  │  + ffmpeg audio mux    │
    └──────────┬──────────┘  └────────▲──────────────┘
               │                      │
    ┌──────────▼──────────────────────┴──────────────┐
    │              render/canvas.py                   │
    │         pygame Surface orchestrator             │
    │   - draws background fade                       │
    │   - calls world.step()                          │
    │   - calls strand.draw() for each strand         │
    │   - applies bloom (effects.py)                  │
    └──────────┬──────────────────────────────────────┘
               │
    ┌──────────▼──────────────────────────────────────┐
    │              physics/world.py                    │
    │   - holds all strands                            │
    │   - applies audio forces each frame              │
    │   - calls strand.update() for each               │
    └──────────┬──────────────────────────────────────┘
               │
    ┌──────────▼──────────────────────────────────────┐
    │    physics/strand.py + physics/particle.py       │
    │    Verlet integration, spring constraints        │
    └────────────────────────────────────────────────┘
```

---

## Project Structure

```
silk-visualizer/
├── main.py
├── config.yaml
├── requirements.txt
├── audio/
│   ├── __init__.py
│   └── analyzer.py
├── physics/
│   ├── __init__.py
│   ├── particle.py
│   ├── strand.py
│   └── world.py
├── render/
│   ├── __init__.py
│   ├── canvas.py
│   └── effects.py
└── export/
    ├── __init__.py
    └── recorder.py
```

---

## Dependencies

```txt
# requirements.txt
librosa>=0.10.0          # Audio analysis, FFT, beat tracking
numpy>=1.24.0            # Math everywhere
pygame>=2.5.0            # Real-time rendering window + surface drawing
opencv-python>=4.8.0     # Video frame writing
scipy>=1.11.0            # Gaussian blur for bloom, signal processing
moviepy>=1.0.3           # Mux audio back into exported video
pyyaml>=6.0              # Config file parsing
click>=8.1.0             # CLI interface
```

Install:
```bash
pip install librosa numpy pygame opencv-python scipy moviepy pyyaml click
```

> **Note:** moviepy requires ffmpeg. Install via `pip install imageio[ffmpeg]` or `brew install ffmpeg` / `apt install ffmpeg`.

---

## Module-by-Module Spec

---

### `audio/analyzer.py`

**Purpose:** Load audio, run analysis, expose per-frame feature data as numpy arrays.

#### Class: `AudioAnalyzer`

```python
class AudioAnalyzer:
    def __init__(self, filepath: str, fps: int = 30, sr: int = 22050):
        ...
```

**Constructor:**
- `filepath` — path to MP3/WAV/FLAC/OGG
- `fps` — target video framerate (default 30)
- `sr` — sample rate to resample to (22050 is standard for librosa)

**Internal processing (all in `__init__` or `analyze()`):**

1. `librosa.load(filepath, sr=sr, mono=True)` → `y` (waveform), `sr`
2. Compute hop length: `hop_length = sr // fps` (samples per video frame)
3. STFT: `librosa.stft(y, n_fft=2048, hop_length=hop_length)`
4. Magnitude spectrogram: `np.abs(stft)`
5. Mel spectrogram for perceptual accuracy: `librosa.feature.melspectrogram(...)`
6. Beat tracking: `librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)` → `tempo`, `beat_frames`
7. Onset strength: `librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)`

**Per-frame arrays (all shape `(n_frames,)`):**

| Attribute | How computed |
|---|---|
| `self.rms` | `librosa.feature.rms(y=y, hop_length=hop_length)[0]` — normalized 0–1 |
| `self.bass` | Mean energy of mel bands 0–10 (20–200 Hz), per frame, normalized |
| `self.mids` | Mean energy of mel bands 10–60 (200–2000 Hz), per frame, normalized |
| `self.highs` | Mean energy of mel bands 60–128 (2000–16k Hz), per frame, normalized |
| `self.onset` | `onset_strength`, normalized 0–1 |
| `self.is_beat` | Boolean array — True at frames where a beat occurs |
| `self.n_frames` | Total number of frames = `len(y) // hop_length` |
| `self.duration` | Audio duration in seconds |

**Normalization helper:**
```python
def _normalize(arr: np.ndarray, percentile: float = 97) -> np.ndarray:
    # Clip at 97th percentile to avoid outliers dominating, then scale 0–1
    peak = np.percentile(arr, percentile)
    return np.clip(arr / (peak + 1e-8), 0, 1)
```

**Public method:**
```python
def get_frame(self, frame_idx: int) -> dict:
    return {
        "rms":     self.rms[frame_idx],
        "bass":    self.bass[frame_idx],
        "mids":    self.mids[frame_idx],
        "highs":   self.highs[frame_idx],
        "onset":   self.onset[frame_idx],
        "is_beat": self.is_beat[frame_idx],
    }
```

---

### `physics/particle.py`

**Purpose:** Single point mass in 2D. Uses Verlet integration (no explicit velocity — position history IS velocity).

#### Class: `Particle`

```python
@dataclass
class Particle:
    x: float
    y: float
    prev_x: float
    prev_y: float
    pinned: bool = False          # If True, particle does not move (anchor)
    mass: float = 1.0
    damping: float = 0.98         # Velocity damping per frame (0.95–0.99)
```

**Method: `apply_force(fx, fy)`**
```python
def apply_force(self, fx: float, fy: float):
    if self.pinned:
        return
    # In Verlet, force is added to the "acceleration" term
    # We store accumulated force and apply during update
    self.acc_x += fx / self.mass
    self.acc_y += fy / self.mass
```

**Method: `update(dt)`**  
Verlet integration:
```python
def update(self, dt: float):
    if self.pinned:
        return
    vx = (self.x - self.prev_x) * self.damping
    vy = (self.y - self.prev_y) * self.damping
    new_x = self.x + vx + self.acc_x * dt * dt
    new_y = self.y + vy + self.acc_y * dt * dt
    self.prev_x, self.prev_y = self.x, self.y
    self.x, self.y = new_x, new_y
    self.acc_x = self.acc_y = 0.0   # Reset accumulator
```

**Method: `constrain_distance(other: Particle, rest_length: float, stiffness: float = 1.0)`**  
Spring constraint (XPBD-style iterative correction):
```python
def constrain_distance(self, other: 'Particle', rest_length: float, stiffness: float):
    dx = other.x - self.x
    dy = other.y - self.y
    dist = math.sqrt(dx*dx + dy*dy) + 1e-8
    diff = (dist - rest_length) / dist * stiffness
    if not self.pinned:
        self.x  += dx * 0.5 * diff
        self.y  += dy * 0.5 * diff
    if not other.pinned:
        other.x -= dx * 0.5 * diff
        other.y -= dy * 0.5 * diff
```

---

### `physics/strand.py`

**Purpose:** A chain of particles forming one silk ribbon. The root particle is pinned at the center. Force is applied at the tip.

#### Class: `Strand`

```python
class Strand:
    def __init__(
        self,
        origin: tuple[float, float],  # Center of canvas
        angle: float,                  # Radians — which direction this strand points
        n_particles: int = 20,         # Chain length
        segment_length: float = 8.0,   # Rest length per segment (pixels)
        stiffness: float = 0.85,       # Spring stiffness 0–1
        hue: float = 0.55,             # Base hue (0–1)
    ):
```

**Initialization:**
- Create `n_particles` Particle objects
- Space them from origin outward along `angle`
- Pin particle[0] (root at center sphere)
- All others free

**Attributes:**
- `self.particles: list[Particle]`
- `self.angle: float` — base direction
- `self.hue: float` — base color hue
- `self.stiffness: float`
- `self.n_particles: int`

**Method: `apply_audio_force(features: dict, dt: float)`**

This is where audio drives physics:
```python
def apply_audio_force(self, features: dict, dt: float):
    bass    = features["bass"]
    mids    = features["mids"]
    highs   = features["highs"]
    onset   = features["onset"]
    is_beat = features["is_beat"]

    # Base wind force perpendicular to strand direction (makes it wave)
    perp_angle = self.angle + math.pi / 2
    wind_x = math.cos(perp_angle)
    wind_y = math.sin(perp_angle)

    # Force magnitude: bass drives big sweeping motion
    force_magnitude = bass * 800.0 + mids * 200.0

    # On beat: burst force outward from center along strand direction
    if is_beat:
        burst = onset * 1200.0
        for i, p in enumerate(self.particles[1:], start=1):
            weight = i / self.n_particles  # tip particles affected more
            p.apply_force(
                math.cos(self.angle) * burst * weight,
                math.sin(self.angle) * burst * weight
            )

    # Continuous wind on tip particles (last 40%)
    start = int(self.n_particles * 0.6)
    for i, p in enumerate(self.particles[start:], start=start):
        weight = (i - start) / (self.n_particles - start)
        p.apply_force(
            wind_x * force_magnitude * weight,
            wind_y * force_magnitude * weight
        )

    # Highs add random jitter to all free particles
    jitter = highs * 60.0
    for p in self.particles[1:]:
        p.apply_force(
            random.gauss(0, jitter),
            random.gauss(0, jitter)
        )
```

**Method: `update(dt: float, constraint_iterations: int = 5)`**
```python
def update(self, dt: float, constraint_iterations: int = 5):
    # 1. Update all particle positions
    for p in self.particles:
        p.update(dt)

    # 2. Enforce spring constraints (multiple iterations = more stiff)
    for _ in range(constraint_iterations):
        for i in range(len(self.particles) - 1):
            self.particles[i].constrain_distance(
                self.particles[i+1],
                rest_length=self.segment_length,
                stiffness=self.stiffness
            )
        # Also constrain root to origin (re-pin)
        self.particles[0].x = self.origin[0]
        self.particles[0].y = self.origin[1]
```

**Method: `get_points() -> list[tuple[float, float]]`**  
Returns list of (x, y) for each particle in the chain.

**Method: `get_color(features: dict, global_hue_offset: float) -> tuple[int,int,int]`**  
See [Color System](#color-system).

---

### `physics/world.py`

**Purpose:** Container for all strands. Manages global state, applies per-frame audio data.

#### Class: `World`

```python
class World:
    def __init__(self, config: dict, canvas_w: int, canvas_h: int):
        self.cx = canvas_w // 2
        self.cy = canvas_h // 2
        self.strands: list[Strand] = []
        self.global_hue: float = 0.0
        self.hue_drift_speed: float = config.get("hue_drift_speed", 0.0003)
        self._build_strands(config)
```

**`_build_strands(config)`:**
- Read `n_strands` from config (e.g., 8)
- Place strands at evenly spaced angles: `angle = i * (2π / n_strands)`
- Each strand gets `hue = i / n_strands` (spread across color wheel)
- Apply optional rotational symmetry: if `symmetry = 4`, you have 4 groups of `n_strands // 4` strands mirrored

**`step(features: dict, dt: float)`:**
```python
def step(self, features: dict, dt: float):
    self.global_hue = (self.global_hue + self.hue_drift_speed) % 1.0
    for strand in self.strands:
        strand.apply_audio_force(features, dt)
        strand.update(dt)
```

**`get_sphere_radius(features: dict, base_radius: float) -> float`:**
```python
def get_sphere_radius(self, features: dict, base_radius: float) -> float:
    pulse = features["bass"] * 0.4 + features["onset"] * 0.2
    return base_radius * (1.0 + pulse)
```

---

### `render/canvas.py`

**Purpose:** All drawing logic. Uses pygame surfaces.

#### Class: `Canvas`

```python
class Canvas:
    def __init__(self, width: int, height: int, config: dict):
        pygame.init()
        self.width  = width
        self.height = height
        self.surface = pygame.Surface((width, height))   # Offscreen
        self.config  = config
        self.font    = None  # Optional debug overlay
```

**`draw_frame(world: World, features: dict) -> pygame.Surface`:**

Returns a fully composited frame surface. Steps:

**Step 1 — Trail fade (motion blur effect):**
```python
# Draw a near-black rect with low alpha over existing surface
# This leaves "echoes" of previous frames → silk trail effect
fade = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
fade.fill((0, 0, 0, config["trail_alpha"]))  # e.g., alpha=25
self.surface.blit(fade, (0, 0))
# OR: simpler — multiply all pixel values by 0.92 using numpy
```

> **Implementation note:** The numpy approach is faster:
> ```python
> arr = pygame.surfarray.pixels3d(self.surface)
> arr[:] = (arr * config["trail_decay"]).astype(np.uint8)  # e.g., 0.88
> del arr
> ```

**Step 2 — Draw silk strands:**

For each strand in world.strands:
1. Get particle points: `points = strand.get_points()`
2. Get color: `color = strand.get_color(features, world.global_hue)`
3. Draw as a smooth curve using **Catmull-Rom spline** (see below)
4. Draw with varying line width: thick at root, thin at tip

```python
def draw_strand(self, surface, points, color, base_width=3):
    n = len(points)
    for i in range(n - 1):
        t = i / n
        width = max(1, int(base_width * (1.0 - t * 0.7)))  # tapers toward tip
        alpha = int(200 * (0.4 + 0.6 * (1.0 - t)))
        r, g, b = color
        # pygame doesn't support per-segment alpha on lines easily
        # Use a subsurface approach or draw to a temp alpha surface
        pygame.draw.line(surface, (r, g, b), 
                         (int(points[i][0]),   int(points[i][1])),
                         (int(points[i+1][0]), int(points[i+1][1])),
                         width)
```

For smooth curves, interpolate between particles using Catmull-Rom:
```python
def catmull_rom_points(points: list, subdivisions: int = 4) -> list:
    """
    Given a list of control points, return a denser list of interpolated points
    using Catmull-Rom spline. This makes strands look smooth, not jagged.
    """
    result = []
    for i in range(1, len(points) - 2):
        p0, p1, p2, p3 = points[i-1], points[i], points[i+1], points[i+2]
        for j in range(subdivisions):
            t = j / subdivisions
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * ((2*p1[0]) + (-p0[0]+p2[0])*t +
                       (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 +
                       (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
            y = 0.5 * ((2*p1[1]) + (-p0[1]+p2[1])*t +
                       (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 +
                       (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
            result.append((x, y))
    return result
```

**Step 3 — Draw center sphere:**
```python
def draw_sphere(self, surface, cx, cy, radius, features):
    bass = features["bass"]
    # Outer glow rings (additive blend)
    for i in range(4, 0, -1):
        glow_r = int(radius * (1 + i * 0.4))
        alpha  = int(30 * bass * (1 - i * 0.2))
        glow_s = pygame.Surface((glow_r*2, glow_r*2), pygame.SRCALPHA)
        pygame.draw.circle(glow_s, (*sphere_color, alpha), (glow_r, glow_r), glow_r)
        surface.blit(glow_s, (cx - glow_r, cy - glow_r), special_flags=pygame.BLEND_ADD)
    # Core circle
    pygame.draw.circle(surface, sphere_color, (int(cx), int(cy)), int(radius))
```

**Step 4 — Bloom pass:**
```python
bloom_surface = effects.apply_bloom(self.surface, config["bloom_threshold"], config["bloom_blur"])
self.surface.blit(bloom_surface, (0,0), special_flags=pygame.BLEND_ADD)
```

**Step 5 — Return frame:**
```python
return self.surface.copy()
```

---

### `render/effects.py`

**Purpose:** Post-processing effects — bloom/glow.

#### `apply_bloom(surface, threshold=180, blur_radius=15) -> pygame.Surface`

```python
def apply_bloom(surface: pygame.Surface, threshold: int, blur_radius: int) -> pygame.Surface:
    # 1. Get pixels as numpy array (H, W, 3)
    arr = pygame.surfarray.array3d(surface).transpose(1, 0, 2).astype(np.float32)

    # 2. Extract bright pixels only (above threshold)
    bright_mask = np.max(arr, axis=2, keepdims=True) > threshold
    bright = arr * bright_mask

    # 3. Gaussian blur the bright layer (creates glow spread)
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(bright, sigma=[blur_radius, blur_radius, 0])

    # 4. Convert back to pygame surface
    blurred = np.clip(blurred, 0, 255).astype(np.uint8)
    bloom_arr = blurred.transpose(1, 0, 2)
    bloom_surf = pygame.surfarray.make_surface(bloom_arr)
    return bloom_surf
```

> `BLEND_ADD` during blit makes bloom purely additive — bright areas get brighter, dark areas unaffected.

---

### `export/recorder.py`

**Purpose:** Write frames to video and mux audio.

#### Class: `Recorder`

```python
class Recorder:
    def __init__(self, output_path: str, width: int, height: int, fps: int):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path + "_noaudio.mp4", fourcc, fps, (width, height))
        self.output_path = output_path
        self.temp_path   = output_path + "_noaudio.mp4"
```

**`write_frame(surface: pygame.Surface)`:**
```python
def write_frame(self, surface: pygame.Surface):
    # pygame surface → numpy → BGR (OpenCV format)
    arr = pygame.surfarray.array3d(surface)  # (W, H, 3) RGB
    arr = arr.transpose(1, 0, 2)             # → (H, W, 3)
    arr = arr[:, :, ::-1]                    # RGB → BGR
    self.writer.write(arr)
```

**`finalize(audio_path: str)`:**
```python
def finalize(self, audio_path: str):
    self.writer.release()
    # Mux audio using moviepy
    from moviepy.editor import VideoFileClip, AudioFileClip
    video = VideoFileClip(self.temp_path)
    audio = AudioFileClip(audio_path)
    final = video.set_audio(audio.subclip(0, video.duration))
    final.write_videofile(self.output_path, codec='libx264', audio_codec='aac', logger=None)
    import os
    os.remove(self.temp_path)
```

---

### `main.py`

**Purpose:** CLI entry point + main render loop.

```python
import click
import pygame
import yaml
import time
from audio.analyzer   import AudioAnalyzer
from physics.world    import World
from render.canvas    import Canvas
from export.recorder  import Recorder

@click.command()
@click.option("--input",   "-i", required=True,  help="Input audio file path")
@click.option("--output",  "-o", default="silk_output.mp4", help="Output video path")
@click.option("--config",  "-c", default="config.yaml",     help="Config YAML path")
@click.option("--preview", "-p", is_flag=True,              help="Show real-time pygame window")
@click.option("--width",         default=1920,  help="Output width (default 1920)")
@click.option("--height",        default=1080,  help="Output height (default 1080)")
@click.option("--fps",           default=30,    help="Frames per second (default 30)")
def main(input, output, config, preview, width, height, fps):
    # 1. Load config
    with open(config) as f:
        cfg = yaml.safe_load(f)
    cfg.update({"width": width, "height": height, "fps": fps})

    # 2. Analyze audio
    print(f"[silk] Analyzing audio: {input}")
    analyzer = AudioAnalyzer(input, fps=fps)
    print(f"[silk] {analyzer.n_frames} frames | {analyzer.duration:.1f}s | {fps} fps")

    # 3. Build world + canvas
    world    = World(cfg, width, height)
    canvas   = Canvas(width, height, cfg)
    recorder = Recorder(output, width, height, fps) if not preview else None

    # 4. Setup pygame display (headless if not preview)
    if preview:
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Silk Visualizer")
        clock = pygame.time.Clock()
    else:
        pygame.display.init()
        screen = None

    dt = 1.0 / fps

    # 5. Main loop
    print("[silk] Rendering...")
    for frame_idx in range(analyzer.n_frames):
        # Handle pygame quit
        if preview:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

        # Get audio features for this frame
        features = analyzer.get_frame(frame_idx)

        # Step physics
        world.step(features, dt)

        # Draw frame
        frame_surface = canvas.draw_frame(world, features)

        # Output
        if preview:
            screen.blit(frame_surface, (0, 0))
            pygame.display.flip()
            clock.tick(fps)
        else:
            recorder.write_frame(frame_surface)
            if frame_idx % (fps * 5) == 0:
                elapsed = frame_idx / fps
                print(f"[silk] {elapsed:.0f}s / {analyzer.duration:.0f}s rendered...")

    # 6. Finalize
    if recorder:
        print("[silk] Muxing audio...")
        recorder.finalize(input)
        print(f"[silk] Done → {output}")

    pygame.quit()


if __name__ == "__main__":
    main()
```

---

## Physics Deep Dive

### Why Verlet Integration?

Verlet integration is used instead of Euler because:
- No explicit velocity variable — velocity is inferred from `(current_pos - prev_pos)`
- Naturally damped via damping coefficient on velocity
- More stable for spring constraints with iterative solving
- The silk industry (cloth sims) has used this since the Jakobsen 2001 paper

### Spring Constraint Solving

The `constrain_distance` method is called multiple times per frame (iterations). More iterations = stiffer cloth. 5–8 iterations gives silk-like behavior. 1–2 = loose jelly.

The iteration order matters. Start from root → tip, then tip → root (alternating) for faster convergence.

### Gravity

Optional. Real silk visualization typically has **no gravity** or very weak downward pull (0–2 pixels/frame²). Add a tiny gravity if you want strands to droop slightly:
```python
p.apply_force(0, gravity * p.mass)  # gravity = 0.5–2.0
```

### Damping

Set `damping = 0.97–0.99` for energetic strands that slowly settle. Lower values (0.90) = very quick settling, like underwater. Higher (0.999) = barely any energy loss, chaotic.

---

## Audio → Visual Mapping

```
FREQUENCY BAND    →    VISUAL PARAMETER
─────────────────────────────────────────────────────────
Bass (20–200 Hz)  →    Strand force magnitude (big swings)
                  →    Center sphere radius (pulsing)
                  →    Trail decay rate (more bass = shorter trails)

Mids (200–2kHz)   →    Spring stiffness (mids = tighter cloth)
                  →    Color saturation boost

Highs (2k–16kHz)  →    Bloom intensity
                  →    Random tip jitter (sparkle effect)
                  →    Line width at tips

RMS Amplitude     →    Overall strand brightness (alpha)
                  →    Background color tint intensity

Beat Onset        →    Burst radial force (explosive outward push)
                  →    Sphere color flash (white pulse)
                  →    Hue jump (optional: sudden color shift on beat)
```

### Smoothing

Raw audio features are noisy per-frame. Apply exponential moving average:
```python
SMOOTHING = 0.7  # 0 = no smoothing, 0.9 = heavy smoothing
smoothed_bass = prev_bass * SMOOTHING + new_bass * (1 - SMOOTHING)
```

Do this inside `AudioAnalyzer` as a post-processing step after normalization.

---

## Color System

### HSV → RGB for strands

```python
import colorsys

def get_color(self, features: dict, global_hue: float) -> tuple[int, int, int]:
    # Hue: strand's base hue + global drift + small mids push
    h = (self.hue + global_hue + features["mids"] * 0.05) % 1.0

    # Saturation: base 0.7, boosted by mids
    s = min(1.0, 0.65 + features["mids"] * 0.35)

    # Value/Brightness: base 0.6, boosted by RMS and bass
    v = min(1.0, 0.5 + features["rms"] * 0.3 + features["bass"] * 0.2)

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))
```

### Sphere Color

Sphere uses a brighter, more saturated version of the average strand hue. On beats, flash to near-white (saturation → 0, value → 1).

### Background

Static `(5, 5, 10)` — very dark blue-black. Never pure black (gives depth).

### Palette Presets (in config.yaml)

```yaml
palette: warm       # warm = reds/oranges/yellows
# palette: cool     # cool = blues/purples/cyans
# palette: neon     # neon = high saturation, bright
# palette: mono     # single hue drift only
```

Implement by offsetting `global_hue` starting point and constraining hue range.

---

## CLI Interface

```
Usage: python main.py [OPTIONS]

Options:
  -i, --input  PATH     Input audio file [required]
  -o, --output PATH     Output video file [default: silk_output.mp4]
  -c, --config PATH     Config YAML [default: config.yaml]
  -p, --preview         Show real-time window instead of exporting
  --width  INT          Video width  [default: 1920]
  --height INT          Video height [default: 1080]
  --fps    INT          Frames per second [default: 30]
  --help                Show this message and exit.

Examples:
  python main.py -i track.mp3
  python main.py -i track.mp3 -o output.mp4 --width 1080 --height 1080
  python main.py -i track.mp3 --preview
  python main.py -i track.mp3 -c my_config.yaml
```

---

## Config File

```yaml
# config.yaml — all tunable parameters

# Canvas
background_color: [5, 5, 10]

# Strands
n_strands: 10              # Total number of silk strands
n_particles: 24            # Particles per strand (chain length)
segment_length: 9.0        # Rest length of each spring (pixels)
stiffness: 0.82            # Spring stiffness 0.0–1.0
damping: 0.975             # Verlet velocity damping per frame
constraint_iterations: 6   # Spring solve iterations per frame (more = stiffer)

# Physics forces
bass_force_mult: 900.0     # Multiplier for bass → swing force
mids_force_mult: 250.0     # Multiplier for mids → force
beat_burst_mult: 1400.0    # Force on beat onset
jitter_mult: 50.0          # Random high-freq jitter

# Sphere
sphere_base_radius: 28.0   # Base radius in pixels
sphere_pulse_scale: 0.45   # How much bass makes it grow

# Color
palette: "cool"            # warm / cool / neon / mono
hue_drift_speed: 0.0002    # How fast colors shift over time

# Rendering
trail_decay: 0.88          # Frame echo: 0.80 = heavy trails, 0.95 = short trails
strand_base_width: 3       # Pixel width at root (tapers to 1 at tip)
bloom_threshold: 160       # Pixel brightness to trigger bloom (0–255)
bloom_blur: 18             # Bloom gaussian blur radius (pixels)

# Audio smoothing
smoothing_bass:  0.75
smoothing_mids:  0.65
smoothing_highs: 0.55
smoothing_rms:   0.70
```

---

## Performance Notes

| Task | Bottleneck | Solution |
|---|---|---|
| Bloom pass | Scipy gaussian on full frame | Downscale frame 4× for blur, then upscale back before ADD blit |
| Strand drawing | Python loop over particles | Precompute all Catmull-Rom points into one numpy array, single blit |
| Audio analysis | librosa STFT | Do all analysis upfront, store arrays in RAM — not per-frame |
| Video export | OpenCV per-frame write | Use `mp4v` codec, avoid PNG frames on disk |
| Preview mode | 1080p at 30fps | Render at 720p for preview with `--width 1280 --height 720` |

For 1920×1080 at 30fps, expect **~15–25 minutes** export time on a mid-range CPU for a 3-min track. Most of that is bloom + video write.

---

## Execution Flow (Full Pipeline)

```
1. Parse CLI args
2. Load config.yaml
3. AudioAnalyzer(input_file, fps)
   └── librosa.load → waveform
   └── STFT → magnitude spectrogram
   └── Extract bass/mids/highs/rms/onset per frame
   └── Apply exponential smoothing
   └── Beat track → boolean array
4. World.__init__()
   └── Create N Strand objects, evenly spaced angles
   └── Each strand: N Particle chain, root pinned
5. Canvas.__init__()
   └── pygame.Surface (offscreen)
6. Recorder.__init__() [if not preview]
   └── cv2.VideoWriter to temp file
7. FOR each frame in 0..n_frames:
   a. features = analyzer.get_frame(frame_idx)
   b. world.step(features, dt)
      └── global_hue += drift
      └── FOR each strand:
          └── strand.apply_audio_force(features, dt)
          └── strand.update(dt)
              └── Verlet integrate particles
              └── Solve spring constraints × iterations
   c. canvas.draw_frame(world, features)
      └── Trail fade (numpy multiply)
      └── FOR each strand: draw Catmull-Rom curve
      └── Draw sphere with glow rings
      └── Bloom pass (scipy blur → BLEND_ADD)
   d. Recorder.write_frame(surface) OR screen.blit
8. Recorder.finalize(input_file)
   └── cv2.VideoWriter.release()
   └── moviepy: VideoFileClip + AudioFileClip → write MP4
9. Done
```

---

## Edge Cases & Gotchas

| Scenario | What happens | Fix |
|---|---|---|
| Very quiet audio | All features near 0, strands barely move | Add `min_force = 50.0` baseline force |
| Very loud bass | Forces blow particles off screen | Clamp particle positions to canvas bounds with bounce |
| Short audio (<10s) | n_frames tiny, looks fine | No issue |
| Mono vs stereo | librosa handles both, converts to mono | No issue |
| WAV with 24-bit samples | librosa handles this | No issue |
| MP3 vs FLAC vs OGG | librosa + soundfile handles all | Need `pip install soundfile` |
| n_strands=1 | Works, looks odd | Min recommended: 4 |
| Bloom on dark frame | Bloom pass cheap, no visible effect | Fine, no optimization needed |
| Preview slow (drops frames) | Physics still synced to frame count, not wall time | Preview is "best effort" — reduce resolution |
| strand segments escape canvas | Particles fly off screen | Add soft boundary: `x = clip(x, margin, W-margin)` with small restoring force |