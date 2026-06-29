# Silk Audio Visualizer

A physics-based audio visualization tool that generates silk-like cloth simulations synchronized to music. Feed it an audio file, get back a stunning MP4 video — or watch it live in a real-time preview window.

## Features

- **Physics-driven silk strands** — particle chains using Verlet integration and spring constraints create natural cloth-like motion
- **Audio-reactive** — bass, mids, highs, and beat onsets each drive different visual parameters
- **Glowing center sphere** — pulses with the bass and radiates silk strands symmetrically
- **Bloom post-processing** — Gaussian blur glow pass on bright pixels for a luminous, dreamy look
- **Motion blur trails** — previous frames fade out smoothly, giving depth and flow
- **Color palettes** — warm, cool, neon, and mono presets with slow hue drift over time
- **Real-time preview** — interactive pygame window for experimenting with settings
- **MP4 export** — H.264 video with the original audio muxed back in via ffmpeg
- **YAML configuration** — every visual and physics parameter is tunable without touching code

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/silk-visualizer.git
cd silk-visualizer
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

The dependencies are:

```
librosa>=0.10.0       # Audio analysis, FFT, beat tracking
numpy>=1.24.0         # Math everywhere
pygame>=2.5.0         # Real-time rendering and surface drawing
opencv-python>=4.8.0  # Video frame writing
scipy>=1.11.0         # Gaussian blur for bloom effects
moviepy>=1.0.3        # Mux audio back into exported video
pyyaml>=6.0           # Config file parsing
click>=8.1.0          # CLI interface
```

### 3. Install ffmpeg

moviepy requires ffmpeg to mux audio into the exported video.

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu / Debian:**
```bash
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your PATH.

Alternatively, install via pip:
```bash
pip install imageio[ffmpeg]
```

## Usage

### Basic export

```bash
python main.py --input track.mp3
```

Produces `silk_output.mp4` in the current directory with the default 1920×1080 @ 30fps settings.

### Specify output file

```bash
python main.py --input track.mp3 --output my_video.mp4
```

### Preview mode (real-time window, no export)

```bash
python main.py --input track.mp3 --preview
```

Opens a pygame window. Close it to stop. Useful for tweaking config values before committing to a full export.

### Custom resolution and framerate

```bash
python main.py --input track.mp3 --output square.mp4 --width 1080 --height 1080 --fps 60
```

### Custom config file

```bash
python main.py --input track.mp3 --config my_config.yaml
```

### All options

```
Usage: python main.py [OPTIONS]

Options:
  -i, --input  PATH     Input audio file (required)
  -o, --output PATH     Output video file [default: silk_output.mp4]
  -c, --config PATH     Config YAML path [default: config.yaml]
  -p, --preview         Show real-time pygame window instead of exporting
      --width  INT      Output width in pixels [default: 1920]
      --height INT      Output height in pixels [default: 1080]
      --fps    INT      Frames per second [default: 30]
      --help            Show this message and exit
```

Supported audio formats: MP3, WAV, FLAC, OGG.

## Configuration Reference

All parameters live in `config.yaml`. Missing parameters fall back to the defaults listed below.

### Canvas

| Parameter | Default | Description |
|---|---|---|
| `background_color` | `[5, 5, 10]` | RGB background color. Near-black with a subtle blue tint. |

### Strand Physics

| Parameter | Default | Range | Description |
|---|---|---|---|
| `n_strands` | `10` | 4–12 | Number of silk strands radiating from the center. |
| `n_particles` | `24` | 15–30 | Particles per strand (chain length). More = longer, more flexible strands. |
| `segment_length` | `9.0` | 8.0–12.0 | Rest length of each spring segment in pixels. |
| `stiffness` | `0.82` | 0.0–1.0 | Spring stiffness. Higher = stiffer cloth, less deformation. |
| `damping` | `0.975` | 0.95–0.99 | Velocity damping per frame. Lower = energy dissipates faster. |
| `constraint_iterations` | `6` | 5–8 | Spring solver iterations per frame. More = stiffer, more stable. |

### Audio-Driven Forces

| Parameter | Default | Description |
|---|---|---|
| `bass_force_mult` | `800` | Multiplier for bass energy → perpendicular wind force on strand tips. |
| `mids_force_mult` | `200` | Multiplier for mids energy → additional wind force. |
| `beat_burst_mult` | `1200` | Force magnitude on beat onset — radial burst outward from center. |
| `jitter_mult` | `60` | Standard deviation of random jitter applied to all free particles (driven by highs). |

### Center Sphere

| Parameter | Default | Range | Description |
|---|---|---|---|
| `sphere_base_radius` | `28.0` | 20–40 | Base sphere radius in pixels. |

The sphere radius pulses as: `base_radius × (1.0 + bass × 0.4 + onset × 0.2)`.

### Color

| Parameter | Default | Options | Description |
|---|---|---|---|
| `palette` | `"warm"` | `warm`, `cool`, `neon`, `mono` | Color palette preset. |
| `hue_drift_speed` | `0.0003` | 0.0001–0.0005 | How fast the global hue shifts per frame. |

Palette hue ranges:
- `warm` — reds, oranges, yellows (hue 0.0–0.2)
- `cool` — blues, cyans, purples (hue 0.5–0.7)
- `neon` — full spectrum, high saturation (>0.85)
- `mono` — all strands share the same hue

### Rendering

| Parameter | Default | Range | Description |
|---|---|---|---|
| `trail_decay` | `0.88` | 0.80–0.95 | Pixel fade multiplier per frame. Lower = longer trails. |
| `strand_base_width` | `3` | 2–5 | Line width at the strand root in pixels. Tapers to 1px at the tip. |
| `bloom_threshold` | `160` | 140–200 | Pixel brightness (0–255) above which bloom glow is applied. |
| `bloom_blur` | `18` | 10–25 | Gaussian blur radius for the bloom pass in pixels. |

### Audio Smoothing

Exponential moving average (EMA) coefficients applied to each frequency band. Higher values = more smoothing, slower response.

| Parameter | Default | Description |
|---|---|---|
| `smoothing_bass` | `0.75` | EMA coefficient for bass energy. |
| `smoothing_mids` | `0.65` | EMA coefficient for mids energy. |
| `smoothing_highs` | `0.55` | EMA coefficient for highs energy. |
| `smoothing_rms` | `0.70` | EMA coefficient for RMS amplitude. |

### Example config.yaml

```yaml
# Canvas
background_color: [5, 5, 10]

# Strand physics
n_strands: 10
n_particles: 24
segment_length: 9.0
stiffness: 0.82
damping: 0.975
constraint_iterations: 6

# Audio-driven forces
bass_force_mult: 800
mids_force_mult: 200
beat_burst_mult: 1200
jitter_mult: 60

# Center sphere
sphere_base_radius: 28.0

# Color
palette: "warm"
hue_drift_speed: 0.0003

# Rendering
trail_decay: 0.88
strand_base_width: 3
bloom_threshold: 160
bloom_blur: 18

# Audio smoothing
smoothing_bass: 0.75
smoothing_mids: 0.65
smoothing_highs: 0.55
smoothing_rms: 0.70
```

## Project Structure

```
silk-visualizer/
├── main.py                # CLI entry point and main render loop
├── config.yaml            # Default configuration
├── requirements.txt       # Python dependencies
├── audio/
│   └── analyzer.py        # AudioAnalyzer — FFT, mel spectrogram, beat detection
├── physics/
│   ├── particle.py        # Particle — Verlet integration, spring constraints
│   ├── strand.py          # Strand — particle chain, audio force application
│   └── world.py           # World — strand container, global state
├── render/
│   ├── canvas.py          # Canvas — drawing pipeline (trails, strands, sphere)
│   └── effects.py         # apply_bloom — Gaussian glow post-processing
├── export/
│   └── recorder.py        # Recorder — OpenCV video writer, moviepy audio mux
└── tests/
    ├── test_audio.py
    ├── test_config.py
    ├── test_strand.py
    └── test_world.py
```

## Running Tests

```bash
pytest tests/
```

To run with verbose output:

```bash
pytest tests/ -v
```

## Troubleshooting

### `[ERROR] Audio file not found: track.mp3`

The path you passed to `--input` doesn't exist. Check the file path and try again with an absolute path if needed:

```bash
python main.py --input /full/path/to/track.mp3
```

### `[ERROR] moviepy not installed or ffmpeg not found`

ffmpeg is missing. Install it:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Or via pip
pip install imageio[ffmpeg]
```

### `[ERROR] Configuration file not found: config.yaml`

The default `config.yaml` is expected in the current working directory. Either run the script from the project root, or pass an explicit path:

```bash
python main.py --input track.mp3 --config /path/to/config.yaml
```

### `[ERROR] Invalid YAML syntax in configuration file`

Your config file has a syntax error. Common causes:
- Missing colon after a key (`n_strands 10` → should be `n_strands: 10`)
- Incorrect indentation
- Tabs instead of spaces

Validate your YAML at [yamllint.com](https://www.yamllint.com/) or run:

```bash
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

### `[ERROR] Failed to initialize video writer`

OpenCV couldn't create the output file. Check that:
- The output directory exists (`mkdir -p output/`)
- You have write permissions to the output path
- The output path doesn't contain special characters

### Preview window is very slow / choppy

Preview mode renders in real time. On slower machines or at high resolutions, frames may drop. Try reducing resolution:

```bash
python main.py --input track.mp3 --preview --width 960 --height 540
```

### Export takes a very long time

A 3-minute track at 1920×1080 @ 30fps is ~5400 frames. Expected export time is under 25 minutes on a mid-range CPU. The bloom pass is the main bottleneck — reduce `bloom_blur` or increase `bloom_threshold` to speed things up:

```yaml
bloom_threshold: 200   # fewer pixels get bloom
bloom_blur: 10         # smaller blur radius
```

### Strands look stiff / not flowing

Lower the stiffness and increase damping for more fluid motion:

```yaml
stiffness: 0.5
damping: 0.99
constraint_iterations: 4
```

### Strands barely move on quiet sections

The minimum force baseline of 50.0 keeps strands moving even during silence. If you want more motion on quiet sections, increase the force multipliers:

```yaml
bass_force_mult: 1200
mids_force_mult: 400
jitter_mult: 100
```

## How It Works

The system follows a pipeline architecture:

1. **Audio analysis** — librosa loads the audio file, computes a mel spectrogram via STFT, and extracts per-frame bass/mids/highs energy, RMS amplitude, onset strength, and beat flags. All analysis is done upfront during initialization.

2. **Physics simulation** — N silk strands radiate from the canvas center. Each strand is a chain of particles connected by springs. Each frame, audio features are translated into forces: bass drives large perpendicular wind forces on the strand tips, mids add moderate wind, highs add random jitter, and beat onsets trigger radial burst forces. Verlet integration updates particle positions, and iterative spring constraints maintain strand structure.

3. **Rendering** — Each frame: existing pixels are multiplied by the trail decay coefficient (motion blur), strands are drawn as smooth Catmull-Rom splines with tapering width, the center sphere is drawn with concentric glow rings, and a bloom pass extracts bright pixels, blurs them, and composites them back additively.

4. **Export** — Frames are written to a temporary MP4 via OpenCV, then moviepy muxes the original audio into the final output file.
