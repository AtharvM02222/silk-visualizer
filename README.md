# Silk Visualizer

Audio-reactive silk cloth sphere visualizer. Give it an MP3 and it produces
an MP4 video of a glowing, physics-simulated silk sphere that reacts to your
music — matching the aesthetic of [Resonation's "Silk"](https://resonation.io/visualizers/silk/).

![Silk sphere screenshot](eg/silk_background.mp4)

---

## How it works

`main.mp4` contains a pre-rendered physics-based silk cloth simulation.
The script:

1. **Loops** `main.mp4` using a seamless ping-pong (forward ↔ reverse) sequence
2. **Analyzes** your audio (RMS energy + onset/beat detection via ffmpeg + scipy)
3. **Per-frame effects** synchronized to the music:
   - **Beat flash** — brightness spike on transients
   - **Energy lift** — overall brightness tracks loudness
   - **Saturation boost** — colours become richer at loud passages
   - **Warm tint** — amber colour push on beats
   - **Zoom pulse** — subtle centre-zoom on strong beats
   - **Speed variation** — cloth animation speeds up during loud parts, slows during quiet

---

## Requirements

| Dependency | Install |
|------------|---------|
| Python 3.9+ | — |
| numpy | `pip3 install numpy` |
| scipy | `pip3 install scipy` |
| Pillow | `pip3 install pillow` |
| **ffmpeg** | `brew install ffmpeg` (macOS) |

All Python deps at once:
```bash
pip3 install numpy scipy pillow
```

---

## Usage

```bash
# Basic — uses main.mp4 as silk source, outputs output.mp4
python3 silk_visualizer.py your_song.mp3

# Custom output filename
python3 silk_visualizer.py your_song.mp3 my_video.mp4

# Custom output resolution (letterboxed to preserve silk aspect ratio)
python3 silk_visualizer.py your_song.mp3 my_video.mp4 --width 1920 --height 1080

# Custom silk source video
python3 silk_visualizer.py your_song.mp3 out.mp4 --source path/to/other_silk.mp4

# All options
python3 silk_visualizer.py --help
```

---

## Performance

| Resolution | Approx. render speed | 3-min song |
|------------|----------------------|------------|
| 1896×800 (native) | ~7 fps | ~13 min |
| 1920×1080 | ~5 fps | ~18 min |
| 960×540 | ~20 fps | ~4.5 min |

Render speed depends on CPU. The hot loop is fully vectorised numpy — no GPU needed.

---

## Project structure

```
silk-visualizer/
├── silk_visualizer.py   # Main script
├── main.mp4             # Silk cloth source animation (3.7s loop)
├── requirements.txt     # Python dependencies
└── eg/                  # Example outputs
```
