# 🔮 SILK FLUID ORB VISUALIZER

Beautiful fluid silk/lava orb visualization with organic flowing patterns and audio-reactive effects.

## ✨ Features

- **🎵 Audio-Reactive**: Bass → pulses, Mids → flow speed, Highs → shimmer
- **🚀 GPU Acceleration**: Optional CuPy support for faster rendering
- **⚡ Multiprocessing**: Parallel frame generation (5-8x faster)
- **💾 Checkpoint/Resume**: Save progress and resume interrupted renders
- **✨ Particle System**: Beat-reactive floating particles
- **🎨 Custom Colors**: Load your own color palettes from JSON
- **🎼 BPM Detection**: Automatic tempo detection for beat-aligned effects
- **📊 Spectral Analysis**: Brightness and spectral centroid tracking

## 🚀 Quick Start

```bash
# Install dependencies
pip install numpy pillow scipy tqdm

# Optional: GPU acceleration (NVIDIA)
pip install cupy-cuda12x

# Optional: Enhanced audio analysis
pip install librosa

# Basic usage
python silk_visualizer.py song.mp3

# Quick preview (faster, lower quality)
python silk_visualizer.py song.mp3 --preview

# 2K resolution with lava colors
python silk_visualizer.py song.mp3 -r 2k -c lava

# 4K with GPU acceleration
python silk_visualizer.py song.mp3 -r 4k --gpu
```

## 📖 CLI Options

```
usage: silk_visualizer [-h] [-o OUTPUT] [-r {720p,1080p,2k,4k,4k+}]
                       [--fps FPS] [-c COLOR] [-s SENSITIVITY]
                       [-w WORKERS] [--gpu] [--no-particles] [--preview]
                       [--resume] [--batch] [--start START]
                       [--config CONFIG] [--colors-file FILE]
                       audio [audio ...]

Options:
  -o, --output          Output video path
  -r, --resolution      720p, 1080p, 2k, 4k, 4k+ (default: 1080p)
  --fps                 Frame rate (default: 60)
  -c, --color           Color scheme (see below)
  -s, --sensitivity     Audio reactivity 0.5-2.0 (default: 1.0)
  -w, --workers         Parallel workers (default: auto)
  --gpu                 Enable GPU acceleration
  --no-particles        Disable particle system
  --preview             Quick preview mode
  --resume              Resume from checkpoint
  --batch               Process multiple files
  --colors-file         Custom colors JSON file
  --list-colors         Show available colors
```

## 📐 Resolutions

| Preset | Size | Use Case |
|--------|------|----------|
| `720p` | 720×800 | Fast preview |
| `1080p` | 1080×1200 | Default |
| `2k` | 1440×1600 | Recommended |
| `4k` | 2160×2400 | High quality |
| `4k+` | 3456×3840 | Maximum |

## 🎨 Color Schemes

**Built-in:**
- `purple` - Deep violet/magenta glow
- `lava` - Orange/red molten flow
- `ocean` - Deep blue/cyan
- `golden` - Warm amber/gold
- `emerald` - Green/teal
- `volcano` - Deep red to orange
- `cyber` - Cyan/pink cyberpunk
- `sunset` - Orange to pink gradient

**Custom Colors:**
Create a `colors.json` file:
```json
{
    "neon": {
        "dark": [5, 5, 20],
        "mid": [50, 0, 150],
        "bright": [200, 0, 255],
        "hot": [255, 100, 255],
        "glow": [180, 0, 255]
    }
}
```

Then use: `python silk_visualizer.py song.mp3 --colors-file colors.json -c neon`

## 🔧 Advanced Usage

```bash
# Resume interrupted render
python silk_visualizer.py song.mp3 --resume

# Batch process multiple files
python silk_visualizer.py *.mp3 --batch -r 2k -c lava

# High sensitivity for quiet audio
python silk_visualizer.py ambient.mp3 -s 1.5

# Low sensitivity for loud audio
python silk_visualizer.py metal.mp3 -s 0.7

# Start at specific time
python silk_visualizer.py song.mp3 --start 30
```

## ⚡ Performance Tips

1. **Use `--preview`** for quick tests before full renders
2. **Enable GPU** with `--gpu` if you have NVIDIA CUDA
3. **Adjust workers** with `-w 4` to match your CPU cores
4. **Lower resolution** first, then upscale if needed
5. **Use checkpoints** with `--resume` for long renders

## ☁️ Run in Codespaces

Rendering is CPU-intensive. Use GitHub Codespaces for cloud processing:

1. Go to https://github.com/AtharvM02222/silk-visualizer
2. Click **Code** → **Codespaces** → **Create codespace**
3. Upload your audio file
4. Run: `python silk_visualizer.py audio.mp3 -r 2k -c volcano`
5. Download the result!

## 🎬 How It Works

1. **Audio Analysis**: FFT-based spectral analysis extracts bass, mids, highs
2. **BPM Detection**: Automatic tempo detection (uses librosa if available)
3. **Frame Generation**: Perlin noise + domain warping creates fluid patterns
4. **Particle System**: Beat-reactive particles spawn on transients
5. **Multiprocessing**: Frames rendered in parallel across CPU cores
6. **Video Encoding**: FFmpeg combines frames with original audio

## 📦 Requirements

- Python 3.7+
- FFmpeg (pre-installed in Codespaces)
- numpy, pillow, scipy, tqdm

**Optional:**
- `cupy` - GPU acceleration (NVIDIA CUDA)
- `librosa` - Enhanced BPM detection

## 🎯 Perfect For

- Music visualizations for YouTube
- VJ loops for live performances
- Social media content (TikTok, Reels)
- Background visuals for streams
- Album artwork animations

## 📝 License

MIT License - feel free to use and modify!
