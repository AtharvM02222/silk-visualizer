# 🔮 SILK FLUID ORB VISUALIZER

Beautiful fluid silk/lava orb visualization with organic flowing patterns.

## ✨ Features
- **Realistic Fluid Simulation**: Domain warping + Perlin noise for organic flow
- **Glowing Orb Effect**: Centered sphere with soft edge glow
- **Multiple Color Schemes**: Purple, Lava, Ocean, Golden, Emerald
- **High Quality**: Up to 4K+ resolution, 60fps

## 🚀 Quick Start

```bash
# Install dependencies
pip install numpy pillow

# Basic usage (1080p, 30fps, purple)
python silk_visualizer.py audio.mp3

# 2K resolution with lava colors
python silk_visualizer.py audio.mp3 2k 30 lava

# 4K at 60fps
python silk_visualizer.py audio.mp3 4k 60 purple
```

## 📐 Resolutions
- `720p` - 720x800 (fast preview)
- `1080p` - 1080x1200 (default)
- `2k` - 1440x1600 (recommended)
- `4k` - 2160x2400 (high quality)
- `4k+` - 3456x3840 (maximum)

## 🎨 Color Schemes
- **purple**: Deep violet/magenta glow
- **lava**: Orange/red molten flow
- **ocean**: Deep blue/cyan
- **golden**: Warm amber/gold
- **emerald**: Green/teal

### 🎯 Static Colors
- red, blue, purple, green, gold, cyan, white, pink

## 📐 Resolutions
- `720p`: 1280×720
- `1080p`: 1920×1080 (default)
- `2k`: 2560×1440
- `4k`: 3840×2160

## 🔧 Usage

```bash
python3 silk_visualizer.py <audio> [color/palette] [resolution] [start_seconds]
```

**Examples:**
```bash
# Full song with volcano transitions
python3 silk_visualizer.py audio5.mp3 volcano 1080p

# Start at 40 seconds
python3 silk_visualizer.py audio5.mp3 volcano 1080p 40

# 4K ocean blues
python3 silk_visualizer.py song.mp3 ocean 4k

# Static gold color
python3 silk_visualizer.py track.mp3 gold 720p
```

## ☁️ Run in Codespaces (Recommended)

Rendering is CPU-intensive. Use GitHub Codespaces for cloud processing:

1. Go to https://github.com/AtharvM02222/silk-visualizer
2. Click **Code** → **Codespaces** → **Create codespace**
3. Upload your audio file
4. Run: `python3 silk_visualizer.py audio5.mp3 volcano 1080p 40`
5. Download the result!

See `.devcontainer/README.md` for details.

## 🎬 How It Works

1. **Base Animation**: Loops `silk_background.mp4`
2. **Color Isolation**: Extracts bright silk sphere with lumakey
3. **Dynamic Transitions**: Smooth sine-wave interpolation between colors
4. **Beat Sync**: Pulsing effects at ~2.5Hz
5. **Audio Merge**: Combines with original audio

## 📦 Requirements
- Python 3.6+
- FFmpeg (pre-installed in Codespaces)

## 🎯 Perfect For
- Music visualizations
- VJ loops
- Social media content
- Background visuals for streams
