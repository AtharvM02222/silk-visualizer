# 🌟 SILK VISUALIZER

Automated beat-synced silk animations with **dynamic color transitions** 🌈

## ✨ Features
- **Dynamic Color Palettes**: Smooth transitions between multiple shades
- **Audio-Reactive**: Pulsing effects synced to beats
- **No Dependencies**: Pure Python + FFmpeg
- **High Quality**: 60fps, up to 4K resolution

## 🚀 Quick Start

```bash
# Dynamic volcano reds (audio5.mp3 from 40s)
python3 silk_visualizer.py audio5.mp3 volcano 1080p 40

# Static purple color
python3 silk_visualizer.py song.mp3 purple 1080p

# Dynamic ocean blues in 4K
python3 silk_visualizer.py track.mp3 ocean 4k
```

## 🎨 Color Options

### 🌈 Dynamic Palettes (Smooth Transitions)
- **volcano**: Dark red → Blood red → Lava orange → Ember red
- **ocean**: Deep blue → Cyan wave → Teal
- **sunset**: Purple → Pink → Orange

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
