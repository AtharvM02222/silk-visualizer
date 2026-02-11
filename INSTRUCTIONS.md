# Red Silk Visualizer - Final Instructions

## Quick Start

1. **Open the visualizer:**
   ```bash
   open red-silk-final.html
   ```
   (Or double-click the file in Finder)

2. **The visualizer will auto-load audio2.mp3**

3. **Click "START RECORDING"** to begin recording

4. **The recording will:**
   - Automatically start playing audio2.mp3
   - Record for the full 18.19 seconds
   - Apply 5 different red shades with smooth transitions
   - Match beats throughout the entire song
   - Only color the sphere (background stays dark)
   - Record at 1728x1920 portrait, 60fps, maximum quality

5. **Click "STOP RECORDING"** when done (or it auto-stops at end)

6. **The browser will download:** `red-silk-1728x1920-60fps.webm`

7. **Convert to MP4** (for compatibility):
   ```bash
   ffmpeg -i red-silk-1728x1920-60fps.webm \
     -c:v libx264 -preset slow -crf 15 \
     -c:a aac -b:a 320k \
     red_silk_final.mp4
   ```

## Features

✅ **5 Red Shades:**
- Bright Red
- Deep Red
- Vivid Red
- Dark Red
- Pure Red

✅ **Beat-Matched:** Color changes sync with music beats throughout full duration

✅ **High Quality:** 
- 1728x1920 portrait resolution
- 60fps smooth motion
- 50Mbps bitrate (maximum quality)
- No blurriness

✅ **Sphere Only:** Only the silk sphere turns red, background stays dark

✅ **Smooth Transitions:** Seamless color transitions and animations between shades

## Troubleshooting

**If recording is blurry:**
- Make sure you're using a modern browser (Chrome/Edge recommended)
- Close other tabs to free up resources
- The WebM output is high quality - blurriness may occur during MP4 conversion if CRF is too high

**If colors aren't red enough:**
- Edit `red-silk-final.html`
- Find the `redShades` array
- Increase the `sat` (saturation) values

**If beat matching isn't working:**
- Make sure audio2.mp3 is in the same folder
- Check browser console for errors
- Try refreshing the page

## Quality Comparison

Reference videos (EOBS.mp4, AW!.mp4, AW@.mp4):
- Resolution: 3456x3840
- Frame rate: 60fps
- Bitrate: ~57Mbps

This output:
- Resolution: 1728x1920 (half size for faster processing)
- Frame rate: 60fps
- Bitrate: ~50Mbps (maximum browser recording quality)

To match reference quality exactly, you can upscale after recording:
```bash
ffmpeg -i red_silk_final.mp4 \
  -vf scale=3456:3840:flags=lanczos \
  -c:v libx264 -preset slow -crf 12 \
  -c:a copy \
  red_silk_final_4k.mp4
```
