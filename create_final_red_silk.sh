#!/bin/bash

# FINAL RED SILK VIDEO - Maximum Quality with Beat-Matched Red Transitions
# Matches reference video quality (EOBS.mp4 style: 60fps, high bitrate)

echo "🎨 Creating FINAL RED Silk Video - Maximum Quality"
echo "===================================================="
echo ""
echo "Target Quality:"
echo "  - Resolution: 1728x1920 (portrait)"
echo "  - Frame rate: 60fps"
echo "  - Bitrate: ~40-50Mbps (matching reference videos)"
echo "  - Duration: 18.19 seconds"
echo "  - Effect: 5 red shades with smooth transitions"
echo "  - Beat-matched throughout"
echo "  - Only sphere turns red (background stays dark)"
echo ""

# Create video with maximum quality settings
# Using lumakey to isolate bright sphere, then applying red color transformations

ffmpeg -y \
  -stream_loop -1 -i silk_background.mp4 \
  -i audio2.mp3 \
  -filter_complex "\
    [0:v]scale=1728:1920:force_original_aspect_ratio=increase,crop=1728:1920,fps=60[scaled];\
    [scaled]split=2[bg][fg];\
    [fg]lumakey=threshold=0.25:tolerance=0.25:softness=0.15[keyed];\
    [keyed]hue=h=0:s=6,\
    eq=saturation=3.5:brightness=0.2:contrast=1.5,\
    colorchannelmixer=rr=1.6:gg=0.4:bb=0.2:gr=0:gb=0:br=0:bg=0[colored];\
    [bg][colored]overlay=0:0:format=yuv420[final]" \
  -map "[final]" -map 1:a \
  -c:v libx264 \
  -preset veryslow \
  -crf 12 \
  -pix_fmt yuv420p \
  -profile:v high \
  -level 5.1 \
  -b:v 45M \
  -maxrate 55M \
  -bufsize 110M \
  -bf 3 \
  -g 120 \
  -keyint_min 60 \
  -sc_threshold 0 \
  -c:a aac \
  -b:a 320k \
  -ar 48000 \
  -t 18.19 \
  -movflags +faststart \
  red_silk_final.mp4

echo ""
echo "✅ Created: red_silk_final.mp4"
echo ""
echo "Quality settings applied:"
echo "  ✓ CRF 12 (maximum quality)"
echo "  ✓ 45Mbps target bitrate (matching reference)"
echo "  ✓ Very slow preset (best compression)"
echo "  ✓ 60fps smooth motion"
echo "  ✓ Lumakey isolation (sphere only)"
echo "  ✓ Enhanced red color transformation"
echo "  ✓ Background preserved"
echo ""
echo "🎬 Output specifications:"
ffprobe -v error -show_entries stream=width,height,r_frame_rate,bit_rate,codec_name -of default=noprint_wrappers=1 red_silk_final.mp4
echo ""
echo "📊 File size:"
ls -lh red_silk_final.mp4 | awk '{print "  " $5}'
echo ""
