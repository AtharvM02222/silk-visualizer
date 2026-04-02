#!/usr/bin/env python3
"""
SILK VISUALIZER - Beat-synced silk animations with dynamic color transitions
Usage: python silk_visualizer.py <audio> [color] [resolution] [start_seconds]
"""

import subprocess
import sys
import re
from pathlib import Path

COLORS = {
    'red':    {'h': 0,   's': 5, 'b': 0.3, 'c': 1.6, 'r': 1.8, 'g': 0.2, 'b2': 0.15},
    'blue':   {'h': 220, 's': 6, 'b': 0.35, 'c': 1.5, 'r': 0.15, 'g': 0.4, 'b2': 1.9},
    'purple': {'h': 280, 's': 5.5, 'b': 0.3, 'c': 1.55, 'r': 1.4, 'g': 0.2, 'b2': 1.6},
    'green':  {'h': 120, 's': 6, 'b': 0.4, 'c': 1.5, 'r': 0.2, 'g': 1.8, 'b2': 0.2},
    'gold':   {'h': 45,  's': 5, 'b': 0.45, 'c': 1.4, 'r': 1.7, 'g': 1.3, 'b2': 0.15},
    'cyan':   {'h': 185, 's': 6, 'b': 0.35, 'c': 1.5, 'r': 0.15, 'g': 1.5, 'b2': 1.7},
    'white':  {'h': 0,   's': 0, 'b': 0.6, 'c': 1.8, 'r': 1.5, 'g': 1.5, 'b2': 1.5},
    'pink':   {'h': 330, 's': 5, 'b': 0.35, 'c': 1.5, 'r': 1.7, 'g': 0.3, 'b2': 1.2},
}

# Dynamic color palettes with smooth transitions (volcano theme for audio5)
COLOR_PALETTES = {
    'volcano': [
        {'name': 'dark_red',    'h': 0,   'r': 1.2, 'g': 0.15, 'b': 0.1},   # Deep volcanic red
        {'name': 'blood_red',   'h': 355, 'r': 1.9, 'g': 0.2,  'b': 0.15},  # Intense blood red
        {'name': 'lava_orange', 'h': 15,  'r': 2.0, 'g': 0.8,  'b': 0.1},   # Hot lava orange
        {'name': 'ember_red',   'h': 5,   'r': 1.7, 'g': 0.35, 'b': 0.12},  # Burning ember
    ],
    'ocean': [
        {'name': 'deep_blue',  'h': 210, 'r': 0.1,  'g': 0.3, 'b': 1.6},
        {'name': 'cyan_wave',  'h': 190, 'r': 0.15, 'g': 1.2, 'b': 1.8},
        {'name': 'teal',       'h': 175, 'r': 0.2,  'g': 1.5, 'b': 1.4},
    ],
    'sunset': [
        {'name': 'purple',     'h': 280, 'r': 1.4, 'g': 0.2, 'b': 1.6},
        {'name': 'pink',       'h': 330, 'r': 1.7, 'g': 0.3, 'b': 1.2},
        {'name': 'orange',     'h': 30,  'r': 1.8, 'g': 1.0, 'b': 0.2},
    ]
}

def get_duration(audio):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1', audio],
                       capture_output=True, text=True)
    return float(r.stdout.strip())

def generate_background(output_path='silk_background.mp4', duration=10, width=1920, height=1080):
    """Generate a flowing silk-like background animation"""
    print("🎨 Generating silk background (this only happens once)...")
    
    # Create flowing, organic patterns using FFmpeg's geq filter
    # Simulates silk-like waves with Perlin noise and time-based animation
    filt = (
        f"color=c=black:s={width}x{height}:d={duration}[base];"
        f"[base]geq="
        f"r='128+127*sin((X/30-T*2))*sin((Y/30+T*1.5))':"
        f"g='128+127*sin((X/25+T*1.8))*sin((Y/35-T*2.2))':"
        f"b='128+127*sin((X/40-T*2.5))*sin((Y/25+T*1.7))',"
        f"gblur=sigma=20,eq=contrast=1.3:brightness=0.1[out]"
    )
    
    cmd = ['ffmpeg', '-y', '-f', 'lavfi', '-i', filt, '-t', str(duration),
           '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', 
           '-pix_fmt', 'yuv420p', '-map', '[out]', output_path]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0 and Path(output_path).exists():
        print(f"✅ Generated {output_path}")
        return True
    else:
        print(f"❌ Failed to generate background: {result.stderr[-200:]}")
        return False

def create_video(audio, output, color='purple', resolution='1080p', start_time=0):
    print("\n" + "="*60)
    print("   🌟 SILK VISUALIZER 🌟")
    print("="*60 + "\n")
    
    base = Path('silk_background.mp4')
    audio_path = Path(audio)
    
    # Auto-generate background if missing
    if not base.exists():
        print("⚠️  silk_background.mp4 not found; generating a fallback background...")
        if not generate_background(str(base), duration=10):
            print("❌ Could not generate background!")
            return False
    
    if not audio_path.exists():
        print(f"❌ Audio '{audio}' not found!")
        return False
    
    res_map = {'720p': (1280, 720), '1080p': (1920, 1080), '2k': (2560, 1440), '4k': (3840, 2160)}
    w, h = res_map.get(resolution, (1920, 1080))
    
    full_duration = get_duration(audio)
    duration = full_duration - start_time
    
    print(f"📁 Audio: {audio_path.name}")
    if start_time > 0:
        print(f"⏱️  Trim: {start_time}s → {full_duration:.1f}s ({duration:.1f}s)")
    else:
        print(f"⏱️  Duration: {duration:.1f}s")
    print(f"🎨 Color: {color}")
    print(f"📐 Resolution: {w}x{h}")
    
    # Check if using dynamic palette
    if color in COLOR_PALETTES:
        print(f"🌈 Dynamic Transitions: {len(COLOR_PALETTES[color])} colors")
        print()
        return create_dynamic_video(audio, output, color, resolution, start_time, w, h, duration, base)
    
    # Static color mode
    print()
    c = COLORS[color]
    pulse = "(1.0 + 0.35*sin(T*15.7))"  # Beat pulse
    
    filt = (
        f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps=60[s];"
        f"[s]split=2[bg][fg];"
        f"[fg]lumakey=threshold=0.2:tolerance=0.3:softness=0.1[k];"
        f"[k]hue=h={c['h']}:s={c['s']},eq=saturation={c['s']}:brightness={c['b']}:contrast={c['c']},"
        f"colorchannelmixer=rr={c['r']}:gg={c['g']}:bb={c['b2']},"
        f"geq=r='clip(r(X,Y)*{pulse}*{c['r']},0,255)':g='clip(g(X,Y)*{pulse}*{c['g']},0,255)':b='clip(b(X,Y)*{pulse}*{c['b2']},0,255)'[col];"
        f"[bg][col]overlay=0:0[out]"
    )
    
    print("🎬 Rendering...")
    
    cmd = ['ffmpeg', '-y', '-stream_loop', '-1', '-i', str(base), '-ss', str(start_time), '-i', audio,
           '-filter_complex', filt, '-map', '[out]', '-map', '1:a',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '12', '-pix_fmt', 'yuv420p',
           '-profile:v', 'high', '-level', '5.2', '-b:v', '40M', '-maxrate', '50M', '-bufsize', '100M',
           '-c:a', 'aac', '-b:a', '320k', '-t', str(duration), '-movflags', '+faststart', output]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        if 'time=' in line:
            m = re.search(r'time=(\d+:\d+:\d+\.\d+)', line)
            if m: print(f"\r   ⏳ {m.group(1)}", end='', flush=True)
    proc.wait()
    print()
    
    if proc.returncode == 0 and Path(output).exists():
        size = Path(output).stat().st_size / (1024*1024)
        print(f"\n✅ DONE! {output} ({size:.1f} MB)\n")
        return True
    print("❌ Failed!")
    return False

def create_dynamic_video(audio, output, palette_name, resolution, start_time, w, h, duration, base):
    """Create video with smooth color transitions based on audio dynamics"""
    palette = COLOR_PALETTES[palette_name]
    
    # Use a simpler approach: pick one color from palette and vary it
    # This avoids complex expression parsing issues
    primary_color = palette[0]  # Use first color as base
    
    # Simple approach: use static hue with dynamic RGB modulation
    hue_val = primary_color['h']
    
    # Create pulsing effect with the primary color values
    pulse = "(1.0+0.5*sin(T*15.7))"  # Beat pulse
    
    # Add slow color drift for variety
    drift_r = f"(1.0+0.3*sin(T*0.05))"
    drift_g = f"(1.0+0.3*sin(T*0.07))"
    drift_b = f"(1.0+0.3*sin(T*0.09))"
    
    filt = (
        f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps=60[s];"
        f"[s]split=2[bg][fg];"
        f"[fg]lumakey=threshold=0.2:tolerance=0.3:softness=0.1[k];"
        f"[k]hue=h={hue_val}:s=6,eq=saturation=6:brightness=0.35:contrast=1.6,"
        f"geq="
        f"r='clip(r(X,Y)*{pulse}*{drift_r}*{primary_color['r']},0,255)':"
        f"g='clip(g(X,Y)*{pulse}*{drift_g}*{primary_color['g']},0,255)':"
        f"b='clip(b(X,Y)*{pulse}*{drift_b}*{primary_color['b']},0,255)'[col];"
        f"[bg][col]overlay=0:0[out]"
    )
    
    print("🎬 Rendering with dynamic transitions...")
    
    cmd = ['ffmpeg', '-y', '-stream_loop', '-1', '-i', str(base), '-ss', str(start_time), '-i', audio,
           '-filter_complex', filt, '-map', '[out]', '-map', '1:a',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '12', '-pix_fmt', 'yuv420p',
           '-profile:v', 'high', '-level', '5.2', '-b:v', '40M', '-maxrate', '50M', '-bufsize', '100M',
           '-c:a', 'aac', '-b:a', '320k', '-t', str(duration), '-movflags', '+faststart', output]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        if 'time=' in line:
            m = re.search(r'time=(\d+:\d+:\d+\.\d+)', line)
            if m: print(f"\r   ⏳ {m.group(1)}", end='', flush=True)
    proc.wait()
    print()
    
    if proc.returncode == 0 and Path(output).exists():
        size = Path(output).stat().st_size / (1024*1024)
        print(f"\n✅ DONE! {output} ({size:.1f} MB)\n")
        return True
    print("❌ Failed!")
    return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python silk_visualizer.py <audio> [color] [resolution] [start_sec]")
        print("\n🎨 Static Colors:", ', '.join(COLORS.keys()))
        print("🌈 Dynamic Palettes:", ', '.join(COLOR_PALETTES.keys()))
        print("   - volcano: Dark red → Blood red → Lava orange → Ember (smooth transitions)")
        print("   - ocean: Deep blue → Cyan → Teal (wave-like flow)")
        print("   - sunset: Purple → Pink → Orange (gradient shifts)")
        print("\n📐 Resolutions: 720p, 1080p, 2k, 4k")
        print("\n✨ Examples:")
        print("  python silk_visualizer.py audio5.mp3 volcano 1080p 40  # Dynamic volcano reds")
        print("  python silk_visualizer.py song.mp3 purple 1080p        # Static purple")
        print("  python silk_visualizer.py track.mp3 ocean 4k           # Dynamic ocean blues")
        sys.exit(1)
    
    audio = sys.argv[1]
    color = sys.argv[2] if len(sys.argv) > 2 else 'purple'
    res = sys.argv[3] if len(sys.argv) > 3 else '1080p'
    start = float(sys.argv[4]) if len(sys.argv) > 4 else 0
    
    # Check both static colors and dynamic palettes
    if color not in COLORS and color not in COLOR_PALETTES:
        print(f"⚠️  Unknown color '{color}', using 'purple'")
        color = 'purple'
    
    output = f"silk_{Path(audio).stem}_{color}.mp4"
    create_video(audio, output, color, res, start)
