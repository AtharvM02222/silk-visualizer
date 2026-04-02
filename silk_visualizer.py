#!/usr/bin/env python3
"""
SILK VISUALIZER - Beat-synced silk animations
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

def get_duration(audio):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1', audio],
                       capture_output=True, text=True)
    return float(r.stdout.strip())

def create_video(audio, output, color='purple', resolution='1080p', start_time=0):
    print("\n" + "="*60)
    print("   🌟 SILK VISUALIZER 🌟")
    print("="*60 + "\n")
    
    base = Path('silk_background.mp4')
    audio_path = Path(audio)
    
    if not base.exists():
        print("❌ silk_background.mp4 not found!")
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

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python silk_visualizer.py <audio> [color] [resolution] [start_sec]")
        print("Colors:", ', '.join(COLORS.keys()))
        print("Resolutions: 720p, 1080p, 2k, 4k")
        print("\nExamples:")
        print("  python silk_visualizer.py song.mp3 purple 1080p")
        print("  python silk_visualizer.py audio.mp3 blue 4k 40  # Start at 40s")
        sys.exit(1)
    
    audio = sys.argv[1]
    color = sys.argv[2] if len(sys.argv) > 2 else 'purple'
    res = sys.argv[3] if len(sys.argv) > 3 else '1080p'
    start = float(sys.argv[4]) if len(sys.argv) > 4 else 0
    
    if color not in COLORS:
        color = 'purple'
    
    output = f"silk_{Path(audio).stem}_{color}.mp4"
    create_video(audio, output, color, res, start)
