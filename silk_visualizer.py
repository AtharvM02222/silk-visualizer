#!/usr/bin/env python3
"""
SILK FLUID ORB VISUALIZER - Realistic fluid simulation inside glowing sphere
Matches the organic flowing silk/lava patterns from reference videos

Usage: python silk_visualizer.py <audio> [resolution] [fps] [color_scheme]
"""

import subprocess
import sys
import os
import re
import math
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ==================== VECTORIZED PERLIN NOISE ====================
class VectorizedNoise:
    """Fast vectorized Perlin noise using numpy"""
    
    def __init__(self, seed=0):
        np.random.seed(seed)
        self.perm = np.arange(256, dtype=np.int32)
        np.random.shuffle(self.perm)
        self.perm = np.tile(self.perm, 2)
        
        # Gradient vectors
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        self.gradients = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    
    def _fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def noise2d(self, x, y):
        """Vectorized 2D Perlin noise"""
        # Grid coordinates
        xi = np.floor(x).astype(np.int32) & 255
        yi = np.floor(y).astype(np.int32) & 255
        xf = x - np.floor(x)
        yf = y - np.floor(y)
        
        # Fade curves
        u = self._fade(xf)
        v = self._fade(yf)
        
        # Hash coordinates
        aa = self.perm[self.perm[xi] + yi] % 8
        ab = self.perm[self.perm[xi] + yi + 1] % 8
        ba = self.perm[self.perm[xi + 1] + yi] % 8
        bb = self.perm[self.perm[xi + 1] + yi + 1] % 8
        
        # Gradient dot products
        g_aa = self.gradients[aa, 0] * xf + self.gradients[aa, 1] * yf
        g_ba = self.gradients[ba, 0] * (xf - 1) + self.gradients[ba, 1] * yf
        g_ab = self.gradients[ab, 0] * xf + self.gradients[ab, 1] * (yf - 1)
        g_bb = self.gradients[bb, 0] * (xf - 1) + self.gradients[bb, 1] * (yf - 1)
        
        # Interpolate
        x1 = g_aa + u * (g_ba - g_aa)
        x2 = g_ab + u * (g_bb - g_ab)
        return x1 + v * (x2 - x1)
    
    def fbm(self, x, y, octaves=5, lacunarity=2.0, persistence=0.5):
        """Fractal Brownian Motion - layered noise"""
        value = np.zeros_like(x)
        amplitude = 1.0
        max_val = 0.0
        
        for _ in range(octaves):
            value += amplitude * self.noise2d(x, y)
            max_val += amplitude
            amplitude *= persistence
            x = x * lacunarity
            y = y * lacunarity
        
        return value / max_val
    
    def turbulence(self, x, y, octaves=5):
        """Turbulent noise - absolute values for wispy patterns"""
        value = np.zeros_like(x)
        amplitude = 1.0
        max_val = 0.0
        
        for _ in range(octaves):
            value += amplitude * np.abs(self.noise2d(x, y))
            max_val += amplitude
            amplitude *= 0.5
            x = x * 2
            y = y * 2
        
        return value / max_val


# ==================== COLOR SCHEMES ====================
COLOR_SCHEMES = {
    'purple': {  # Purple/magenta silk - matches reference (DEEPER DARKS)
        'dark': np.array([15, 0, 25]),      # Nearly black with purple tint
        'mid': np.array([120, 30, 150]),    # Deep purple
        'bright': np.array([220, 120, 240]),# Bright magenta
        'hot': np.array([255, 200, 255]),   # Hot pink/white
        'glow': (180, 80, 220),
    },
    'lava': {  # Red/orange lava flow
        'dark': np.array([20, 5, 0]),
        'mid': np.array([180, 40, 10]),
        'bright': np.array([255, 130, 40]),
        'hot': np.array([255, 230, 180]),
        'glow': (255, 80, 30),
    },
    'ocean': {  # Blue/cyan water
        'dark': np.array([0, 10, 30]),
        'mid': np.array([20, 80, 180]),
        'bright': np.array([80, 180, 255]),
        'hot': np.array([200, 240, 255]),
        'glow': (60, 150, 255),
    },
    'golden': {  # Gold/amber
        'dark': np.array([25, 15, 0]),
        'mid': np.array([180, 120, 20]),
        'bright': np.array([255, 190, 70]),
        'hot': np.array([255, 245, 200]),
        'glow': (255, 160, 40),
    },
    'emerald': {  # Green plasma
        'dark': np.array([0, 20, 10]),
        'mid': np.array([20, 150, 60]),
        'bright': np.array([80, 240, 130]),
        'hot': np.array([200, 255, 230]),
        'glow': (60, 230, 120),
    },
}


def colormap_vectorized(t, colors):
    """Vectorized color interpolation through gradient stops - HIGH CONTRAST"""
    t = np.clip(t, 0, 1)
    
    # Apply aggressive contrast curve - push darks darker, brights brighter
    t = np.power(t, 0.7)  # Lift midtones
    
    result = np.zeros((*t.shape, 3), dtype=np.float32)
    
    # Dark to mid (0 - 0.3) - extended dark range
    mask1 = t < 0.3
    ratio1 = t[mask1] / 0.3
    # Use smoothstep for the ratio to make transition smoother
    ratio1 = ratio1 * ratio1 * (3 - 2 * ratio1)
    result[mask1] = colors['dark'] + (colors['mid'] - colors['dark']) * ratio1[:, np.newaxis]
    
    # Mid to bright (0.3 - 0.6)
    mask2 = (t >= 0.3) & (t < 0.6)
    ratio2 = (t[mask2] - 0.3) / 0.3
    result[mask2] = colors['mid'] + (colors['bright'] - colors['mid']) * ratio2[:, np.newaxis]
    
    # Bright to hot (0.6 - 0.82)
    mask3 = (t >= 0.6) & (t < 0.82)
    ratio3 = (t[mask3] - 0.6) / 0.22
    result[mask3] = colors['bright'] + (colors['hot'] - colors['bright']) * ratio3[:, np.newaxis]
    
    # Hot to white (0.82 - 1.0) - dramatic hot spots
    mask4 = t >= 0.82
    ratio4 = (t[mask4] - 0.82) / 0.18
    white = np.array([255, 255, 255])
    result[mask4] = colors['hot'] + (white - colors['hot']) * ratio4[:, np.newaxis]
    
    return np.clip(result, 0, 255).astype(np.uint8)


# ==================== FRAME GENERATION (VECTORIZED) ====================
def generate_frame_vectorized(frame_num, w, h, radius, cx, cy, noise_engines, colors, fps):
    """Generate single frame using fully vectorized operations - FAST"""
    t = frame_num / fps
    
    noise1, noise2, noise3 = noise_engines
    
    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Calculate distance from center
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    
    # Create circular mask (inside orb)
    mask = dist <= radius + 5
    
    # Only compute for pixels inside the orb
    x_masked = x_grid[mask]
    y_masked = y_grid[mask]
    dist_masked = dist[mask]
    
    # Convert to polar coordinates for swirl effect
    dx = x_masked - cx
    dy = y_masked - cy
    angle = np.arctan2(dy, dx)
    r_norm = dist_masked / radius  # 0 at center, 1 at edge
    
    # DOMAIN WARPING - key to organic fluid look
    # First pass: get base warp coordinates
    warp_scale = 0.006
    warp_x = noise1.fbm(x_masked * warp_scale, y_masked * warp_scale + t * 0.08, octaves=3)
    warp_y = noise2.fbm(x_masked * warp_scale + 100, y_masked * warp_scale - t * 0.06, octaves=3)
    
    # Apply warp to coordinates (domain warping for organic flow)
    warped_x = x_masked + warp_x * 80
    warped_y = y_masked + warp_y * 80
    
    # Second pass: sample noise at warped coordinates
    scale = 0.008
    
    # Main flowing silk layer - uses warped coordinates
    silk1 = noise1.fbm(
        warped_x * scale + t * 0.12,
        warped_y * scale + t * 0.08,
        octaves=5,
        persistence=0.55
    )
    
    # Second warp pass for more complexity
    warp2_x = noise2.fbm(warped_x * scale * 0.7, warped_y * scale * 0.7 - t * 0.05, octaves=3)
    warp2_y = noise3.fbm(warped_x * scale * 0.7 + 50, warped_y * scale * 0.7 + t * 0.04, octaves=3)
    
    double_warped_x = warped_x + warp2_x * 50
    double_warped_y = warped_y + warp2_y * 50
    
    # Deep silk layer with double warping
    silk2 = noise2.fbm(
        double_warped_x * scale * 1.2 - t * 0.1,
        double_warped_y * scale * 1.2 + t * 0.07,
        octaves=4,
        persistence=0.5
    )
    
    # Swirling vortex effect using polar coordinates
    swirl_amount = (1 - r_norm) * 2.5  # More swirl at center
    swirled_angle = angle + swirl_amount * np.sin(t * 0.5) + t * 0.3
    swirl_pattern = noise3.fbm(
        np.cos(swirled_angle) * r_norm * radius * scale + t * 0.15,
        np.sin(swirled_angle) * r_norm * radius * scale,
        octaves=4
    )
    
    # Fine wispy detail
    wisp = noise3.turbulence(
        double_warped_x * scale * 2.5 + t * 0.2,
        double_warped_y * scale * 2.5 - t * 0.15,
        octaves=3
    ) * 0.3
    
    # Combine layers with emphasis on smooth flow
    density = (
        silk1 * 0.35 +           # Main silk flow
        silk2 * 0.30 +           # Secondary depth
        swirl_pattern * 0.25 +   # Swirl motion
        wisp * 0.10              # Fine detail
    )
    
    # Normalize and add contrast
    density = (density + 0.5)
    
    # Apply S-curve for more dramatic contrast
    density = np.clip(density, 0, 1)
    density = density * density * (3 - 2 * density)  # Smoothstep for natural contrast
    
    # Add brightness variation based on flow direction (fake lighting)
    flow_brightness = noise1.fbm(
        warped_x * scale * 0.5 + t * 0.05,
        warped_y * scale * 0.5,
        octaves=2
    ) * 0.2
    density += flow_brightness
    
    # Create "hot spots" - bright concentrated areas
    hotspot = noise2.fbm(
        double_warped_x * scale * 0.8 - t * 0.06,
        double_warped_y * scale * 0.8 + t * 0.04,
        octaves=3
    )
    hotspot = np.clip(hotspot + 0.3, 0, 1) ** 3  # Power curve for concentrated bright spots
    density += hotspot * 0.15
    
    # Edge darkening (vignette inside orb) - stronger near edges
    edge_vignette = 1.0 - (r_norm ** 1.5) * 0.5
    density *= np.clip(edge_vignette, 0.3, 1.0)
    
    # Smooth soft edge falloff at orb boundary
    edge_falloff = np.clip((radius - dist_masked) / 25, 0, 1)
    edge_falloff = edge_falloff ** 0.7  # Smoother falloff
    density *= edge_falloff
    
    # Final clamp
    density = np.clip(density, 0, 1)
    
    # Map density to colors
    rgb_masked = colormap_vectorized(density, colors)
    
    # Create output image array
    img_array = np.zeros((h, w, 3), dtype=np.uint8)
    img_array[mask] = rgb_masked
    
    # Create PIL image
    img = Image.fromarray(img_array, 'RGB')
    
    # Slight blur for smoother silk look
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    
    # Add glow effect
    glow = create_glow_layer(w, h, radius, cx, cy, colors['glow'], glow_size=50)
    
    # Composite
    result = Image.new('RGBA', (w, h), (0, 0, 0, 255))
    result = Image.alpha_composite(result, glow)
    result = Image.alpha_composite(result, img.convert('RGBA'))
    
    return result.convert('RGB')


def create_glow_layer(w, h, radius, center_x, center_y, glow_color, glow_size=60):
    """Create outer glow effect"""
    glow = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow)
    
    # Multiple concentric rings with decreasing opacity
    for i in range(glow_size, 0, -2):
        alpha = int(100 * (1 - i / glow_size) ** 1.5)
        r = radius + i
        color = (*glow_color, alpha)
        draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r], fill=color)
    
    return glow.filter(ImageFilter.GaussianBlur(radius=glow_size // 3))


# ==================== VIDEO GENERATION ====================
def get_duration(audio):
    """Get audio duration using ffprobe"""
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=noprint_wrappers=1:nokey=1', audio],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def create_silk_orb_video(audio, output, resolution='1080p', fps=30, color_scheme='purple', start_time=0):
    """Create beautiful fluid silk orb visualization"""
    
    print("\n" + "="*60)
    print("   🔮 SILK FLUID ORB VISUALIZER 🔮")
    print("="*60 + "\n")
    
    audio_path = Path(audio)
    if not audio_path.exists():
        print(f"❌ Audio '{audio}' not found!")
        return False
    
    # Resolution presets (9:10 portrait like reference)
    res_map = {
        '720p': (720, 800),
        '1080p': (1080, 1200),
        '2k': (1440, 1600),
        '4k': (2160, 2400),
        '4k+': (3456, 3840),  # Matches reference exactly
    }
    w, h = res_map.get(resolution, (1080, 1200))
    
    full_duration = get_duration(audio)
    duration = full_duration - start_time
    total_frames = int(duration * fps)
    
    colors = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES['purple'])
    
    print(f"📁 Audio: {audio_path.name}")
    print(f"⏱️  Duration: {duration:.1f}s ({total_frames} frames)")
    print(f"📐 Resolution: {w}x{h} @ {fps}fps")
    print(f"🎨 Color Scheme: {color_scheme}")
    print()
    
    # Orb parameters
    radius = int(min(w, h) * 0.42)
    cx, cy = w // 2, h // 2
    
    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp(prefix='silk_frames_')
    print(f"📂 Temp frames: {temp_dir}")
    
    try:
        # Initialize noise engines (once, reused for all frames)
        noise_engines = (
            VectorizedNoise(seed=42),
            VectorizedNoise(seed=137),
            VectorizedNoise(seed=256)
        )
        
        # Generate frames
        print("\n🎨 Generating fluid frames...")
        
        for frame_num in range(total_frames):
            frame_path = os.path.join(temp_dir, f"frame_{frame_num:06d}.png")
            frame = generate_frame_vectorized(frame_num, w, h, radius, cx, cy, noise_engines, colors, fps)
            frame.save(frame_path, 'PNG')
            
            # Progress
            progress = (frame_num + 1) / total_frames * 100
            elapsed_sec = frame_num / fps
            print(f"\r   🖼️  Frame {frame_num + 1}/{total_frames} ({progress:.1f}%) - {elapsed_sec:.1f}s", end='', flush=True)
        
        print("\n\n🎬 Encoding video with FFmpeg...")
        
        # Combine frames with audio using FFmpeg
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%06d.png'),
            '-ss', str(start_time),
            '-i', audio,
            '-c:v', 'libx264',
            '-preset', 'slow',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'high',
            '-level', '4.2',
            '-c:a', 'aac',
            '-b:a', '256k',
            '-shortest',
            '-movflags', '+faststart',
            output
        ]
        
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            if 'frame=' in line:
                match = re.search(r'frame=\s*(\d+)', line)
                if match:
                    encoded = int(match.group(1))
                    pct = min(100, encoded / total_frames * 100)
                    print(f"\r   📹 Encoding: {encoded}/{total_frames} ({pct:.1f}%)", end='', flush=True)
        
        proc.wait()
        print()
        
        if proc.returncode == 0 and Path(output).exists():
            size = Path(output).stat().st_size / (1024 * 1024)
            print(f"\n✅ SUCCESS! {output} ({size:.1f} MB)")
            return True
        else:
            print("❌ FFmpeg encoding failed!")
            return False
            
    finally:
        # Cleanup temp frames
        print(f"\n🧹 Cleaning up temp files...")
        shutil.rmtree(temp_dir, ignore_errors=True)


# ==================== BATCH PROCESSING ====================
def process_batch(audio_files, resolution='1080p', fps=30, color_scheme='purple'):
    """Process multiple audio files"""
    print("\n" + "="*60)
    print("   🎵 BATCH PROCESSING")
    print("="*60)
    print(f"\n📋 Files to process: {len(audio_files)}")
    
    results = []
    for i, audio in enumerate(audio_files, 1):
        print(f"\n{'─'*50}")
        print(f"📌 [{i}/{len(audio_files)}] Processing: {Path(audio).name}")
        print('─'*50)
        
        output = f"silk_{Path(audio).stem}.mp4"
        success = create_silk_orb_video(audio, output, resolution, fps, color_scheme)
        results.append((audio, output, success))
    
    print("\n" + "="*60)
    print("   📊 BATCH RESULTS")
    print("="*60 + "\n")
    
    for audio, output, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {Path(audio).name} → {output}")
    
    return results


# ==================== CLI ====================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("""
🔮 SILK FLUID ORB VISUALIZER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage: python silk_visualizer.py <audio> [options]

Options:
  resolution   720p, 1080p, 2k, 4k, 4k+ (default: 1080p)
  fps          Frame rate (default: 30, use 60 for smooth)
  color        purple, lava, ocean, golden, emerald

Examples:
  python silk_visualizer.py song.mp3
  python silk_visualizer.py track.mp3 1080p 60 purple
  python silk_visualizer.py audio.mp3 4k 30 lava

Batch Processing:
  python silk_visualizer.py file1.mp3 file2.mp3 file3.mp3
""")
        sys.exit(1)
    
    # Check if batch mode (multiple audio files)
    audio_files = [f for f in sys.argv[1:] if Path(f).exists() and f.endswith(('.mp3', '.wav', '.m4a', '.aac', '.mp4', '.flac'))]
    
    if len(audio_files) > 1:
        # Batch mode
        process_batch(audio_files)
    elif len(audio_files) == 1:
        # Single file mode
        audio = audio_files[0]
        res = sys.argv[2] if len(sys.argv) > 2 and not Path(sys.argv[2]).exists() else '1080p'
        fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        color = sys.argv[4] if len(sys.argv) > 4 else 'purple'
        
        output = f"silk_{Path(audio).stem}.mp4"
        create_silk_orb_video(audio, output, res, fps, color)
    else:
        print(f"❌ No valid audio files found in arguments!")
        sys.exit(1)
