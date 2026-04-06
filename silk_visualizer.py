#!/usr/bin/env python3
"""
SILK FLUID ORB VISUALIZER - Audio-Reactive Beat-Synced Visualization
Recreates the Resonation.io "Silk" effect with full spectrum audio reactivity

Features:
- GPU acceleration (optional, via cupy)
- Multiprocessing for parallel frame generation
- Checkpoint/resume capability
- Enhanced audio reactivity (BPM, spectral centroid)
- Particle system
- Custom color palettes from JSON

Usage: python silk_visualizer.py <audio> [options]
       python silk_visualizer.py --help
"""

import subprocess
import sys
import os
import re
import math
import tempfile
import shutil
import wave
import struct
import json
import argparse
import hashlib
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw, ImageFilter

# Optional GPU support
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    
# Advanced audio analysis
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# ==================== CONFIGURATION ====================
DEFAULT_CONFIG = {
    'audio_reactivity': {
        'attack': 0.25,
        'decay': 0.05,
        'smooth_sigma_ms': 80,
        'onset_decay_ms': 500,
        'bass_range': [0, 250],
        'mid_range': [250, 4000],
        'high_range': [4000, 22050],
    },
    'rendering': {
        'use_gpu': False,  # Auto-detect if True and available
        'num_workers': None,  # Auto-detect CPU count
        'chunk_size': 60,  # Frames per chunk for memory optimization
    },
    'particles': {
        'enabled': True,
        'max_count': 50,
        'spawn_on_beat': True,
        'lifetime_frames': 90,
        'size_range': [2, 8],
    }
}

def load_config(config_path=None):
    """Load configuration from JSON file or return defaults"""
    config = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Deep merge
            for key, value in user_config.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value
            print(f"📝 Loaded config from {config_path}")
        except Exception as e:
            print(f"⚠️  Config load error: {e}, using defaults")
    return config


# ==================== ERROR HANDLING & VALIDATION ====================
class SilkVisualizerError(Exception):
    """Base exception for silk visualizer errors"""
    pass

class AudioError(SilkVisualizerError):
    """Audio file related errors"""
    pass

class FFmpegError(SilkVisualizerError):
    """FFmpeg related errors"""
    pass

class DiskSpaceError(SilkVisualizerError):
    """Insufficient disk space error"""
    pass


def check_ffmpeg():
    """Verify FFmpeg is installed and accessible"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise FFmpegError("FFmpeg returned non-zero exit code")
        # Extract version
        version_match = re.search(r'ffmpeg version (\S+)', result.stdout)
        version = version_match.group(1) if version_match else 'unknown'
        return version
    except FileNotFoundError:
        raise FFmpegError("FFmpeg not found. Please install FFmpeg: https://ffmpeg.org/download.html")
    except subprocess.TimeoutExpired:
        raise FFmpegError("FFmpeg check timed out")


def check_ffprobe():
    """Verify ffprobe is installed"""
    try:
        subprocess.run(['ffprobe', '-version'], capture_output=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        raise FFmpegError("ffprobe not found. Please install FFmpeg with ffprobe.")


def validate_audio_file(audio_path):
    """Validate audio file exists and is readable"""
    path = Path(audio_path)
    if not path.exists():
        raise AudioError(f"Audio file not found: {audio_path}")
    if not path.is_file():
        raise AudioError(f"Not a file: {audio_path}")
    
    # Check file extension
    valid_extensions = {'.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.opus', '.mp4', '.mkv'}
    if path.suffix.lower() not in valid_extensions:
        raise AudioError(f"Unsupported audio format: {path.suffix}. Supported: {', '.join(valid_extensions)}")
    
    # Check file size (minimum 1KB)
    if path.stat().st_size < 1024:
        raise AudioError(f"Audio file too small (possibly corrupt): {audio_path}")
    
    # Verify FFmpeg can read it
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(path)
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0 or not result.stdout.strip():
            raise AudioError(f"FFmpeg cannot read audio file: {result.stderr}")
            
        duration = float(result.stdout.strip())
        if duration < 1.0:
            raise AudioError(f"Audio too short ({duration:.1f}s). Minimum 1 second required.")
            
        return duration
    except ValueError as e:
        raise AudioError(f"Invalid audio duration: {e}")
    except subprocess.TimeoutExpired:
        raise AudioError("Audio file validation timed out")


def check_disk_space(path, required_mb=500):
    """Check if sufficient disk space is available"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        free_mb = free / (1024 * 1024)
        if free_mb < required_mb:
            raise DiskSpaceError(f"Insufficient disk space. Need {required_mb}MB, have {free_mb:.0f}MB")
        return free_mb
    except Exception as e:
        if isinstance(e, DiskSpaceError):
            raise
        print(f"⚠️  Could not check disk space: {e}")
        return None


def estimate_disk_usage(width, height, total_frames, fps):
    """Estimate disk usage for temp frames in MB"""
    # PNG frame size estimate: ~0.5-2 bytes per pixel for compressed
    bytes_per_frame = width * height * 1.5  # Conservative estimate
    total_bytes = bytes_per_frame * total_frames
    return total_bytes / (1024 * 1024) + 100  # Add 100MB buffer


# ==================== AUDIO ANALYSIS MODULE ====================
class AudioAnalyzer:
    """
    Extracts per-frame audio features for beat-reactive visualization.
    Uses FFT-based spectral analysis with 3-band separation.
    Enhanced with BPM detection and spectral centroid.
    """
    
    def __init__(self, audio_path, fps=60, sensitivity=1.0, config=None):
        self.fps = fps
        self.sensitivity = sensitivity
        self.audio_path = audio_path
        self.config = config or DEFAULT_CONFIG['audio_reactivity']
        
        # Load and analyze audio
        self._load_audio()
        self._analyze_spectrum()
        self._detect_onsets()
        self._detect_bpm()
        self._compute_spectral_centroid()
        self._apply_envelope()
        
    def _load_audio(self):
        """Load audio file using ffmpeg → WAV conversion"""
        print("📊 Analyzing audio spectrum...")
        
        # Convert to mono WAV using ffmpeg
        temp_wav = tempfile.mktemp(suffix='.wav')
        cmd = [
            'ffmpeg', '-y', '-i', str(self.audio_path),
            '-ac', '1',  # mono
            '-ar', '44100',  # 44.1kHz sample rate
            '-f', 'wav',
            '-loglevel', 'error',
            temp_wav
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise AudioError(f"Failed to convert audio: {result.stderr}")
        
        # Read WAV file
        try:
            with wave.open(temp_wav, 'rb') as wf:
                self.sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
        
        # Convert to numpy array (16-bit signed integers → float)
        self.audio_data = np.array(struct.unpack(f'{n_frames}h', raw_data), dtype=np.float32)
        self.audio_data /= 32768.0  # Normalize to -1 to 1
        
        self.duration = len(self.audio_data) / self.sample_rate
        self.total_frames = int(self.duration * self.fps)
        self.samples_per_frame = self.sample_rate // self.fps
        
        print(f"   ⏱️  Duration: {self.duration:.1f}s, {self.total_frames} frames")
        
    def _analyze_spectrum(self):
        """Extract 3-band energy (bass/mid/high) per frame using STFT"""
        print("   🎵 Computing spectral analysis...")
        
        # Initialize arrays
        self.bass = np.zeros(self.total_frames)
        self.mid = np.zeros(self.total_frames)
        self.high = np.zeros(self.total_frames)
        self.raw_energy = np.zeros(self.total_frames)
        
        # FFT parameters
        fft_size = 2048
        hop_size = self.samples_per_frame
        freqs = np.fft.rfftfreq(fft_size, 1/self.sample_rate)
        
        # Frequency band masks
        bass_mask = freqs < 250  # Sub-bass + bass: 0-250Hz
        mid_mask = (freqs >= 250) & (freqs < 4000)  # Mids: 250-4000Hz
        high_mask = freqs >= 4000  # Highs: 4000Hz+
        
        # Hann window for smoother FFT
        window = np.hanning(fft_size)
        
        for frame_idx in range(self.total_frames):
            start = frame_idx * hop_size
            end = start + fft_size
            
            if end > len(self.audio_data):
                break
                
            # Get windowed segment
            segment = self.audio_data[start:end] * window
            
            # Compute FFT magnitude
            spectrum = np.abs(np.fft.rfft(segment))
            
            # Extract band energies (RMS-like)
            self.bass[frame_idx] = np.sqrt(np.mean(spectrum[bass_mask] ** 2))
            self.mid[frame_idx] = np.sqrt(np.mean(spectrum[mid_mask] ** 2))
            self.high[frame_idx] = np.sqrt(np.mean(spectrum[high_mask] ** 2))
            self.raw_energy[frame_idx] = np.sqrt(np.mean(spectrum ** 2))
            
        # Normalize each band to 0-1 with headroom
        for band in [self.bass, self.mid, self.high, self.raw_energy]:
            if band.max() > 0:
                # Use 95th percentile for normalization (allows peaks > 1)
                p95 = np.percentile(band, 95)
                if p95 > 0:
                    band /= p95
                    
    def _detect_onsets(self):
        """Detect transients/onsets for 'snap' effects"""
        print("   🥁 Detecting beat transients...")
        
        # Onset detection using spectral flux
        self.onset_strength = np.zeros(self.total_frames)
        
        # Compute derivative of energy (onset = sudden increase)
        energy_diff = np.diff(self.raw_energy, prepend=0)
        
        # Half-wave rectification (only positive changes = onsets)
        self.onset_strength = np.maximum(0, energy_diff)
        
        # Also weight by bass (bass hits are more important)
        bass_diff = np.diff(self.bass, prepend=0)
        bass_onset = np.maximum(0, bass_diff)
        
        # Combine: 60% spectral flux, 40% bass onset
        self.onset_strength = 0.6 * self.onset_strength + 0.4 * bass_onset
        
        # Normalize
        if self.onset_strength.max() > 0:
            self.onset_strength /= np.percentile(self.onset_strength, 98)
            
        # Peak picking: suppress non-peaks
        peaks, _ = signal.find_peaks(self.onset_strength, height=0.3, distance=int(self.fps * 0.1))
        
        # Create impulse signal (sharp spike on peaks)
        self.onset_impulse = np.zeros(self.total_frames)
        self.onset_impulse[peaks] = self.onset_strength[peaks]
        self.beat_frames = peaks  # Store beat frame indices for particle spawning
        
        print(f"   ✨ Found {len(peaks)} beat transients")
    
    def _detect_bpm(self):
        """Detect BPM for beat-aligned effects"""
        print("   🎼 Detecting BPM...")
        
        self.bpm = 120.0  # Default fallback
        self.beat_phase = np.zeros(self.total_frames)
        
        if LIBROSA_AVAILABLE:
            try:
                # Use librosa for accurate BPM detection
                tempo, beat_frames = librosa.beat.beat_track(
                    y=self.audio_data, 
                    sr=self.sample_rate,
                    units='frames',
                    hop_length=self.samples_per_frame
                )
                self.bpm = float(tempo) if hasattr(tempo, '__len__') else tempo
                print(f"   🎵 Detected BPM: {self.bpm:.1f}")
            except Exception as e:
                print(f"   ⚠️  BPM detection failed: {e}, using estimate")
                self._estimate_bpm_from_onsets()
        else:
            self._estimate_bpm_from_onsets()
        
        # Compute beat phase (0-1 cycle between beats)
        beat_period_frames = (60.0 / self.bpm) * self.fps
        for i in range(self.total_frames):
            self.beat_phase[i] = (i % beat_period_frames) / beat_period_frames
    
    def _estimate_bpm_from_onsets(self):
        """Estimate BPM from onset intervals when librosa unavailable"""
        if len(self.beat_frames) > 2:
            intervals = np.diff(self.beat_frames)
            median_interval = np.median(intervals)
            if median_interval > 0:
                self.bpm = (60.0 * self.fps) / median_interval
                self.bpm = np.clip(self.bpm, 60, 200)  # Reasonable BPM range
        print(f"   🎵 Estimated BPM: {self.bpm:.1f}")
    
    def _compute_spectral_centroid(self):
        """Compute spectral centroid (brightness) per frame"""
        print("   ✨ Computing spectral brightness...")
        
        self.brightness = np.zeros(self.total_frames)
        fft_size = 2048
        hop_size = self.samples_per_frame
        freqs = np.fft.rfftfreq(fft_size, 1/self.sample_rate)
        window = np.hanning(fft_size)
        
        for frame_idx in range(self.total_frames):
            start = frame_idx * hop_size
            end = start + fft_size
            
            if end > len(self.audio_data):
                break
            
            segment = self.audio_data[start:end] * window
            spectrum = np.abs(np.fft.rfft(segment))
            
            # Spectral centroid: weighted average of frequencies
            if spectrum.sum() > 0:
                centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
                # Normalize to 0-1 (assuming max centroid ~10kHz)
                self.brightness[frame_idx] = np.clip(centroid / 10000.0, 0, 1)
        
        # Smooth the brightness curve
        self.brightness = gaussian_filter1d(self.brightness, sigma=self.fps * 0.05)
        
    def _apply_envelope(self):
        """Apply attack/decay envelope for BUTTERY SMOOTH response"""
        # SMOOTH: Slower attack, much longer decay for fluid motion
        
        # Apply heavy smoothing to each band for fluid motion
        for band in [self.bass, self.mid, self.high]:
            # Multi-pass smoothing for ultra-smooth transitions
            smoothed = np.zeros_like(band)
            smoothed[0] = band[0]
            
            for i in range(1, len(band)):
                if band[i] > smoothed[i-1]:
                    # Attack: gradual rise (not instant)
                    alpha = 0.25  # Smooth attack (was 0.7)
                else:
                    # Decay: very slow fall for smooth trails
                    alpha = 0.05  # Very slow decay (was 0.15)
                smoothed[i] = alpha * band[i] + (1 - alpha) * smoothed[i-1]
            
            band[:] = smoothed
        
        # Additional Gaussian smoothing for extra smoothness
        from scipy.ndimage import gaussian_filter1d
        smooth_sigma = self.fps * 0.08  # ~80ms smoothing window
        self.bass = gaussian_filter1d(self.bass, sigma=smooth_sigma)
        self.mid = gaussian_filter1d(self.mid, sigma=smooth_sigma)
        self.high = gaussian_filter1d(self.high, sigma=smooth_sigma)
        
        # Onset envelope: much longer, smoother decay for fluid pulses
        decay_frames = int(self.fps * 0.5)  # 500ms decay (was 150ms)
        decay_kernel = np.exp(-np.arange(decay_frames) / (decay_frames / 3))
        self.onset_envelope = np.convolve(self.onset_impulse, decay_kernel, mode='same')
        self.onset_envelope = gaussian_filter1d(self.onset_envelope, sigma=self.fps * 0.05)
        self.onset_envelope = np.clip(self.onset_envelope, 0, 1.5)
        
    def get_frame_params(self, frame_idx):
        """Get audio-reactive parameters for a specific frame - BUTTERY SMOOTH"""
        if frame_idx >= self.total_frames:
            frame_idx = self.total_frames - 1
            
        bass = np.clip(self.bass[frame_idx] * self.sensitivity, 0, 1.5)
        mid = np.clip(self.mid[frame_idx] * self.sensitivity, 0, 1.5)
        high = np.clip(self.high[frame_idx] * self.sensitivity, 0, 1.5)
        onset = np.clip(self.onset_envelope[frame_idx] * self.sensitivity, 0, 1.2)
        brightness = self.brightness[frame_idx] if hasattr(self, 'brightness') else 0.5
        beat_phase = self.beat_phase[frame_idx] if hasattr(self, 'beat_phase') else 0.0
        
        return {
            'bass': bass,
            'mid': mid,
            'high': high,
            'onset': onset,
            'brightness': brightness,  # Spectral centroid
            'beat_phase': beat_phase,  # 0-1 cycle between beats
            'bpm': self.bpm if hasattr(self, 'bpm') else 120.0,
            # BUTTERY SMOOTH: Gentler, more gradual parameters
            'scale': 1.0 + bass * 0.15,  # Gentler scaling: 1.0-1.15 (was 1.0-1.25)
            'warp_intensity': 60 + bass * 50,  # Softer deformation (was 80 + bass*100)
            'flow_speed_mult': 1.0 + mid * 0.4,  # Gentler flow variation (was 0.8)
            'detail_mult': 1.0 + high * 0.6,  # Subtler detail (was 1.2)
            'snap_intensity': onset * 0.6,  # Much softer snap (was 1.5)
            'glow_boost': 1.0 + bass * 0.3 + onset * 0.2,  # Gentler glow (was 0.6/0.4)
            'shimmer': brightness * high * 0.5,  # Brightness-driven shimmer
        }
    
    def is_beat_frame(self, frame_idx):
        """Check if this frame is on a detected beat"""
        return frame_idx in self.beat_frames if hasattr(self, 'beat_frames') else False

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


# ==================== PARTICLE SYSTEM ====================
class ParticleSystem:
    """Beat-reactive particle system for enhanced visuals"""
    
    def __init__(self, width, height, max_particles=50, config=None):
        self.width = width
        self.height = height
        self.max_particles = max_particles
        self.config = config or DEFAULT_CONFIG['particles']
        
        # Particle arrays (pre-allocated for performance)
        self.positions = np.zeros((max_particles, 2), dtype=np.float32)  # x, y
        self.velocities = np.zeros((max_particles, 2), dtype=np.float32)  # vx, vy
        self.lifetimes = np.zeros(max_particles, dtype=np.float32)  # remaining frames
        self.sizes = np.zeros(max_particles, dtype=np.float32)
        self.colors = np.zeros((max_particles, 4), dtype=np.uint8)  # RGBA
        self.active = np.zeros(max_particles, dtype=bool)
        
        self.next_particle = 0
        self.center_x = width // 2
        self.center_y = height // 2
        
    def spawn_particles(self, count, orb_radius, glow_color, audio_params):
        """Spawn new particles (typically on beats)"""
        bass = audio_params.get('bass', 0.5)
        brightness = audio_params.get('brightness', 0.5)
        
        for _ in range(min(count, self.max_particles)):
            # Find an inactive slot
            idx = self.next_particle % self.max_particles
            self.next_particle += 1
            
            # Spawn at orb edge with random angle
            angle = np.random.uniform(0, 2 * np.pi)
            spawn_radius = orb_radius * (0.9 + np.random.uniform(0, 0.2))
            
            self.positions[idx] = [
                self.center_x + np.cos(angle) * spawn_radius,
                self.center_y + np.sin(angle) * spawn_radius
            ]
            
            # Velocity: outward with some randomness, bass affects speed
            speed = 1.5 + bass * 2.5 + np.random.uniform(0, 1)
            self.velocities[idx] = [
                np.cos(angle) * speed + np.random.uniform(-0.5, 0.5),
                np.sin(angle) * speed + np.random.uniform(-0.5, 0.5)
            ]
            
            # Size based on bass and random
            size_range = self.config.get('size_range', [2, 8])
            self.sizes[idx] = size_range[0] + (size_range[1] - size_range[0]) * (bass * 0.5 + np.random.uniform(0, 0.5))
            
            # Color: use glow color with some variation
            alpha = int(180 + brightness * 75)
            self.colors[idx] = [
                min(255, glow_color[0] + np.random.randint(-20, 20)),
                min(255, glow_color[1] + np.random.randint(-20, 20)),
                min(255, glow_color[2] + np.random.randint(-20, 20)),
                alpha
            ]
            
            # Lifetime
            self.lifetimes[idx] = self.config.get('lifetime_frames', 90) * (0.7 + np.random.uniform(0, 0.6))
            self.active[idx] = True
    
    def update(self, audio_params):
        """Update particle positions and lifetimes"""
        active_mask = self.active
        
        if not np.any(active_mask):
            return
        
        # Update positions
        self.positions[active_mask] += self.velocities[active_mask]
        
        # Apply subtle gravity toward center (creates swirl effect)
        dx = self.center_x - self.positions[active_mask, 0]
        dy = self.center_y - self.positions[active_mask, 1]
        dist = np.sqrt(dx**2 + dy**2) + 1
        gravity = 0.02 * audio_params.get('mid', 0.5)
        self.velocities[active_mask, 0] += (dx / dist) * gravity
        self.velocities[active_mask, 1] += (dy / dist) * gravity
        
        # Apply drag
        self.velocities[active_mask] *= 0.98
        
        # Decrease lifetime
        self.lifetimes[active_mask] -= 1
        
        # Fade out particles
        fade_mask = (self.lifetimes > 0) & (self.lifetimes < 30) & active_mask
        if np.any(fade_mask):
            fade_factor = self.lifetimes[fade_mask] / 30.0
            self.colors[fade_mask, 3] = (fade_factor * 180).astype(np.uint8)
        
        # Deactivate dead particles
        self.active[self.lifetimes <= 0] = False
        
        # Deactivate particles that left the screen
        out_of_bounds = (
            (self.positions[:, 0] < -50) | (self.positions[:, 0] > self.width + 50) |
            (self.positions[:, 1] < -50) | (self.positions[:, 1] > self.height + 50)
        )
        self.active[out_of_bounds] = False
    
    def render(self, img):
        """Render particles onto an image"""
        if not np.any(self.active):
            return img
        
        # Convert to RGBA if needed
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        draw = ImageDraw.Draw(img, 'RGBA')
        
        for i in range(self.max_particles):
            if not self.active[i]:
                continue
            
            x, y = self.positions[i]
            size = self.sizes[i]
            color = tuple(self.colors[i])
            
            # Draw glowing circle
            for s in range(int(size), 0, -1):
                alpha = int(color[3] * (s / size) * 0.5)
                glow_color = (color[0], color[1], color[2], alpha)
                draw.ellipse([x - s, y - s, x + s, y + s], fill=glow_color)
        
        return img
    
    def get_state(self):
        """Get serializable state for checkpointing"""
        return {
            'positions': self.positions.copy(),
            'velocities': self.velocities.copy(),
            'lifetimes': self.lifetimes.copy(),
            'sizes': self.sizes.copy(),
            'colors': self.colors.copy(),
            'active': self.active.copy(),
            'next_particle': self.next_particle,
        }
    
    def set_state(self, state):
        """Restore state from checkpoint"""
        self.positions = state['positions']
        self.velocities = state['velocities']
        self.lifetimes = state['lifetimes']
        self.sizes = state['sizes']
        self.colors = state['colors']
        self.active = state['active']
        self.next_particle = state['next_particle']


# ==================== GPU-ACCELERATED NOISE (Optional) ====================
class GPUNoise:
    """CuPy-based GPU-accelerated noise generation"""
    
    def __init__(self, seed=0):
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available for GPU acceleration")
        
        cp.random.seed(seed)
        self.perm = cp.arange(256, dtype=cp.int32)
        cp.random.shuffle(self.perm)
        self.perm = cp.tile(self.perm, 2)
        
        angles = cp.linspace(0, 2 * cp.pi, 8, endpoint=False)
        self.gradients = cp.stack([cp.cos(angles), cp.sin(angles)], axis=1)
    
    def _fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def noise2d(self, x, y):
        """GPU-accelerated 2D Perlin noise"""
        xi = cp.floor(x).astype(cp.int32) & 255
        yi = cp.floor(y).astype(cp.int32) & 255
        xf = x - cp.floor(x)
        yf = y - cp.floor(y)
        
        u = self._fade(xf)
        v = self._fade(yf)
        
        aa = self.perm[self.perm[xi] + yi] % 8
        ab = self.perm[self.perm[xi] + yi + 1] % 8
        ba = self.perm[self.perm[xi + 1] + yi] % 8
        bb = self.perm[self.perm[xi + 1] + yi + 1] % 8
        
        g_aa = self.gradients[aa, 0] * xf + self.gradients[aa, 1] * yf
        g_ba = self.gradients[ba, 0] * (xf - 1) + self.gradients[ba, 1] * yf
        g_ab = self.gradients[ab, 0] * xf + self.gradients[ab, 1] * (yf - 1)
        g_bb = self.gradients[bb, 0] * (xf - 1) + self.gradients[bb, 1] * (yf - 1)
        
        x1 = g_aa + u * (g_ba - g_aa)
        x2 = g_ab + u * (g_bb - g_ab)
        return x1 + v * (x2 - x1)
    
    def fbm(self, x, y, octaves=5, lacunarity=2.0, persistence=0.5):
        """Fractal Brownian Motion on GPU"""
        value = cp.zeros_like(x)
        amplitude = 1.0
        max_val = 0.0
        
        for _ in range(octaves):
            value += amplitude * self.noise2d(x, y)
            max_val += amplitude
            amplitude *= persistence
            x = x * lacunarity
            y = y * lacunarity
        
        return value / max_val
    
    def to_numpy(self, arr):
        """Transfer GPU array back to numpy"""
        return cp.asnumpy(arr)


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
    'volcano': {  # Deep red to orange
        'dark': np.array([30, 5, 5]),
        'mid': np.array([150, 30, 20]),
        'bright': np.array([255, 100, 30]),
        'hot': np.array([255, 200, 100]),
        'glow': (255, 60, 20),
    },
    'cyber': {  # Cyan/pink cyberpunk
        'dark': np.array([5, 10, 30]),
        'mid': np.array([30, 150, 200]),
        'bright': np.array([150, 220, 255]),
        'hot': np.array([255, 180, 255]),
        'glow': (100, 200, 255),
    },
    'sunset': {  # Orange to pink gradient
        'dark': np.array([30, 10, 20]),
        'mid': np.array([200, 80, 60]),
        'bright': np.array([255, 150, 100]),
        'hot': np.array([255, 200, 180]),
        'glow': (255, 120, 80),
    },
}


def load_custom_colors(json_path):
    """Load custom color schemes from JSON file
    
    JSON format:
    {
        "scheme_name": {
            "dark": [r, g, b],
            "mid": [r, g, b],
            "bright": [r, g, b],
            "hot": [r, g, b],
            "glow": [r, g, b]
        }
    }
    """
    try:
        with open(json_path, 'r') as f:
            custom = json.load(f)
        
        for name, scheme in custom.items():
            COLOR_SCHEMES[name] = {
                'dark': np.array(scheme['dark']),
                'mid': np.array(scheme['mid']),
                'bright': np.array(scheme['bright']),
                'hot': np.array(scheme['hot']),
                'glow': tuple(scheme['glow']),
            }
        print(f"📝 Loaded {len(custom)} custom color schemes from {json_path}")
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"⚠️  Failed to load custom colors: {e}")
        return False


def list_color_schemes():
    """List all available color schemes"""
    return list(COLOR_SCHEMES.keys())


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


# ==================== FRAME GENERATION (AUDIO-REACTIVE) ====================
def generate_frame_vectorized(frame_num, w, h, base_radius, cx, cy, noise_engines, colors, fps, audio_params=None):
    """
    Generate single frame with audio-reactive beat synchronization.
    
    Audio parameters modulate:
    - Sphere scale (bass → expansion)
    - Deformation intensity (bass + onset → warping)
    - Flow speed (mids → motion)
    - Detail level (highs → fine texture)
    - Snap impulse (onsets → sharp deformation)
    """
    t = frame_num / fps
    
    # Default audio params if not provided (for testing without audio)
    if audio_params is None:
        audio_params = {
            'bass': 0, 'mid': 0, 'high': 0, 'onset': 0,
            'scale': 1.0, 'warp_intensity': 80, 'flow_speed_mult': 1.0,
            'detail_mult': 1.0, 'snap_intensity': 0, 'glow_boost': 1.0
        }
    
    # Extract audio-reactive parameters
    scale_factor = audio_params['scale']
    warp_intensity = audio_params['warp_intensity']
    flow_mult = audio_params['flow_speed_mult']
    detail_mult = audio_params['detail_mult']
    snap = audio_params['snap_intensity']
    glow_boost = audio_params['glow_boost']
    bass = audio_params['bass']
    
    # Apply audio-reactive scaling to radius
    radius = int(base_radius * scale_factor)
    
    noise1, noise2, noise3 = noise_engines
    
    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Calculate distance from center
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    
    # Create circular mask with some padding for scaled radius
    max_radius = int(base_radius * 1.4)  # Allow for expansion
    mask = dist <= max_radius + 10
    
    # Only compute for pixels inside the extended orb area
    x_masked = x_grid[mask]
    y_masked = y_grid[mask]
    dist_masked = dist[mask]
    
    # Convert to polar coordinates for swirl + deformation effects
    dx = x_masked - cx
    dy = y_masked - cy
    angle = np.arctan2(dy, dx)
    r_norm = dist_masked / radius  # 0 at center, 1 at edge
    
    # ===== BUTTERY SMOOTH RADIAL DEFORMATION =====
    # Gentle, flowing deformation instead of sharp snaps
    
    # Base radial wobble from noise - slower, gentler
    wobble_angle = noise1.fbm(angle * 2 + t * 0.15, t * 0.25, octaves=2) * 0.08
    
    # Smooth radial "pulse" on transients (no sharp oscillations)
    snap_deform = snap * np.sin(angle * 2 + t * 3) * 0.05  # Slow, gentle wave
    
    # Bass-driven radial pulse - very smooth
    bass_pulse = bass * np.sin(angle * 1.5 + t * 2) * 0.06
    bass_pulse += bass * 0.03  # Gentle overall expansion
    
    # Combined radial deformation - subtle and smooth
    radial_deform = 1.0 + wobble_angle + snap_deform + bass_pulse
    
    # Apply deformation to normalized radius
    r_deformed = r_norm * radial_deform
    
    # ===== DOMAIN WARPING (BUTTERY SMOOTH) =====
    warp_scale = 0.005  # Slightly larger scale for smoother patterns
    base_flow = t * 0.06 * flow_mult  # Slower base flow
    
    warp_x = noise1.fbm(x_masked * warp_scale, y_masked * warp_scale + base_flow, octaves=4)
    warp_y = noise2.fbm(x_masked * warp_scale + 100, y_masked * warp_scale - base_flow * 0.75, octaves=4)
    
    # Warp intensity - gentler response
    warped_x = x_masked + warp_x * warp_intensity
    warped_y = y_masked + warp_y * warp_intensity
    
    # Second pass: sample noise at warped coordinates
    scale = 0.007  # Slightly larger for smoother patterns
    flow_t = t * 0.08 * flow_mult  # Slower flow
    
    # Main flowing silk layer - uses warped coordinates
    silk1 = noise1.fbm(
        warped_x * scale + flow_t,
        warped_y * scale + flow_t * 0.67,
        octaves=5,
        persistence=0.55
    )
    
    # Second warp pass for more complexity
    warp2_scale = scale * 0.7
    warp2_x = noise2.fbm(warped_x * warp2_scale, warped_y * warp2_scale - t * 0.05 * flow_mult, octaves=3)
    warp2_y = noise3.fbm(warped_x * warp2_scale + 50, warped_y * warp2_scale + t * 0.04 * flow_mult, octaves=3)
    
    # Secondary warp intensity also audio-reactive
    secondary_warp = 50 + bass * 40
    double_warped_x = warped_x + warp2_x * secondary_warp
    double_warped_y = warped_y + warp2_y * secondary_warp
    
    # Deep silk layer with double warping
    silk2 = noise2.fbm(
        double_warped_x * scale * 1.2 - flow_t * 0.83,
        double_warped_y * scale * 1.2 + flow_t * 0.58,
        octaves=4,
        persistence=0.5
    )
    
    # Swirling vortex effect using polar coordinates
    swirl_amount = (1 - r_deformed) * 2.5  # Use deformed radius
    swirl_speed = 0.5 * flow_mult
    swirled_angle = angle + swirl_amount * np.sin(t * swirl_speed) + t * 0.3 * flow_mult
    swirl_pattern = noise3.fbm(
        np.cos(swirled_angle) * r_deformed * radius * scale + t * 0.15 * flow_mult,
        np.sin(swirled_angle) * r_deformed * radius * scale,
        octaves=4
    )
    
    # Fine wispy detail (intensity scales with high frequencies)
    detail_scale = scale * 2.5 * detail_mult
    wisp = noise3.turbulence(
        double_warped_x * detail_scale + t * 0.2 * flow_mult,
        double_warped_y * detail_scale - t * 0.15 * flow_mult,
        octaves=3
    ) * 0.3 * detail_mult
    
    # ===== COMBINE LAYERS (AUDIO-WEIGHTED) =====
    # On beats, emphasize the flowing layers more
    silk_weight = 0.35 + bass * 0.1
    swirl_weight = 0.25 + snap * 0.15
    
    density = (
        silk1 * silk_weight +        # Main silk flow (boosted on bass)
        silk2 * 0.30 +               # Secondary depth
        swirl_pattern * swirl_weight + # Swirl motion (boosted on snap)
        wisp * 0.10                  # Fine detail
    )
    
    # Normalize and add contrast
    density = (density + 0.5)
    
    # BUTTERY SMOOTH: Gentler contrast curve
    density = np.clip(density, 0, 1)
    contrast_power = 1.8 + bass * 0.2  # Gentler contrast variation
    density = np.power(density, 1/contrast_power) * density * (3 - 2 * density)
    
    # Add brightness variation based on flow direction
    flow_brightness = noise1.fbm(
        warped_x * scale * 0.5 + t * 0.04 * flow_mult,
        warped_y * scale * 0.5,
        octaves=2
    ) * 0.15  # Gentler brightness variation
    density += flow_brightness
    
    # Create "hot spots" - gentler intensity
    hotspot = noise2.fbm(
        double_warped_x * scale * 0.8 - t * 0.04 * flow_mult,
        double_warped_y * scale * 0.8 + t * 0.03 * flow_mult,
        octaves=3
    )
    hotspot_intensity = 0.12 + bass * 0.1 + snap * 0.08  # Gentler hotspots
    hotspot = np.clip(hotspot + 0.3, 0, 1) ** 3
    density += hotspot * hotspot_intensity
    
    # Edge darkening (vignette inside orb)
    edge_vignette = 1.0 - (r_deformed ** 1.5) * 0.45  # Slightly softer vignette
    density *= np.clip(edge_vignette, 0.35, 1.0)
    
    # Smooth soft edge falloff at orb boundary (using deformed radius)
    edge_dist = radius * radial_deform - dist_masked
    edge_falloff = np.clip(edge_dist / 35, 0, 1)  # Softer edge
    edge_falloff = edge_falloff ** 0.6
    density *= edge_falloff
    
    # Beat-reactive brightness boost - gentler
    density *= (1.0 + bass * 0.15 + snap * 0.1)
    
    # Final clamp
    density = np.clip(density, 0, 1)
    
    # Map density to colors
    rgb_masked = colormap_vectorized(density, colors)
    
    # Create output image array
    img_array = np.zeros((h, w, 3), dtype=np.uint8)
    img_array[mask] = rgb_masked
    
    # Create PIL image
    img = Image.fromarray(img_array, 'RGB')
    
    # BUTTERY SMOOTH: More blur for silky soft look
    blur_radius = 1.8  # Consistent soft blur (was variable 0.8-1.2)
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Add glow effect (gentler intensity variation)
    glow_size = int(50 + 15 * glow_boost)  # Gentler glow scaling
    glow = create_glow_layer(w, h, radius, cx, cy, colors['glow'], glow_size=glow_size, intensity=glow_boost)
    
    # Composite
    result = Image.new('RGBA', (w, h), (0, 0, 0, 255))
    result = Image.alpha_composite(result, glow)
    result = Image.alpha_composite(result, img.convert('RGBA'))
    
    return result.convert('RGB')


def create_glow_layer(w, h, radius, center_x, center_y, glow_color, glow_size=60, intensity=1.0):
    """Create outer glow effect with audio-reactive intensity"""
    glow = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow)
    
    # Multiple concentric rings with decreasing opacity
    base_alpha = int(100 * intensity)
    for i in range(glow_size, 0, -2):
        alpha = int(base_alpha * (1 - i / glow_size) ** 1.5)
        alpha = min(255, alpha)
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


# ==================== CHECKPOINTING ====================
class RenderCheckpoint:
    """Checkpoint system for resumable rendering"""
    
    def __init__(self, audio_path, output_path, resolution, fps, color_scheme, sensitivity):
        # Create unique checkpoint ID based on parameters
        params_str = f"{audio_path}_{resolution}_{fps}_{color_scheme}_{sensitivity}"
        self.checkpoint_id = hashlib.md5(params_str.encode()).hexdigest()[:12]
        self.checkpoint_dir = Path(tempfile.gettempdir()) / f"silk_checkpoint_{self.checkpoint_id}"
        self.checkpoint_file = self.checkpoint_dir / "progress.pkl"
        self.frames_dir = self.checkpoint_dir / "frames"
        
    def exists(self):
        """Check if a checkpoint exists"""
        return self.checkpoint_file.exists()
    
    def load(self):
        """Load checkpoint data"""
        if not self.exists():
            return None
        try:
            with open(self.checkpoint_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            return None
    
    def save(self, data):
        """Save checkpoint data"""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(exist_ok=True)
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
    
    def get_frames_dir(self):
        """Get directory for rendered frames"""
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        return str(self.frames_dir)
    
    def cleanup(self):
        """Remove checkpoint after successful completion"""
        try:
            if self.checkpoint_dir.exists():
                shutil.rmtree(self.checkpoint_dir)
        except Exception as e:
            print(f"⚠️  Failed to cleanup checkpoint: {e}")
    
    def get_completed_frames(self):
        """Get list of already rendered frame numbers"""
        if not self.frames_dir.exists():
            return set()
        completed = set()
        for f in self.frames_dir.glob("frame_*.png"):
            match = re.search(r'frame_(\d+)\.png', f.name)
            if match:
                completed.add(int(match.group(1)))
        return completed


# ==================== MULTIPROCESSING FRAME GENERATION ====================
def _render_frame_worker(args):
    """Worker function for parallel frame generation"""
    (frame_num, w, h, base_radius, cx, cy, noise_seeds, colors_dict, fps, 
     audio_params, output_path, particle_state) = args
    
    # Recreate noise engines in worker process
    noise_engines = (
        VectorizedNoise(seed=noise_seeds[0]),
        VectorizedNoise(seed=noise_seeds[1]),
        VectorizedNoise(seed=noise_seeds[2])
    )
    
    # Convert colors dict back to numpy arrays
    colors = {
        'dark': np.array(colors_dict['dark']),
        'mid': np.array(colors_dict['mid']),
        'bright': np.array(colors_dict['bright']),
        'hot': np.array(colors_dict['hot']),
        'glow': tuple(colors_dict['glow']),
    }
    
    # Generate frame
    frame = generate_frame_vectorized(
        frame_num, w, h, base_radius, cx, cy,
        noise_engines, colors, fps, audio_params
    )
    
    # Save frame
    frame.save(output_path, 'PNG', compress_level=1)
    
    return frame_num


def render_frames_parallel(frame_range, w, h, base_radius, cx, cy, noise_seeds, colors, 
                          fps, audio_params_list, output_dir, num_workers=None,
                          particle_system=None, audio_analyzer=None):
    """Render frames in parallel using multiprocessing"""
    
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    # Convert colors to serializable format
    colors_dict = {
        'dark': colors['dark'].tolist(),
        'mid': colors['mid'].tolist(),
        'bright': colors['bright'].tolist(),
        'hot': colors['hot'].tolist(),
        'glow': list(colors['glow']),
    }
    
    # Prepare arguments for each frame
    tasks = []
    for frame_num in frame_range:
        output_path = os.path.join(output_dir, f"frame_{frame_num:06d}.png")
        audio_params = audio_params_list[frame_num] if frame_num < len(audio_params_list) else audio_params_list[-1]
        
        tasks.append((
            frame_num, w, h, base_radius, cx, cy, noise_seeds, colors_dict, fps,
            audio_params, output_path, None  # particle_state placeholder
        ))
    
    completed = []
    
    if num_workers > 1:
        # Use multiprocessing for parallel rendering
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            if TQDM_AVAILABLE:
                futures = {executor.submit(_render_frame_worker, task): task[0] for task in tasks}
                with tqdm(total=len(tasks), desc="🎨 Rendering frames", unit="frame") as pbar:
                    for future in as_completed(futures):
                        try:
                            frame_num = future.result()
                            completed.append(frame_num)
                            pbar.update(1)
                        except Exception as e:
                            print(f"\n⚠️  Frame {futures[future]} failed: {e}")
            else:
                futures = list(executor.map(_render_frame_worker, tasks))
                completed = futures
    else:
        # Single-threaded fallback
        if TQDM_AVAILABLE:
            for task in tqdm(tasks, desc="🎨 Rendering frames", unit="frame"):
                frame_num = _render_frame_worker(task)
                completed.append(frame_num)
        else:
            for i, task in enumerate(tasks):
                frame_num = _render_frame_worker(task)
                completed.append(frame_num)
                progress = (i + 1) / len(tasks) * 100
                print(f"\r   🖼️  Frame {i + 1}/{len(tasks)} ({progress:.1f}%)", end='', flush=True)
            print()
    
    return completed


# ==================== MEMORY-OPTIMIZED STREAMING TO FFMPEG ====================
def stream_frames_to_ffmpeg(frame_generator, total_frames, output_path, audio_path, 
                            fps, width, height, start_time=0):
    """Stream frames directly to FFmpeg via pipe (memory efficient)"""
    
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',  # Read from stdin
        '-ss', str(start_time),
        '-i', audio_path,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        '-profile:v', 'high',
        '-level', '4.2',
        '-c:a', 'aac',
        '-b:a', '256k',
        '-shortest',
        '-movflags', '+faststart',
        output_path
    ]
    
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        if TQDM_AVAILABLE:
            pbar = tqdm(total=total_frames, desc="🎬 Encoding", unit="frame")
        
        for frame_num, frame in enumerate(frame_generator):
            # Convert PIL image to raw bytes
            frame_bytes = np.array(frame.convert('RGB')).tobytes()
            proc.stdin.write(frame_bytes)
            
            if TQDM_AVAILABLE:
                pbar.update(1)
            else:
                if frame_num % 30 == 0:
                    progress = (frame_num + 1) / total_frames * 100
                    print(f"\r   🎬 Encoding: {frame_num + 1}/{total_frames} ({progress:.1f}%)", end='', flush=True)
        
        if TQDM_AVAILABLE:
            pbar.close()
        else:
            print()
        
        proc.stdin.close()
        proc.wait()
        
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode()
            raise FFmpegError(f"FFmpeg encoding failed: {stderr}")
            
        return True
        
    except Exception as e:
        proc.kill()
        raise


def create_silk_orb_video(audio, output, resolution='1080p', fps=60, color_scheme='purple', 
                         start_time=0, sensitivity=1.0, num_workers=None, use_gpu=False,
                         enable_particles=True, resume=False, preview=False, config=None):
    """
    Create audio-reactive silk orb visualization with beat synchronization.
    
    Args:
        audio: Path to audio file
        output: Output video path
        resolution: Resolution preset (720p, 1080p, 2k, 4k, 4k+)
        fps: Frame rate
        color_scheme: Color palette name
        start_time: Start offset in audio (seconds)
        sensitivity: Audio reactivity multiplier (0.5-2.0)
        num_workers: Number of parallel workers (None = auto-detect)
        use_gpu: Enable GPU acceleration if available
        enable_particles: Enable particle system
        resume: Resume from checkpoint if available
        preview: Quick preview mode (lower quality, faster)
        config: Optional config dict or path to JSON config
    """
    
    print("\n" + "="*60)
    print("   🔮 SILK VISUALIZER - AUDIO REACTIVE 🔮")
    print("="*60 + "\n")
    
    # ===== VALIDATION =====
    try:
        ffmpeg_version = check_ffmpeg()
        print(f"✅ FFmpeg: {ffmpeg_version}")
        check_ffprobe()
    except FFmpegError as e:
        print(f"❌ {e}")
        return False
    
    try:
        audio_duration = validate_audio_file(audio)
        print(f"✅ Audio: {Path(audio).name} ({audio_duration:.1f}s)")
    except AudioError as e:
        print(f"❌ {e}")
        return False
    
    audio_path = Path(audio)
    
    # Load configuration
    if isinstance(config, str):
        config = load_config(config)
    elif config is None:
        config = DEFAULT_CONFIG
    
    # Load custom colors if available
    custom_colors_path = Path(audio_path.parent) / "colors.json"
    if custom_colors_path.exists():
        load_custom_colors(custom_colors_path)
    
    # Resolution presets (9:10 portrait like reference)
    res_map = {
        '720p': (720, 800),
        '1080p': (1080, 1200),
        '2k': (1440, 1600),
        '4k': (2160, 2400),
        '4k+': (3456, 3840),
    }
    
    # Preview mode: lower resolution for speed
    if preview:
        res_map = {k: (v[0]//2, v[1]//2) for k, v in res_map.items()}
        fps = 30
        print("⚡ Preview mode: reduced resolution and framerate")
    
    w, h = res_map.get(resolution, (1080, 1200))
    colors = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES['purple'])
    
    print(f"\n📁 Audio: {audio_path.name}")
    print(f"📐 Resolution: {w}x{h} @ {fps}fps")
    print(f"🎨 Color Scheme: {color_scheme}")
    print(f"🎚️  Sensitivity: {sensitivity}")
    
    # GPU status
    if use_gpu and GPU_AVAILABLE:
        print(f"🚀 GPU: Enabled (CuPy)")
    elif use_gpu:
        print(f"⚠️  GPU: Not available, using CPU")
        use_gpu = False
    
    # Worker count
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    print(f"👷 Workers: {num_workers}")
    print()
    
    # ===== CHECKPOINTING =====
    checkpoint = RenderCheckpoint(str(audio_path), output, resolution, fps, color_scheme, sensitivity)
    
    if resume and checkpoint.exists():
        print(f"📌 Found checkpoint, attempting to resume...")
        checkpoint_data = checkpoint.load()
        if checkpoint_data:
            completed_frames = checkpoint.get_completed_frames()
            print(f"   ✅ {len(completed_frames)} frames already rendered")
        else:
            completed_frames = set()
    else:
        completed_frames = set()
        if checkpoint.exists():
            checkpoint.cleanup()
    
    # ===== DISK SPACE CHECK =====
    try:
        estimated_mb = estimate_disk_usage(w, h, int(audio_duration * fps), fps)
        free_mb = check_disk_space(Path(output).parent or Path('.'), required_mb=estimated_mb)
        if free_mb:
            print(f"💾 Disk: {free_mb:.0f}MB free, ~{estimated_mb:.0f}MB needed")
    except DiskSpaceError as e:
        print(f"❌ {e}")
        return False
    
    # ===== AUDIO ANALYSIS =====
    try:
        audio_analyzer = AudioAnalyzer(audio_path, fps=fps, sensitivity=sensitivity, config=config.get('audio_reactivity'))
        total_frames = audio_analyzer.total_frames
    except Exception as e:
        print(f"❌ Audio analysis failed: {e}")
        return False
    
    print(f"\n⏱️  Duration: {audio_analyzer.duration:.1f}s ({total_frames} frames)")
    if hasattr(audio_analyzer, 'bpm'):
        print(f"🎵 BPM: {audio_analyzer.bpm:.1f}")
    print()
    
    # Orb parameters
    base_radius = int(min(w, h) * 0.42)
    cx, cy = w // 2, h // 2
    
    # Initialize particle system
    particle_system = None
    if enable_particles and config.get('particles', {}).get('enabled', True):
        particle_system = ParticleSystem(w, h, config=config.get('particles'))
        print("✨ Particles: Enabled")
    
    # Get frame directory (checkpoint or temp)
    if resume and checkpoint.exists():
        temp_dir = checkpoint.get_frames_dir()
    else:
        temp_dir = checkpoint.get_frames_dir()
    
    print(f"📂 Frames: {temp_dir}")
    
    try:
        # Noise seeds for reproducibility
        noise_seeds = (42, 137, 256)
        
        # Pre-compute all audio parameters
        print("\n📊 Pre-computing audio parameters...")
        audio_params_list = [audio_analyzer.get_frame_params(i) for i in range(total_frames)]
        
        # Determine which frames need rendering
        frames_to_render = [i for i in range(total_frames) if i not in completed_frames]
        
        if len(frames_to_render) < total_frames:
            print(f"⏭️  Skipping {total_frames - len(frames_to_render)} already rendered frames")
        
        # Generate frames with multiprocessing
        print(f"\n🎨 Generating {len(frames_to_render)} frames with {num_workers} workers...")
        
        rendered = render_frames_parallel(
            frames_to_render, w, h, base_radius, cx, cy, noise_seeds, colors,
            fps, audio_params_list, temp_dir, num_workers=num_workers,
            particle_system=particle_system, audio_analyzer=audio_analyzer
        )
        
        # Save checkpoint
        checkpoint.save({
            'completed_frames': list(completed_frames | set(rendered)),
            'total_frames': total_frames,
            'params': {
                'resolution': resolution, 'fps': fps, 'color_scheme': color_scheme,
                'sensitivity': sensitivity
            }
        })
        
        print(f"\n🎬 Encoding video with FFmpeg...")
        
        # Combine frames with audio using FFmpeg
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%06d.png'),
            '-ss', str(start_time),
            '-i', str(audio),
            '-c:v', 'libx264',
            '-preset', 'medium' if preview else 'slow',
            '-crf', '23' if preview else '18',
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'high',
            '-level', '4.2',
            '-c:a', 'aac',
            '-b:a', '192k' if preview else '256k',
            '-shortest',
            '-movflags', '+faststart',
            output
        ]
        
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        if TQDM_AVAILABLE:
            pbar = tqdm(total=total_frames, desc="📹 Encoding", unit="frame")
            last_frame = 0
        
        for line in proc.stdout:
            if 'frame=' in line:
                match = re.search(r'frame=\s*(\d+)', line)
                if match:
                    encoded = int(match.group(1))
                    if TQDM_AVAILABLE:
                        pbar.update(encoded - last_frame)
                        last_frame = encoded
                    else:
                        pct = min(100, encoded / total_frames * 100)
                        print(f"\r   📹 Encoding: {encoded}/{total_frames} ({pct:.1f}%)", end='', flush=True)
        
        if TQDM_AVAILABLE:
            pbar.close()
        
        proc.wait()
        print()
        
        if proc.returncode == 0 and Path(output).exists():
            size = Path(output).stat().st_size / (1024 * 1024)
            print(f"\n✅ SUCCESS! {output} ({size:.1f} MB)")
            
            # Cleanup checkpoint on success
            checkpoint.cleanup()
            return True
        else:
            print("❌ FFmpeg encoding failed!")
            print("💡 Tip: Use --resume to continue from checkpoint")
            return False
            
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted! Progress saved to checkpoint.")
        print(f"💡 Use --resume to continue: python silk_visualizer.py {audio} --resume")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"💡 Use --resume to retry from checkpoint")
        raise
    finally:
        if not resume:
            # Only cleanup if not resuming
            pass  # Keep frames for potential resume


# ==================== BATCH PROCESSING ====================
def process_batch(audio_files, resolution='1080p', fps=60, color_scheme='purple', sensitivity=1.0, 
                  num_workers=None, use_gpu=False, enable_particles=True):
    """Process multiple audio files"""
    print("\n" + "="*60)
    print("   🎵 BATCH PROCESSING - AUDIO REACTIVE")
    print("="*60)
    print(f"\n📋 Files to process: {len(audio_files)}")
    
    results = []
    for i, audio in enumerate(audio_files, 1):
        print(f"\n{'─'*50}")
        print(f"📌 [{i}/{len(audio_files)}] Processing: {Path(audio).name}")
        print('─'*50)
        
        output = f"silk_{Path(audio).stem}.mp4"
        success = create_silk_orb_video(
            audio, output, resolution, fps, color_scheme, 
            sensitivity=sensitivity, num_workers=num_workers,
            use_gpu=use_gpu, enable_particles=enable_particles
        )
        results.append((audio, output, success))
    
    print("\n" + "="*60)
    print("   📊 BATCH RESULTS")
    print("="*60 + "\n")
    
    for audio, output, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {Path(audio).name} → {output}")
    
    return results


# ==================== CLI ====================
def create_parser():
    """Create argument parser with all options"""
    parser = argparse.ArgumentParser(
        prog='silk_visualizer',
        description="""
🔮 SILK FLUID ORB VISUALIZER - Audio-Reactive Beat-Synced Visualization

Creates beautiful fluid silk orb animations synchronized to music.
Bass → sphere pulses, Mids → flow speed, Highs → shimmer
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s song.mp3                          # Basic usage
  %(prog)s track.mp3 -r 2k -c lava           # 2K with lava colors
  %(prog)s audio.mp3 -r 4k --gpu             # 4K with GPU acceleration
  %(prog)s song.mp3 --preview                # Quick preview
  %(prog)s song.mp3 --resume                 # Resume interrupted render
  %(prog)s *.mp3 --batch                     # Process multiple files

Color Schemes: purple, lava, ocean, golden, emerald, volcano, cyber, sunset
Resolutions: 720p, 1080p, 2k, 4k, 4k+
        """
    )
    
    # Positional arguments
    parser.add_argument('audio', nargs='+', help='Audio file(s) to visualize')
    
    # Output options
    parser.add_argument('-o', '--output', help='Output video path (default: silk_<audio>.mp4)')
    
    # Quality options
    parser.add_argument('-r', '--resolution', default='1080p',
                       choices=['720p', '1080p', '2k', '4k', '4k+'],
                       help='Resolution preset (default: 1080p)')
    parser.add_argument('--fps', type=int, default=60,
                       help='Frame rate (default: 60)')
    parser.add_argument('-c', '--color', '--colors', dest='color_scheme', default='purple',
                       help=f'Color scheme (default: purple). Available: {", ".join(list_color_schemes())}')
    parser.add_argument('-s', '--sensitivity', type=float, default=1.0,
                       help='Audio reactivity (0.5-2.0, default: 1.0)')
    
    # Performance options
    parser.add_argument('-w', '--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto-detect)')
    parser.add_argument('--gpu', action='store_true',
                       help='Enable GPU acceleration (requires CuPy)')
    parser.add_argument('--no-particles', action='store_true',
                       help='Disable particle system')
    
    # Workflow options
    parser.add_argument('--preview', action='store_true',
                       help='Quick preview mode (lower quality, faster)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint if available')
    parser.add_argument('--batch', action='store_true',
                       help='Batch process multiple audio files')
    parser.add_argument('--start', type=float, default=0,
                       help='Start time offset in seconds (default: 0)')
    
    # Configuration
    parser.add_argument('--config', help='Path to JSON config file')
    parser.add_argument('--colors-file', help='Path to custom colors JSON file')
    parser.add_argument('--list-colors', action='store_true',
                       help='List available color schemes and exit')
    
    # Information
    parser.add_argument('-v', '--version', action='version', 
                       version='%(prog)s 2.0.0 (Enhanced with GPU, multiprocessing, particles)')
    
    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    
    # Handle --list-colors before parsing (doesn't require audio)
    if '--list-colors' in sys.argv:
        print("\n🎨 Available Color Schemes:")
        print("─" * 40)
        for name in list_color_schemes():
            colors = COLOR_SCHEMES[name]
            glow = colors['glow']
            print(f"  • {name:<12} (glow: RGB{glow})")
        print("\n💡 Use --colors-file to load custom palettes from JSON")
        sys.exit(0)
    
    # Handle no arguments
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    # List colors and exit (redundant but kept for safety)
    if args.list_colors:
        print("\n🎨 Available Color Schemes:")
        print("─" * 40)
        for name in list_color_schemes():
            colors = COLOR_SCHEMES[name]
            glow = colors['glow']
            print(f"  • {name:<12} (glow: RGB{glow})")
        print("\n💡 Use --colors-file to load custom palettes from JSON")
        sys.exit(0)
    
    # Load custom colors if specified
    if args.colors_file:
        load_custom_colors(args.colors_file)
    
    # Validate color scheme
    if args.color_scheme not in COLOR_SCHEMES:
        print(f"❌ Unknown color scheme: {args.color_scheme}")
        print(f"   Available: {', '.join(list_color_schemes())}")
        sys.exit(1)
    
    # Filter valid audio files
    audio_files = [f for f in args.audio if Path(f).exists()]
    
    if not audio_files:
        print(f"❌ No valid audio files found!")
        sys.exit(1)
    
    # Batch mode or single file
    if args.batch or len(audio_files) > 1:
        process_batch(
            audio_files,
            resolution=args.resolution,
            fps=args.fps,
            color_scheme=args.color_scheme,
            sensitivity=args.sensitivity,
            num_workers=args.workers,
            use_gpu=args.gpu,
            enable_particles=not args.no_particles
        )
    else:
        # Single file
        audio = audio_files[0]
        output = args.output or f"silk_{Path(audio).stem}.mp4"
        
        create_silk_orb_video(
            audio=audio,
            output=output,
            resolution=args.resolution,
            fps=args.fps,
            color_scheme=args.color_scheme,
            start_time=args.start,
            sensitivity=args.sensitivity,
            num_workers=args.workers,
            use_gpu=args.gpu,
            enable_particles=not args.no_particles,
            resume=args.resume,
            preview=args.preview,
            config=args.config
        )


if __name__ == '__main__':
    # Multiprocessing support for Windows
    mp.freeze_support()
    main()
