"""
AudioAnalyzer — Load audio, run frequency analysis, expose per-frame features.

Implemented with scipy + soundfile + ffmpeg only (no librosa / numba required).
This ensures compatibility with Python 3.14+ where numba wheels are not available.

Pipeline:
  1. Load audio via soundfile (WAV/FLAC/OGG) or ffmpeg subprocess (MP3/AAC/etc.)
  2. Compute STFT with scipy.signal.stft
  3. Extract bass / mids / highs energy per frame from linear frequency bins
  4. Compute RMS amplitude per frame
  5. Compute onset strength (spectral flux)
  6. Detect beat frames with scipy peak-finding
  7. Normalise all features 0–1 and apply EMA smoothing
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks
from scipy.signal import stft as scipy_stft


class AudioAnalyzer:
    """Load and analyse audio; expose per-frame feature arrays."""

    def __init__(
        self,
        filepath: str,
        fps: int = 30,
        sr: int = 22050,
        config: Optional[dict] = None,
    ):
        self.fps = fps
        self.filepath = filepath
        cfg = config or {}

        smooth_bass = cfg.get("smoothing_bass", 0.75)
        smooth_mids = cfg.get("smoothing_mids", 0.65)
        smooth_highs = cfg.get("smoothing_highs", 0.55)
        smooth_rms = cfg.get("smoothing_rms", 0.70)

        self._analyze(
            filepath, fps, sr, smooth_bass, smooth_mids, smooth_highs, smooth_rms
        )

    # ── Audio loading ────────────────────────────────────────────────────────

    @staticmethod
    def _load_audio(filepath: str, sr: int) -> tuple[np.ndarray, int]:
        """
        Load any audio file to a mono float32 waveform at sample rate `sr`.

        WAV / FLAC / OGG: read directly with soundfile.
        MP3 / AAC / M4A and anything else: decode via ffmpeg to a temp WAV.
        """
        ext = os.path.splitext(filepath)[1].lower()
        sf_formats = {".wav", ".flac", ".ogg", ".aiff", ".aif"}

        if ext in sf_formats:
            y, file_sr = sf.read(filepath, dtype="float32", always_2d=False)
        else:
            # Decode with ffmpeg → temp PCM WAV
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()
            try:
                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        filepath,
                        "-ar",
                        str(sr),
                        "-ac",
                        "1",
                        "-f",
                        "wav",
                        tmp_path,
                    ],
                    capture_output=True,
                    timeout=120,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"ffmpeg failed to decode {filepath}:\n"
                        + result.stderr.decode("utf-8", errors="replace")[-400:]
                    )
                y, file_sr = sf.read(tmp_path, dtype="float32", always_2d=False)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # Ensure mono
        if y.ndim > 1:
            y = y.mean(axis=1)

        # Resample if needed
        if file_sr != sr:
            y = _resample(y, file_sr, sr)

        return y.astype(np.float32), sr

    # ── Analysis ─────────────────────────────────────────────────────────────

    def _analyze(
        self,
        filepath: str,
        fps: int,
        sr: int,
        smooth_bass: float,
        smooth_mids: float,
        smooth_highs: float,
        smooth_rms: float,
    ) -> None:
        y, sr = self._load_audio(filepath, sr)
        self.sr = sr

        hop_length = sr // fps  # samples per video frame
        n_fft = 2048

        # ── STFT ────────────────────────────────────────────────────────────
        # scipy.signal.stft returns (freqs, time, complex coefficients)
        # shape of Zxx: (n_fft//2 + 1, n_time_steps)
        _freqs, _times, Zxx = scipy_stft(
            y,
            fs=sr,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            boundary=None,
            padded=False,
        )
        magnitude = np.abs(Zxx).astype(np.float32)  # (n_freqs, n_stft_frames)

        n_frames = len(y) // hop_length  # target frame count

        # ── Frequency bands (linear STFT bins) ──────────────────────────────
        # Bin k corresponds to frequency k * sr / n_fft
        freq_res = sr / n_fft
        bass_end = max(1, int(200 / freq_res))  # 0 → ~200 Hz
        mids_end = max(bass_end + 1, int(2000 / freq_res))  # ~200 → ~2000 Hz

        bass_raw = magnitude[:bass_end, :].mean(axis=0)
        mids_raw = magnitude[bass_end:mids_end, :].mean(axis=0)
        highs_raw = magnitude[mids_end:, :].mean(axis=0)

        # ── RMS per frame ────────────────────────────────────────────────────
        n_rms_frames = len(y) // hop_length
        rms_raw = np.array(
            [
                float(np.sqrt(np.mean(y[i * hop_length : (i + 1) * hop_length] ** 2)))
                for i in range(n_rms_frames)
            ],
            dtype=np.float32,
        )

        # ── Onset strength (spectral flux — sum of positive magnitude changes) ──
        if magnitude.shape[1] > 1:
            flux = np.maximum(np.diff(magnitude, axis=1), 0).sum(axis=0)
            onset_raw = np.concatenate([[0.0], flux]).astype(np.float32)
        else:
            onset_raw = np.zeros(magnitude.shape[1], dtype=np.float32)

        # ── Trim all arrays to n_frames ──────────────────────────────────────
        n_frames = min(
            n_frames,
            bass_raw.shape[0],
            mids_raw.shape[0],
            highs_raw.shape[0],
            len(rms_raw),
            len(onset_raw),
        )
        bass_raw = bass_raw[:n_frames]
        mids_raw = mids_raw[:n_frames]
        highs_raw = highs_raw[:n_frames]
        rms_raw = rms_raw[:n_frames]
        onset_raw = onset_raw[:n_frames]

        # ── Normalise 0–1 ────────────────────────────────────────────────────
        bass_n = self._normalize(bass_raw)
        mids_n = self._normalize(mids_raw)
        highs_n = self._normalize(highs_raw)
        rms_n = self._normalize(rms_raw)
        onset_n = self._normalize(onset_raw)

        # ── EMA smoothing ────────────────────────────────────────────────────
        self.bass = self._smooth(bass_n, smooth_bass)
        self.mids = self._smooth(mids_n, smooth_mids)
        self.highs = self._smooth(highs_n, smooth_highs)
        self.rms = self._smooth(rms_n, smooth_rms)
        self.onset = onset_n  # keep onset unsmoothed for snappy beat response

        # ── Beat detection (peak-finding on onset) ───────────────────────────
        # Minimum distance between beats: 0.25 s at the target fps
        min_beat_dist = max(1, int(0.25 * fps))
        peaks, _ = find_peaks(
            onset_n,
            distance=min_beat_dist,
            prominence=0.15,
            height=0.10,
        )
        is_beat = np.zeros(n_frames, dtype=bool)
        is_beat[peaks[peaks < n_frames]] = True
        self.is_beat = is_beat

        self.n_frames = n_frames
        self.duration = float(len(y)) / sr

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(arr: np.ndarray, percentile: float = 97.0) -> np.ndarray:
        """Clip at the 97th percentile then scale 0–1 to avoid outliers dominating."""
        peak = np.percentile(arr, percentile)
        return np.clip(arr / (peak + 1e-8), 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _smooth(arr: np.ndarray, alpha: float) -> np.ndarray:
        """Exponential moving average.  alpha=0 → no smoothing,  0.9 → heavy."""
        if alpha <= 0.0:
            return arr.copy()
        out = np.empty_like(arr, dtype=np.float32)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * out[i - 1] + (1.0 - alpha) * arr[i]
        return out

    # ── Public API ───────────────────────────────────────────────────────────

    def get_frame(self, frame_idx: int) -> dict:
        """Return audio feature dict for the given frame (safe, index clamped)."""
        idx = max(0, min(frame_idx, self.n_frames - 1))
        return {
            "rms": float(self.rms[idx]),
            "bass": float(self.bass[idx]),
            "mids": float(self.mids[idx]),
            "highs": float(self.highs[idx]),
            "onset": float(self.onset[idx]),
            "is_beat": bool(self.is_beat[idx]),
        }


# ── Module-level helpers ──────────────────────────────────────────────────────


def _resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear-interpolation resample (good enough for visualisation)."""
    if orig_sr == target_sr:
        return y
    target_len = int(len(y) * target_sr / orig_sr)
    return np.interp(
        np.linspace(0, len(y) - 1, target_len),
        np.arange(len(y)),
        y,
    ).astype(np.float32)
