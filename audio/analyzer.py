"""
AudioAnalyzer — Load an audio file, run frequency analysis and beat detection,
and expose per-frame feature arrays consumed by the physics engine each frame.

All heavy analysis is done once in __init__ (upfront), so the render loop
can call get_frame(i) in O(1) time.
"""

from __future__ import annotations

import librosa
import numpy as np


class AudioAnalyzer:
    """Analyze audio and expose per-frame feature data as numpy arrays."""

    def __init__(
        self, filepath: str, fps: int = 30, sr: int = 22050, config: dict | None = None
    ):
        self.fps = fps
        self.filepath = filepath
        cfg = config or {}

        # EMA smoothing coefficients (fall back to good defaults if not in config)
        smooth_bass = cfg.get("smoothing_bass", 0.75)
        smooth_mids = cfg.get("smoothing_mids", 0.65)
        smooth_highs = cfg.get("smoothing_highs", 0.55)
        smooth_rms = cfg.get("smoothing_rms", 0.70)

        self._analyze(
            filepath, fps, sr, smooth_bass, smooth_mids, smooth_highs, smooth_rms
        )

    # ── Analysis ────────────────────────────────────────────────────────────

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
        # 1. Load audio (resampled to sr, mono)
        y, sr = librosa.load(filepath, sr=sr, mono=True)
        self.sr = sr

        # 2. Hop length = samples per video frame
        hop_length = sr // fps

        # 3. Mel spectrogram (128 bands, better perceptual accuracy than raw FFT)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=hop_length, n_mels=128
        )
        # shape: (128, n_frames)

        # 4. Beat tracking
        _tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)

        # 5. Onset strength envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

        # 6. RMS amplitude
        rms_raw = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        # 7. Band energies from mel spectrogram
        #    Bass:  bands  0–10  → ~20–200 Hz
        #    Mids:  bands 10–60  → ~200–2000 Hz
        #    Highs: bands 60–128 → ~2000–16 000 Hz
        bass_raw = mel[0:10, :].mean(axis=0)
        mids_raw = mel[10:60, :].mean(axis=0)
        highs_raw = mel[60:, :].mean(axis=0)

        # 8. Determine consistent n_frames (trim all arrays to shortest)
        n_frames = min(
            len(rms_raw),
            len(onset_env),
            bass_raw.shape[0],
            mids_raw.shape[0],
            highs_raw.shape[0],
        )

        rms_raw = rms_raw[:n_frames]
        bass_raw = bass_raw[:n_frames]
        mids_raw = mids_raw[:n_frames]
        highs_raw = highs_raw[:n_frames]
        onset_env = onset_env[:n_frames]

        # 9. Normalize each feature 0–1 (robust: clip at 97th percentile)
        bass_n = self._normalize(bass_raw)
        mids_n = self._normalize(mids_raw)
        highs_n = self._normalize(highs_raw)
        rms_n = self._normalize(rms_raw)
        onset_n = self._normalize(onset_env)

        # 10. Apply exponential moving average (EMA) smoothing
        self.bass = self._smooth(bass_n, smooth_bass)
        self.mids = self._smooth(mids_n, smooth_mids)
        self.highs = self._smooth(highs_n, smooth_highs)
        self.rms = self._smooth(rms_n, smooth_rms)
        self.onset = onset_n  # onset kept unsmoothed for snappy beat response

        # 11. Boolean beat array
        is_beat = np.zeros(n_frames, dtype=bool)
        valid = beat_frames[beat_frames < n_frames]
        is_beat[valid] = True
        self.is_beat = is_beat

        self.n_frames = n_frames
        self.duration = float(len(y)) / sr

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(arr: np.ndarray, percentile: float = 97.0) -> np.ndarray:
        """Clip at 97th percentile, then scale 0–1 to avoid outliers dominating."""
        peak = np.percentile(arr, percentile)
        return np.clip(arr / (peak + 1e-8), 0.0, 1.0)

    @staticmethod
    def _smooth(arr: np.ndarray, alpha: float) -> np.ndarray:
        """Exponential moving average.  alpha=0 → raw values,  alpha=0.9 → heavy smoothing."""
        if alpha <= 0.0:
            return arr.copy()
        out = np.empty_like(arr, dtype=np.float32)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * out[i - 1] + (1.0 - alpha) * arr[i]
        return out

    # ── Public API ──────────────────────────────────────────────────────────

    def get_frame(self, frame_idx: int) -> dict:
        """Return a dict of audio features for the given frame index (safe, clamped)."""
        idx = max(0, min(frame_idx, self.n_frames - 1))
        return {
            "rms": float(self.rms[idx]),
            "bass": float(self.bass[idx]),
            "mids": float(self.mids[idx]),
            "highs": float(self.highs[idx]),
            "onset": float(self.onset[idx]),
            "is_beat": bool(self.is_beat[idx]),
        }
