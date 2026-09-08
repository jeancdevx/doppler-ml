"""Lectura de audio y características con librosa, o scipy si no está disponible."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import librosa

    HAS_LIBROSA = True
except Exception:  # pragma: no cover - entorno sin ruedas de librosa
    HAS_LIBROSA = False
    librosa = None  # type: ignore

from scipy.io import wavfile
from scipy.signal import get_window, spectrogram, stft

try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except Exception:
    HAS_SOUNDFILE = False
    sf = None  # type: ignore


TARGET_SR = 22050
AUDIOSET_DURATION_S = 10.0


@dataclass
class AudioClip:
    y: np.ndarray
    sr: int


def peak_normalize(y: np.ndarray, peak: float = 0.99) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    max_abs = float(np.max(np.abs(y))) if y.size else 0.0
    if max_abs < 1e-8:
        return y
    return (y / max_abs) * peak


def rms_normalize(y: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
    if rms < 1e-8:
        return y
    return y * (target_rms / rms)


def pad_or_trim(y: np.ndarray, sr: int, duration_s: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    n = int(round(duration_s * sr))
    if n <= 0:
        return y
    if len(y) >= n:
        return y[:n]
    out = np.zeros(n, dtype=np.float32)
    out[: len(y)] = y
    return out


def load_audio(path: str, sr: int = TARGET_SR, duration: float | None = None) -> AudioClip:
    if HAS_LIBROSA:
        y, out_sr = librosa.load(path, sr=sr, mono=True, duration=duration)
        return AudioClip(y=np.asarray(y, dtype=np.float32), sr=out_sr)

    if HAS_SOUNDFILE:
        y, file_sr = sf.read(path, always_2d=False)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if duration is not None:
            y = y[: int(duration * file_sr)]
        if sr and file_sr != sr:
            duration_s = len(y) / float(file_sr)
            n_out = max(1, int(duration_s * sr))
            x_old = np.linspace(0.0, duration_s, num=len(y), endpoint=False)
            x_new = np.linspace(0.0, duration_s, num=n_out, endpoint=False)
            y = np.interp(x_new, x_old, y).astype(np.float32)
            file_sr = sr
        return AudioClip(y=y, sr=int(file_sr))

    file_sr, data = wavfile.read(path)
    y = np.asarray(data, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    max_abs = np.max(np.abs(y)) or 1.0
    if max_abs > 1.5:
        y = y / 32768.0
    if duration is not None:
        y = y[: int(duration * file_sr)]
    if sr and file_sr != sr:
        duration_s = len(y) / file_sr
        n_out = int(duration_s * sr)
        x_old = np.linspace(0.0, duration_s, num=len(y), endpoint=False)
        x_new = np.linspace(0.0, duration_s, num=n_out, endpoint=False)
        y = np.interp(x_new, x_old, y).astype(np.float32)
        file_sr = sr
    return AudioClip(y=y, sr=file_sr)


def duration_seconds(path: str) -> float:
    return float(wav_probe(path)["duration_s"])


def wav_probe(path: str) -> dict:
    """Metadatos baratos sin resamplear."""
    if HAS_SOUNDFILE:
        info = sf.info(path)
        return {
            "sr": int(info.samplerate),
            "n_channels": int(info.channels),
            "n_samples": int(info.frames),
            "duration_s": float(info.duration),
            "dtype": str(info.subtype),
        }
    try:
        sr, data = wavfile.read(path)
        n_channels = 1 if np.asarray(data).ndim == 1 else np.asarray(data).shape[1]
        n_samples = int(np.asarray(data).shape[0])
        return {
            "sr": int(sr),
            "n_channels": int(n_channels),
            "n_samples": n_samples,
            "duration_s": n_samples / float(sr),
            "dtype": str(np.asarray(data).dtype),
        }
    except Exception:
        if HAS_LIBROSA:
            y, sr = librosa.load(path, sr=None, mono=False)
            y = np.asarray(y)
            n_channels = 1 if y.ndim == 1 else y.shape[0]
            n_samples = y.shape[-1]
            return {
                "sr": int(sr),
                "n_channels": int(n_channels),
                "n_samples": int(n_samples),
                "duration_s": n_samples / float(sr),
                "dtype": str(y.dtype),
            }
        raise


def log_mel_spectrogram(clip: AudioClip, n_mels: int = 64, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    if HAS_LIBROSA:
        S = librosa.feature.melspectrogram(y=clip.y, sr=clip.sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop)
        return librosa.power_to_db(S, ref=np.max)
    f, t, Sxx = spectrogram(clip.y, fs=clip.sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    # Aproximación: filtro triangular en Hz de Mel.
    mel_f = _hz_to_mel(f)
    edges = np.linspace(mel_f.min(), mel_f.max(), n_mels + 2)
    mels = np.zeros((n_mels, Sxx.shape[1]), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        w = np.zeros_like(mel_f)
        left = np.logical_and(mel_f >= lo, mel_f <= mid)
        right = np.logical_and(mel_f >= mid, mel_f <= hi)
        if np.any(left):
            w[left] = (mel_f[left] - lo) / max(mid - lo, 1e-8)
        if np.any(right):
            w[right] = (hi - mel_f[right]) / max(hi - mid, 1e-8)
        mels[i] = w @ Sxx
    mels = np.maximum(mels, 1e-10)
    return 10.0 * np.log10(mels / np.max(mels))


def mfcc(clip: AudioClip, n_mfcc: int = 13) -> np.ndarray:
    if HAS_LIBROSA:
        return librosa.feature.mfcc(y=clip.y, sr=clip.sr, n_mfcc=n_mfcc)
    log_mel = log_mel_spectrogram(clip, n_mels=40)
    # DCT tipo II sobre el eje mel.
    n_mels, n_frames = log_mel.shape
    n = np.arange(n_mels)
    k = np.arange(n_mfcc)[:, None]
    dct = np.cos(np.pi * k * (2 * n + 1) / (2.0 * n_mels))
    return dct @ log_mel


def spectral_centroid(clip: AudioClip) -> np.ndarray:
    if HAS_LIBROSA:
        return librosa.feature.spectral_centroid(y=clip.y, sr=clip.sr)[0]
    f, _, Zxx = stft(clip.y, fs=clip.sr, nperseg=1024)
    mag = np.abs(Zxx)
    denom = np.sum(mag, axis=0) + 1e-10
    return (f[:, None] * mag).sum(axis=0) / denom


def spectral_bandwidth(clip: AudioClip) -> np.ndarray:
    if HAS_LIBROSA:
        return librosa.feature.spectral_bandwidth(y=clip.y, sr=clip.sr)[0]
    f, _, Zxx = stft(clip.y, fs=clip.sr, nperseg=1024)
    mag = np.abs(Zxx)
    denom = np.sum(mag, axis=0) + 1e-10
    centroid = (f[:, None] * mag).sum(axis=0) / denom
    var = ((f[:, None] - centroid) ** 2 * mag).sum(axis=0) / denom
    return np.sqrt(var)


def spectral_rolloff(clip: AudioClip, roll_percent: float = 0.85) -> np.ndarray:
    if HAS_LIBROSA:
        return librosa.feature.spectral_rolloff(y=clip.y, sr=clip.sr, roll_percent=roll_percent)[0]
    f, _, Zxx = stft(clip.y, fs=clip.sr, nperseg=1024)
    mag = np.abs(Zxx)
    csum = np.cumsum(mag, axis=0)
    thresh = roll_percent * (csum[-1] + 1e-10)
    idx = np.argmax(csum >= thresh, axis=0)
    return f[idx]


def zero_crossing_rate(clip: AudioClip, frame_length: int = 2048, hop: int = 512) -> np.ndarray:
    if HAS_LIBROSA:
        return librosa.feature.zero_crossing_rate(clip.y, frame_length=frame_length, hop_length=hop)[0]
    y = clip.y
    n = 1 + max(0, (len(y) - frame_length) // hop)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        frame = y[i * hop : i * hop + frame_length]
        out[i] = np.mean(np.abs(np.diff(np.signbit(frame))))
    return out


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def stft_db(clip: AudioClip, n_fft: int = 1024, hop: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if HAS_LIBROSA:
        S = np.abs(librosa.stft(clip.y, n_fft=n_fft, hop_length=hop))
        db = librosa.amplitude_to_db(S, ref=np.max)
        freqs = librosa.fft_frequencies(sr=clip.sr, n_fft=n_fft)
        times = librosa.frames_to_time(np.arange(db.shape[1]), sr=clip.sr, hop_length=hop)
        return freqs, times, db
    window = get_window("hann", n_fft)
    f, t, Zxx = stft(clip.y, fs=clip.sr, window=window, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.abs(Zxx)
    db = 20.0 * np.log10(np.maximum(mag, 1e-10) / (np.max(mag) + 1e-10))
    return f, t, db
