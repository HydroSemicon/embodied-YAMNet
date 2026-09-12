from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def to_mono(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 1:
        return x
    if x.ndim != 2:
        raise ValueError("audio must have shape [samples] or [samples, channels]")
    return x.mean(axis=1, dtype=np.float32)


def resample(samples: np.ndarray, source_rate: int, target_rate: int = 16000) -> np.ndarray:
    x = to_mono(samples)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if source_rate == target_rate or len(x) == 0:
        return x.copy()
    out_len = int(round(len(x) * target_rate / source_rate))
    old = np.arange(len(x), dtype=np.float64)
    new = np.arange(out_len, dtype=np.float64) * source_rate / target_rate
    return np.interp(new, old, x, left=float(x[0]), right=float(x[-1])).astype(np.float32)


def read_wav(path: str | Path, target_rate: int = 16000) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels, width, rate, frames = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
        raw = wav.readframes(frames)
    if width == 1:
        data = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported PCM sample width: {width}")
    if channels > 1:
        data = data.reshape(-1, channels)
    return resample(data, rate, target_rate), target_rate


@dataclass(frozen=True)
class AudioQuality:
    rms: float
    dc_offset: float
    clipped_fraction: float
    silent: bool
    poor: bool
    reasons: tuple[str, ...]


def inspect_quality(samples: np.ndarray, min_rms: float = 0.0005, max_abs_dc: float = 0.05,
                    max_clipped_fraction: float = 0.01) -> AudioQuality:
    x = to_mono(samples)
    rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
    dc = float(np.mean(x)) if len(x) else 0.0
    clipped = float(np.mean(np.abs(x) >= 0.999)) if len(x) else 0.0
    reasons = []
    if rms < min_rms:
        reasons.append("silent")
    if abs(dc) > max_abs_dc:
        reasons.append("dc_offset")
    if clipped > max_clipped_fraction:
        reasons.append("clipping")
    return AudioQuality(rms, dc, clipped, rms < min_rms, bool(reasons), tuple(reasons))


class RingBuffer:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._data = np.zeros(capacity, dtype=np.float32)
        self._write = 0
        self._size = 0

    def append(self, samples: np.ndarray) -> None:
        x = to_mono(samples)
        if len(x) >= self.capacity:
            x = x[-self.capacity:]
        n = len(x)
        if not n:
            return
        end = self._write + n
        first = min(n, self.capacity - self._write)
        self._data[self._write:self._write + first] = x[:first]
        if first < n:
            self._data[:n - first] = x[first:]
        self._write = end % self.capacity
        self._size = min(self.capacity, self._size + n)

    def latest(self, length: int, pad: bool = True) -> np.ndarray:
        if length <= 0 or length > self.capacity:
            raise ValueError("length must be in (0, capacity]")
        available = min(length, self._size)
        start = (self._write - available) % self.capacity
        if start + available <= self.capacity:
            out = self._data[start:start + available].copy()
        else:
            out = np.concatenate((self._data[start:], self._data[:(start + available) % self.capacity]))
        if pad and available < length:
            out = np.pad(out, (length - available, 0))
        return out.astype(np.float32, copy=False)


def iter_windows(samples: np.ndarray, window_samples: int, hop_samples: int):
    x = to_mono(samples)
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window and hop must be positive")
    for end in range(hop_samples, len(x) + hop_samples, hop_samples):
        clipped_end = min(end, len(x))
        start = max(0, clipped_end - window_samples)
        window = x[start:clipped_end]
        if len(window) < window_samples:
            window = np.pad(window, (window_samples - len(window), 0))
        yield clipped_end / 16000.0, window.astype(np.float32)
        if clipped_end == len(x):
            break

