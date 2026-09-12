from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import LABELS
from .audio import read_wav
from .manifest import ManifestItem, leakage_report
from .models import BeatsExtractor, EmbeddingExtractor, MockExtractor, YamnetExtractor, train_linear_head


def label_vector(labels: tuple[str, ...] | list[str]) -> np.ndarray:
    selected = set(labels)
    return np.asarray([label in selected for label in LABELS], np.float32)


def augment_waveform(x: np.ndarray, rng: np.random.Generator, settings: dict,
                     background: np.ndarray | None = None, rir: np.ndarray | None = None) -> np.ndarray:
    """Conservative waveform augmentation; only call on training split."""
    y = np.asarray(x, np.float32).copy()
    gain_db = float(settings.get("gain_db", 0.0))
    if gain_db:
        y *= 10 ** (rng.uniform(-gain_db, gain_db) / 20)
    shift_ms = int(settings.get("time_shift_ms", 0))
    if shift_ms:
        y = np.roll(y, int(rng.uniform(-shift_ms, shift_ms) * 16))
    stretch = float(settings.get("time_stretch", 0.0))
    if stretch:
        factor = rng.uniform(1 - stretch, 1 + stretch)
        positions = np.arange(len(y), dtype=np.float64) * factor
        y = np.interp(positions, np.arange(len(y)), y, left=0, right=0).astype(np.float32)
    if background is not None and float(settings.get("background_mix", 0)) > 0:
        bg = np.resize(np.asarray(background, np.float32), len(y))
        scale = rng.uniform(0, float(settings["background_mix"]))
        y = y + bg * scale
    if rir is not None and bool(settings.get("rir", False)):
        y = np.convolve(y, np.asarray(rir, np.float32), mode="full")[:len(y)]
    # Waveform time/frequency masking analogue, disabled by default.
    mask_fraction = float(settings.get("specaugment_mask_fraction", 0.0))
    if mask_fraction > 0 and len(y):
        length = int(len(y) * rng.uniform(0, mask_fraction))
        start = int(rng.integers(0, max(1, len(y) - length + 1)))
        y[start:start + length] = 0
    return np.clip(y, -1, 1).astype(np.float32)


def make_extractor(name: str, checkpoint: str | None = None, beats_source: str | None = None,
                   device: str = "cpu") -> EmbeddingExtractor:
    if name == "mock":
        return MockExtractor()
    if name == "yamnet":
        return YamnetExtractor()
    if name == "beats":
        if not checkpoint or not beats_source:
            raise ValueError("BEATs requires --checkpoint and --beats-source")
        return BeatsExtractor(checkpoint, beats_source, device)
    raise ValueError(f"unknown extractor: {name}")


def collect_embeddings(items: list[ManifestItem], split: str, extractor: EmbeddingExtractor,
                       window_seconds: float = 2.0, augmentation: dict | None = None, seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    leak = leakage_report(items)
    if not leak["ok"]:
        raise ValueError(f"refusing to train with leakage: {leak['leaks']}")
    rng, rows_x, rows_y = np.random.default_rng(seed), [], []
    background_paths = [item.path for item in items if item.split == "train" and not item.labels and item.is_hard_negative]
    rir_paths = [Path(p) for p in (augmentation or {}).get("rir_paths", [])]
    window = int(window_seconds * 16000)
    for item in items:
        if item.split != split:
            continue
        waveform, _ = read_wav(item.path)
        if augmentation and split == "train":
            background = read_wav(background_paths[int(rng.integers(len(background_paths)))])[0] if background_paths else None
            rir = read_wav(rir_paths[int(rng.integers(len(rir_paths)))])[0] if rir_paths else None
            waveform = augment_waveform(waveform, rng, augmentation, background, rir)
        if len(waveform) < window:
            clips = [np.pad(waveform, (0, window - len(waveform)))]
        else:
            clips = [waveform[start:start + window] for start in range(0, len(waveform) - window + 1, window)]
        target = label_vector(item.labels)
        for clip in clips:
            rows_x.append(extractor.extract(clip))
            rows_y.append(target)
    if not rows_x:
        raise ValueError(f"no items in split {split!r}")
    return np.stack(rows_x), np.stack(rows_y)


def run_training(items: list[ManifestItem], extractor: EmbeddingExtractor, output_dir: str | Path,
                 epochs: int, learning_rate: float, augmentation: dict | None, checkpoint: str) -> dict:
    x, y = collect_embeddings(items, "train", extractor, augmentation=augmentation)
    head, losses = train_linear_head(x, y, epochs=epochs, learning_rate=learning_rate)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "architecture": f"{extractor.name}_transfer", "extractor": extractor.name,
        "checkpoint": checkpoint, "labels": list(LABELS), "embedding_dimension": int(x.shape[1]),
        "training_examples": int(len(x)), "epochs": epochs, "seed": 17,
        "ood_center": x.mean(axis=0).astype(float).tolist(),
        "ood_scale": np.maximum(x.std(axis=0), 1e-4).astype(float).tolist(),
    }
    version = head.save(out / "head.npz", metadata)
    metadata["classifier_version"] = version
    (out / "training.json").write_text(json.dumps({**metadata, "final_loss": losses[-1], "loss": losses}, indent=2), encoding="utf-8")
    return {**metadata, "final_loss": losses[-1], "head": str(out / "head.npz")}
