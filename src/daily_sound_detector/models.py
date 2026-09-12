from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from . import LABELS


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), -40, 40)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


class Scorer(ABC):
    architecture: str
    checkpoint: str
    classifier_version: str

    @abstractmethod
    def scores(self, short_window: np.ndarray, long_window: np.ndarray) -> np.ndarray:
        """Return five independent sigmoid scores in LABELS order."""


class EmbeddingExtractor(ABC):
    name: str
    dimension: int

    @abstractmethod
    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """Return a fixed-dimensional embedding."""


class MockExtractor(EmbeddingExtractor):
    """Small deterministic extractor for CI, smoke tests, and pipeline validation only."""
    name, dimension = "mock", 16
    device = "cpu"

    def extract(self, waveform: np.ndarray) -> np.ndarray:
        x = np.asarray(waveform, dtype=np.float32)
        if not len(x):
            return np.zeros(self.dimension, np.float32)
        chunks = np.array_split(x, 8)
        means = [float(np.mean(c)) if len(c) else 0.0 for c in chunks]
        rms = [float(np.sqrt(np.mean(c * c))) if len(c) else 0.0 for c in chunks]
        return np.asarray(means + rms, dtype=np.float32)


class YamnetExtractor(EmbeddingExtractor):
    name, dimension = "yamnet", 1024
    HUB_URL = "https://tfhub.dev/google/yamnet/1"

    def __init__(self):
        try:
            import tensorflow_hub as hub
        except ImportError as exc:
            raise RuntimeError("install the 'yamnet' extra: pip install -e .[yamnet]") from exc
        self._model = hub.load(self.HUB_URL)
        self.device = "gpu" if __import__("tensorflow").config.list_physical_devices("GPU") else "cpu"

    def extract(self, waveform: np.ndarray) -> np.ndarray:
        _, embeddings, _ = self._model(np.asarray(waveform, np.float32))
        value = np.asarray(embeddings)
        return value.mean(axis=0).astype(np.float32) if len(value) else np.zeros(self.dimension, np.float32)


class YamnetBaseline(Scorer):
    architecture, checkpoint, classifier_version = "yamnet_standard", "yamnet/1", "audioset-map-v1"
    HUB_URL = YamnetExtractor.HUB_URL
    MAP = {
        "cough": ("Cough",), "sneeze": ("Sneeze",), "laughter": ("Laughter", "Giggle"),
        "alarm": ("Alarm", "Smoke detector, smoke alarm", "Fire alarm", "Siren"),
        "timer_or_ringtone": ("Ringtone", "Alarm clock", "Telephone bell ringing", "Ding-dong"),
    }

    def __init__(self):
        try:
            import csv
            import tensorflow_hub as hub
        except ImportError as exc:
            raise RuntimeError("install the 'yamnet' extra: pip install -e .[yamnet]") from exc
        self._model = hub.load(self.HUB_URL)
        self.model_size_bytes = None
        self.device = "gpu" if __import__("tensorflow").config.list_physical_devices("GPU") else "cpu"
        class_map = self._model.class_map_path().numpy().decode("utf-8")
        with open(class_map, newline="", encoding="utf-8") as handle:
            names = [row["display_name"] for row in csv.DictReader(handle)]
        self._indices = {label: [i for i, name in enumerate(names) if name in mapped] for label, mapped in self.MAP.items()}

    def scores(self, short_window: np.ndarray, long_window: np.ndarray) -> np.ndarray:
        raw_short, _, _ = self._model(np.asarray(short_window, np.float32))
        raw_long, _, _ = self._model(np.asarray(long_window, np.float32))
        means = [np.asarray(raw_short).mean(axis=0), np.asarray(raw_long).mean(axis=0)]
        mapped = [[max((float(mean[i]) for i in self._indices[label]), default=0.0) for label in LABELS] for mean in means]
        return np.asarray([mapped[0][0], mapped[0][1], max(mapped[0][2], mapped[1][2]),
                           max(mapped[0][3], mapped[1][3]), max(mapped[0][4], mapped[1][4])], np.float32)


class BeatsExtractor(EmbeddingExtractor):
    """Adapter for the official microsoft/unilm BEATs.py plus a local official checkpoint."""
    name, dimension = "beats", 768

    def __init__(self, checkpoint: str | Path, beats_source: str | Path, device: str = "cpu"):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("install the 'beats' extra: pip install -e .[beats]") from exc
        checkpoint, beats_source = Path(checkpoint), Path(beats_source)
        if not checkpoint.is_file() or not beats_source.is_file():
            raise FileNotFoundError("BEATs checkpoint and official BEATs.py must exist locally")
        sys.path.insert(0, str(beats_source.parent.resolve()))
        spec = importlib.util.spec_from_file_location("official_beats", beats_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--inference-device cuda requested but CUDA is unavailable")
        state = torch.load(str(checkpoint), map_location=device, weights_only=False)
        cfg = module.BEATsConfig(state["cfg"])
        self._model = module.BEATs(cfg)
        self._model.load_state_dict(state["model"])
        self._model.to(device).eval()
        self._torch = torch
        self.device = device
        self.dimension = int(getattr(cfg, "encoder_embed_dim", 768))

    def extract(self, waveform: np.ndarray) -> np.ndarray:
        source = self._torch.from_numpy(np.asarray(waveform, np.float32)).unsqueeze(0).to(self.device)
        padding = self._torch.zeros_like(source, dtype=self._torch.bool)
        with self._torch.no_grad():
            features, _ = self._model.extract_features(source, padding_mask=padding)
        return features.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)


class LinearHead:
    def __init__(self, weights: np.ndarray, bias: np.ndarray, temperature: np.ndarray | None = None):
        self.weights = np.asarray(weights, np.float32)
        self.bias = np.asarray(bias, np.float32)
        self.temperature = np.ones(5, np.float32) if temperature is None else np.asarray(temperature, np.float32)
        if self.weights.ndim != 2 or self.weights.shape[1] != 5 or self.bias.shape != (5,):
            raise ValueError("head must have weights [embedding_dim, 5] and bias [5]")

    def logits(self, embeddings: np.ndarray) -> np.ndarray:
        return np.asarray(embeddings, np.float32) @ self.weights + self.bias

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return sigmoid(self.logits(embeddings) / np.maximum(self.temperature, 1e-3))

    def save(self, path: str | Path, metadata: dict) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, weights=self.weights, bias=self.bias, temperature=self.temperature,
                            metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        return digest

    @classmethod
    def load(cls, path: str | Path) -> tuple["LinearHead", dict]:
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["metadata"]))
        return cls(data["weights"], data["bias"], data.get("temperature")), metadata


def train_linear_head(x: np.ndarray, y: np.ndarray, epochs: int = 300, learning_rate: float = 0.05,
                      l2: float = 1e-4, seed: int = 17) -> tuple[LinearHead, list[float]]:
    x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
    if x.ndim != 2 or y.shape != (len(x), 5) or not len(x):
        raise ValueError("expected non-empty x [N,D] and y [N,5]")
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 0.01, size=(x.shape[1], 5)).astype(np.float32)
    prevalence = np.maximum(y.sum(axis=0), 1.0)
    pos_weight = np.clip((len(y) - prevalence) / prevalence, 1.0, 20.0)
    history = []
    for _ in range(epochs):
        logits = x @ weights
        prob = sigmoid(logits)
        weight = np.where(y > 0.5, pos_weight, 1.0)
        error = (prob - y) * weight
        grad = x.T @ error / len(x) + l2 * weights
        weights -= learning_rate * grad
        loss = -np.mean(weight * (y * np.log(prob + 1e-7) + (1 - y) * np.log(1 - prob + 1e-7)))
        history.append(float(loss))
    # Bias is learned separately to keep deterministic gradient math simple.
    bias = np.log((prevalence + 0.5) / (len(y) - prevalence + 0.5)).astype(np.float32)
    return LinearHead(weights, bias), history


class TransferScorer(Scorer):
    def __init__(self, extractor: EmbeddingExtractor, head: LinearHead, checkpoint: str, version: str, metadata: dict | None = None):
        self.extractor, self.head = extractor, head
        self.architecture = f"{extractor.name}_transfer"
        self.checkpoint, self.classifier_version = checkpoint, version
        self.device = getattr(extractor, "device", "cpu")
        self.model_size_bytes = int(head.weights.nbytes + head.bias.nbytes + head.temperature.nbytes)
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.is_file(): self.model_size_bytes += checkpoint_path.stat().st_size
        metadata = metadata or {}
        self._ood_center = np.asarray(metadata.get("ood_center", []), np.float32)
        self._ood_scale = np.asarray(metadata.get("ood_scale", []), np.float32)
        self.last_ood_distance: float | None = None

    def scores(self, short_window: np.ndarray, long_window: np.ndarray) -> np.ndarray:
        short_embedding, long_embedding = self.extractor.extract(short_window), self.extractor.extract(long_window)
        short = self.head.predict(short_embedding)
        long = self.head.predict(long_embedding)
        if self._ood_center.shape == long_embedding.shape and self._ood_scale.shape == long_embedding.shape:
            z = (long_embedding - self._ood_center) / np.maximum(self._ood_scale, 1e-4)
            self.last_ood_distance = float(np.sqrt(np.mean(z * z)))
        else:
            self.last_ood_distance = None
        # Transients use the short view; continuing sounds also benefit from the long view.
        return np.asarray([short[0], short[1], max(short[2], long[2]), max(short[3], long[3]), max(short[4], long[4])], np.float32)
