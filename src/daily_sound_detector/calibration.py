from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import LABELS
from .config import DetectorConfig
from .models import LinearHead, sigmoid


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    logits, targets = np.asarray(logits), np.asarray(targets)
    temperatures = np.ones(5, np.float32)
    candidates = np.geomspace(0.25, 4.0, 81)
    for column in range(5):
        losses = []
        for value in candidates:
            p = sigmoid(logits[:, column] / value)
            loss = -np.mean(targets[:, column] * np.log(p + 1e-8) + (1 - targets[:, column]) * np.log(1 - p + 1e-8))
            losses.append(loss)
        temperatures[column] = float(candidates[int(np.argmin(losses))])
    return temperatures


def precision_recall_points(scores: np.ndarray, targets: np.ndarray) -> list[dict]:
    points = []
    for threshold in np.linspace(0, 1, 201):
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (targets == 1)))
        fp = int(np.sum(predicted & (targets == 0)))
        fn = int(np.sum(~predicted & (targets == 1)))
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        points.append({"threshold": float(threshold), "precision": precision, "recall": recall})
    return points


def choose_threshold(scores: np.ndarray, targets: np.ndarray, min_precision: float = 0.95) -> dict:
    points = precision_recall_points(scores, targets)
    eligible = [p for p in points if p["precision"] >= min_precision]
    chosen = max(eligible, key=lambda p: (p["recall"], -p["threshold"])) if eligible else max(points, key=lambda p: (p["precision"], p["recall"]))
    return {**chosen, "met_precision_constraint": chosen["precision"] >= min_precision}


def calibrate_head(head: LinearHead, embeddings: np.ndarray, targets: np.ndarray, config: DetectorConfig,
                   output_dir: str | Path) -> tuple[LinearHead, DetectorConfig, dict]:
    logits = head.logits(embeddings)
    temperature = fit_temperature(logits, targets)
    calibrated = LinearHead(head.weights, head.bias, temperature)
    scores = calibrated.predict(embeddings)
    selections, classes = {}, {}
    for i, label in enumerate(LABELS):
        selection = choose_threshold(scores[:, i], targets[:, i])
        selections[label] = selection
        old = config.classes[label]
        start = float(np.clip(selection["threshold"], 0.05, 0.99))
        end = min(old.end_threshold, max(0.01, start * 0.65))
        classes[label] = replace(old, start_threshold=start, end_threshold=end)
    updated = replace(config, classes=classes)
    result = {"temperature": temperature.tolist(), "thresholds": selections, "source_split": "validation"}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return calibrated, updated, result

