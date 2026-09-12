from __future__ import annotations

import csv
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from . import LABELS
from .audio import inspect_quality, iter_windows, read_wav
from .calibration import precision_recall_points
from .config import DetectorConfig
from .manifest import ManifestItem
from .models import Scorer
from .temporal import TemporalDecisionEngine


def _binary_metrics(target: np.ndarray, predicted: np.ndarray) -> dict:
    tp = int(np.sum((target == 1) & (predicted == 1)))
    fp = int(np.sum((target == 0) & (predicted == 1)))
    fn = int(np.sum((target == 1) & (predicted == 0)))
    tn = int(np.sum((target == 0) & (predicted == 0)))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def score_recording(scorer: Scorer, waveform: np.ndarray, config: DetectorConfig) -> tuple[np.ndarray, list[dict], float]:
    short_n, long_n, hop_n = config.short_window_ms * 16, config.long_window_ms * 16, config.hop_ms * 16
    long_frames = list(iter_windows(waveform, long_n, hop_n))
    rows, elapsed = [], 0.0
    for at_seconds, long_window in long_frames:
        short_window = long_window[-short_n:]
        start = time.perf_counter()
        scores = scorer.scores(short_window, long_window)
        elapsed += time.perf_counter() - start
        quality = inspect_quality(long_window, config.min_rms, config.max_abs_dc, config.max_clipped_fraction)
        rows.append({"at_seconds": at_seconds, "scores": np.asarray(scores).tolist(), "quality": quality,
                     "ood_distance": getattr(scorer, "last_ood_distance", None)})
    max_scores = np.max([row["scores"] for row in rows], axis=0) if rows else np.zeros(5)
    audio_seconds = max(len(waveform) / config.sample_rate, 1e-9)
    return max_scores.astype(np.float32), rows, elapsed / audio_seconds


def evaluate_clips(items: list[ManifestItem], scorer: Scorer, config: DetectorConfig, output_dir: str | Path,
                   split: str = "test", raw_scores: bool = False) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tracemalloc.start()
    targets, scores, records, rtfs = [], [], [], []
    for item in items:
        if item.split != split:
            continue
        waveform, _ = read_wav(item.path, config.sample_rate)
        clip_scores, frames, rtf = score_recording(scorer, waveform, config)
        target = np.asarray([label in item.labels for label in LABELS], np.int8)
        targets.append(target); scores.append(clip_scores); rtfs.append(rtf)
        records.append({"path": str(item.path), "target": target.tolist(), "scores": clip_scores.tolist()})
        if raw_scores:
            with (out / "raw_scores.jsonl").open("a", encoding="utf-8") as handle:
                for frame in frames:
                    handle.write(json.dumps({"path": str(item.path), "at_seconds": frame["at_seconds"], "scores": frame["scores"]}) + "\n")
    if not targets:
        raise ValueError(f"no recordings for split {split!r}")
    y, p = np.stack(targets), np.stack(scores)
    thresholds = np.asarray([config.classes[label].start_threshold for label in LABELS])
    predicted = p >= thresholds
    per_class = {label: _binary_metrics(y[:, i], predicted[:, i]) for i, label in enumerate(LABELS)}
    false_negatives, false_positives = [], []
    for row, truth, pred in zip(records, y, predicted):
        for i, label in enumerate(LABELS):
            if truth[i] and not pred[i]: false_negatives.append({"path": row["path"], "label": label, "score": row["scores"][i]})
            if not truth[i] and pred[i]: false_positives.append({"path": row["path"], "label": label, "score": row["scores"][i]})
    _, peak_memory = tracemalloc.get_traced_memory(); tracemalloc.stop()
    summary = {
        "model": {"architecture": scorer.architecture, "checkpoint": scorer.checkpoint, "classifier_version": scorer.classifier_version},
        "split": split, "clips": len(y), "per_class": per_class,
        "macro_f1": float(np.mean([x["f1"] for x in per_class.values()])),
        "macro_recall": float(np.mean([x["recall"] for x in per_class.values()])),
        "real_time_factor": {getattr(scorer, "device", "cpu"): float(np.mean(rtfs))},
        "peak_python_memory_mb": peak_memory / (1024 * 1024),
        "model_size_bytes": getattr(scorer, "model_size_bytes", None), "false_negatives": false_negatives,
        "false_positives": false_positives,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "false_negatives.json").write_text(json.dumps(false_negatives, indent=2), encoding="utf-8")
    (out / "false_positives.json").write_text(json.dumps(false_positives, indent=2), encoding="utf-8")
    for i, label in enumerate(LABELS):
        with (out / f"pr_curve_{label}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["threshold", "precision", "recall"]); writer.writeheader()
            writer.writerows(precision_recall_points(p[:, i], y[:, i]))
    matrix = np.zeros((6, 6), np.int64)
    for truth, pred_scores in zip(y, p):
        true_idx = int(np.argmax(truth)) if truth.any() else 5
        pred_idx = int(np.argmax(pred_scores)) if pred_scores.max() >= thresholds[int(np.argmax(pred_scores))] else 5
        matrix[true_idx, pred_idx] += 1
    np.savetxt(out / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    return summary


def _relative_ms(iso: str) -> int:
    from datetime import datetime
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)


def evaluate_continuous(items: list[ManifestItem], scorer: Scorer, config: DetectorConfig, output_dir: str | Path,
                        split: str = "test") -> dict:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    detections, truths, negative_seconds, rtfs, negative_paths = [], [], 0.0, [], set()
    for item in items:
        if item.split != split: continue
        waveform, _ = read_wav(item.path, config.sample_rate)
        _, frames, rtf = score_recording(scorer, waveform, config); rtfs.append(rtf)
        engine = TemporalDecisionEngine(config, {"architecture": scorer.architecture, "checkpoint": scorer.checkpoint,
                                                  "classifier_version": scorer.classifier_version}, start_epoch_ms=0)
        local = []
        for frame in frames:
            events, _ = engine.update(np.asarray(frame["scores"]), int(frame["at_seconds"] * 1000), frame["quality"], frame["ood_distance"])
            local.extend(events)
        local.extend(engine.flush(int(len(waveform) / 16)))
        for event in local:
            detections.append({"path": str(item.path), "label": event["label"], "start_ms": _relative_ms(event["started_at"]),
                               "end_ms": _relative_ms(event["ended_at"]), "confidence": event["confidence"]})
        for event in item.events:
            truths.append({"path": str(item.path), "label": event["label"], "start_ms": int(event["start_ms"]), "end_ms": int(event["end_ms"])})
        if not item.events and not item.labels:
            negative_seconds += len(waveform) / config.sample_rate
            negative_paths.add(str(item.path))
    matched_truth, matched_detection, latencies = set(), set(), []
    for di, detection in enumerate(detections):
        candidates = [(ti, truth) for ti, truth in enumerate(truths) if ti not in matched_truth and truth["path"] == detection["path"]
                      and truth["label"] == detection["label"] and detection["start_ms"] <= truth["end_ms"] + 500
                      and detection["end_ms"] >= truth["start_ms"] - 500]
        if candidates:
            ti, truth = min(candidates, key=lambda pair: abs(pair[1]["start_ms"] - detection["start_ms"]))
            matched_truth.add(ti); matched_detection.add(di); latencies.append(detection["start_ms"] - truth["start_ms"])
    false_detections = [d for i, d in enumerate(detections) if i not in matched_detection]
    missed = [t for i, t in enumerate(truths) if i not in matched_truth]
    negative_false_detections = [d for d in false_detections if d["path"] in negative_paths]
    fp_per_hour = len(negative_false_detections) / (negative_seconds / 3600) if negative_seconds else None
    result = {"events": len(truths), "detections": len(detections), "matched": len(matched_truth),
              "false_positives": false_detections, "false_negatives": missed,
              "negative_audio_false_positives": negative_false_detections,
              "negative_audio_hours": negative_seconds / 3600, "false_positives_per_hour": fp_per_hour,
              "latency_ms": {"mean": float(np.mean(latencies)) if latencies else None,
                             "p95": float(np.percentile(latencies, 95)) if latencies else None},
              "mean_real_time_factor": float(np.mean(rtfs)) if rtfs else None}
    (out / "continuous_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
