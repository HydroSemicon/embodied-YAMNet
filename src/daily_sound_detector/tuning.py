from __future__ import annotations

from collections import deque
from dataclasses import replace

import numpy as np

from . import LABELS
from .audio import read_wav
from .config import DetectorConfig
from .evaluation import score_recording
from .manifest import ManifestItem
from .models import Scorer


def _detect(frames: list[dict], column: int, start: float, end: float, history_frames: int,
            required_frames: int, min_duration_ms: int, cooldown_ms: int, hop_ms: int) -> list[tuple[int, int]]:
    history, events = deque(maxlen=history_frames), []
    active, started, last_above, cooldown_until = False, 0, 0, 0
    for frame in frames:
        at = int(frame["at_seconds"] * 1000); score = float(frame["scores"][column])
        allowed = not frame["quality"].poor and score >= start
        history.append(allowed)
        if not active and sum(history) >= required_frames and at >= cooldown_until:
            active, started, last_above = True, at, at
        if active:
            if score >= end and not frame["quality"].poor: last_above = at
            if score < end and at - last_above >= hop_ms:
                if at - started >= min_duration_ms: events.append((started, at))
                active, cooldown_until = False, at + cooldown_ms
                history.clear()
    if active and frames:
        ended = int(frames[-1]["at_seconds"] * 1000)
        if ended - started >= min_duration_ms: events.append((started, ended))
    return events


def tune_temporal(items: list[ManifestItem], scorer: Scorer, config: DetectorConfig) -> tuple[DetectorConfig, dict]:
    """Grid-search validation-only temporal parameters, prioritizing negative-audio FP/hour."""
    recordings = []
    for item in items:
        if item.split != "validation": continue
        waveform, _ = read_wav(item.path, config.sample_rate)
        _, frames, _ = score_recording(scorer, waveform, config)
        recordings.append((item, frames, len(waveform) / config.sample_rate))
    if not recordings: raise ValueError("temporal tuning requires validation recordings")
    updated, details = dict(config.classes), {}
    for column, label in enumerate(LABELS):
        base = updated[label]; trials = []
        starts = sorted(set(float(np.clip(base.start_threshold + delta, .05, .99)) for delta in (-.08, 0, .08)))
        required_values = sorted(set(max(1, min(base.history_frames, base.required_frames + delta)) for delta in (-1, 0, 1)))
        cooldown_values = sorted(set(max(0, int(base.cooldown_ms * factor)) for factor in (.5, 1, 1.5)))
        for start in starts:
            for end_ratio in (.5, .65):
                end = min(start - .01, start * end_ratio)
                for required in required_values:
                    for cooldown in cooldown_values:
                        truths, detections, negative_seconds, negative_fp = [], [], 0.0, 0
                        for recording_index, (item, frames, seconds) in enumerate(recordings):
                            found = _detect(frames, column, start, end, base.history_frames, required,
                                            base.min_duration_ms, cooldown, config.hop_ms)
                            expected = [(recording_index, int(e["start_ms"]), int(e["end_ms"])) for e in item.events if e["label"] == label]
                            if not item.events and not item.labels:
                                negative_seconds += seconds; negative_fp += len(found)
                            truths.extend(expected); detections.extend((recording_index, start_ms, end_ms) for start_ms, end_ms in found)
                        matched, latencies, used = 0, [], set()
                        for recording_index, truth_start, truth_end in truths:
                            choices = [(i, d) for i, d in enumerate(detections) if i not in used and d[0] == recording_index
                                       and d[1] <= truth_end + 500 and d[2] >= truth_start - 500]
                            if choices:
                                i, detection = min(choices, key=lambda pair: abs(pair[1][1] - truth_start)); used.add(i)
                                matched += 1; latencies.append(detection[1] - truth_start)
                        fp_h = negative_fp / (negative_seconds / 3600) if negative_seconds else None
                        trials.append({"start_threshold": start, "end_threshold": end, "required_frames": required,
                                       "cooldown_ms": cooldown, "recall": matched / len(truths) if truths else 0.0,
                                       "fp_per_hour": fp_h, "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None})
        feasible = [t for t in trials if t["fp_per_hour"] is None or t["fp_per_hour"] <= .1]
        pool = feasible or trials
        chosen = max(pool, key=lambda t: (t["recall"], -(t["p95_latency_ms"] if t["p95_latency_ms"] is not None else 1e12),
                                           -t["start_threshold"], -t["required_frames"]))
        updated[label] = replace(base, start_threshold=chosen["start_threshold"], end_threshold=chosen["end_threshold"],
                                 required_frames=chosen["required_frames"], cooldown_ms=chosen["cooldown_ms"])
        details[label] = {"selected": chosen, "candidates": len(trials), "met_fp_constraint": chosen in feasible}
    return replace(config, classes=updated), {"source_split": "validation", "per_class": details, "per_class_fp_budget": .1}
