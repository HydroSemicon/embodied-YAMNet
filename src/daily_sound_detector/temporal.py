from __future__ import annotations

import json
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import LABELS
from .audio import AudioQuality
from .config import DetectorConfig


@dataclass
class _State:
    history: deque = field(default_factory=deque)
    active: bool = False
    started_ms: int = 0
    last_above_end_ms: int = 0
    peak: float = 0.0
    frames_above: int = 0
    cooldown_until_ms: int = 0
    pending: dict | None = None


def _iso_from_epoch_ms(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class TemporalDecisionEngine:
    def __init__(self, config: DetectorConfig, model_info: dict, start_epoch_ms: int | None = None):
        self.config, self.model_info = config, model_info
        self.start_epoch_ms = start_epoch_ms if start_epoch_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
        self.states = {label: _State(deque(maxlen=config.classes[label].history_frames)) for label in LABELS}

    def _event(self, label: str, state: _State, ended_ms: int) -> dict:
        cfg = self.config.classes[label]
        start_abs, end_abs = self.start_epoch_ms + state.started_ms, self.start_epoch_ms + ended_ms
        return {
            "schema_version": "1.0", "event_id": f"evt_{uuid.uuid4()}", "label": label,
            "confidence": round(state.peak, 6), "started_at": _iso_from_epoch_ms(start_abs),
            "ended_at": _iso_from_epoch_ms(end_abs), "duration_ms": max(0, ended_ms - state.started_ms),
            "source": "microphone", "model": self.model_info,
            "decision": {"start_threshold": cfg.start_threshold, "peak_score": round(state.peak, 6),
                         "frames_above_threshold": state.frames_above},
        }

    def update(self, scores: np.ndarray, at_ms: int, quality: AudioQuality, ood_distance: float | None = None) -> tuple[list[dict], dict]:
        values = np.asarray(scores, np.float32)
        if values.shape != (5,):
            raise ValueError("scores must have shape [5]")
        order = np.argsort(values)[::-1]
        ambiguous = values[order[0]] - values[order[1]] < self.config.ambiguity_margin
        ood = self.config.ood_max_distance is not None and ood_distance is not None and ood_distance > self.config.ood_max_distance
        globally_suppressed = quality.poor or ambiguous or ood
        emitted: list[dict] = []
        statuses = {}
        for index, label in enumerate(LABELS):
            state, cfg, score = self.states[label], self.config.classes[label], float(values[index])
            above_start = score >= cfg.start_threshold and not globally_suppressed
            state.history.append(above_start)
            enough = sum(state.history) >= cfg.required_frames
            if state.pending and at_ms - state.pending["_ended_ms"] > cfg.merge_gap_ms:
                pending = dict(state.pending)
                pending.pop("_ended_ms", None)
                pending.pop("_started_ms", None)
                emitted.append(pending)
                state.pending = None
            can_merge = state.pending is not None and at_ms - state.pending["_ended_ms"] <= cfg.merge_gap_ms
            if not state.active and enough and (at_ms >= state.cooldown_until_ms or can_merge):
                if can_merge:
                    state.active, state.started_ms = True, state.pending["_started_ms"]
                    state.peak, state.frames_above = state.pending["confidence"], state.pending["decision"]["frames_above_threshold"]
                    state.pending = None
                else:
                    state.active, state.started_ms, state.peak, state.frames_above = True, at_ms, score, 0
            if state.active:
                state.peak = max(state.peak, score)
                if score >= cfg.start_threshold:
                    state.frames_above += 1
                if score >= cfg.end_threshold and not quality.poor:
                    state.last_above_end_ms = at_ms
                duration = at_ms - state.started_ms
                ended = (score < cfg.end_threshold and at_ms - state.last_above_end_ms >= self.config.hop_ms) or duration >= cfg.max_duration_ms
                if ended:
                    if duration >= cfg.min_duration_ms:
                        event = self._event(label, state, at_ms)
                        event["_ended_ms"], event["_started_ms"] = at_ms, state.started_ms
                        state.pending = event
                    state.active = False
                    state.cooldown_until_ms = at_ms + cfg.cooldown_ms
                    state.history.clear()
            statuses[label] = "detected" if state.active else ("pending" if state.pending else ("suppressed" if above_start and globally_suppressed else "idle"))
        return emitted, {"ambiguous": bool(ambiguous), "quality_reasons": list(quality.reasons), "ood": bool(ood), "statuses": statuses}

    def flush(self, at_ms: int) -> list[dict]:
        output = []
        for label, state in self.states.items():
            if state.active:
                event = self._event(label, state, at_ms)
                if event["duration_ms"] >= self.config.classes[label].min_duration_ms:
                    output.append(event)
                state.active = False
            if state.pending:
                pending = dict(state.pending)
                pending.pop("_ended_ms", None)
                pending.pop("_started_ms", None)
                output.append(pending)
                state.pending = None
        return output


def append_jsonl(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            handle.write(json.dumps(clean, ensure_ascii=False) + "\n")
