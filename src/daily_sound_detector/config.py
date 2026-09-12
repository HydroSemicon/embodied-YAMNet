from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import LABELS


@dataclass(frozen=True)
class ClassDecisionConfig:
    start_threshold: float
    end_threshold: float
    history_frames: int
    required_frames: int
    min_duration_ms: int
    cooldown_ms: int
    merge_gap_ms: int
    max_duration_ms: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClassDecisionConfig":
        obj = cls(**value)
        if not 0 <= obj.end_threshold < obj.start_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= end < start <= 1")
        if not 1 <= obj.required_frames <= obj.history_frames:
            raise ValueError("required_frames must be between 1 and history_frames")
        if min(obj.min_duration_ms, obj.cooldown_ms, obj.merge_gap_ms) < 0 or obj.max_duration_ms <= 0:
            raise ValueError("durations must be non-negative and max_duration_ms positive")
        return obj


@dataclass(frozen=True)
class DetectorConfig:
    sample_rate: int = 16000
    hop_ms: int = 250
    short_window_ms: int = 1000
    long_window_ms: int = 2000
    ambiguity_margin: float = 0.08
    min_rms: float = 0.0005
    max_abs_dc: float = 0.05
    max_clipped_fraction: float = 0.01
    ood_max_distance: float | None = None
    classes: dict[str, ClassDecisionConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DetectorConfig":
        unknown = set(raw.get("classes", {})) - set(LABELS)
        missing = set(LABELS) - set(raw.get("classes", {}))
        if unknown or missing:
            raise ValueError(f"classes mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
        kwargs = {k: v for k, v in raw.items() if k != "classes"}
        cfg = cls(classes={k: ClassDecisionConfig.from_dict(v) for k, v in raw["classes"].items()}, **kwargs)
        if cfg.sample_rate != 16000 or cfg.hop_ms <= 0:
            raise ValueError("sample_rate must be 16000 and hop_ms must be positive")
        if not 0 <= cfg.ambiguity_margin <= 1:
            raise ValueError("ambiguity_margin must be in [0, 1]")
        return cfg


def load_config(path: str | Path) -> DetectorConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return DetectorConfig.from_dict(json.load(handle))

