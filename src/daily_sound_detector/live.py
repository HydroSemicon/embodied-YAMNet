from __future__ import annotations

import json
import queue
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import LABELS
from .audio import RingBuffer, inspect_quality, resample
from .config import DetectorConfig
from .models import Scorer
from .temporal import TemporalDecisionEngine, append_jsonl


def _print_confirmed_event(event: dict) -> None:
    """Print one compact, human-readable line for a finalized event."""
    print(
        f"{event['ended_at']} | EVENT | label={event['label']} | "
        f"confidence={float(event['confidence']):.3f} | "
        f"duration_ms={int(event['duration_ms'])} | "
        f"model={event['model']['architecture']}"
    )


def _print_live_update(events_only: bool, events: list[dict], now: str, scores: np.ndarray,
                       decision: dict, config: DetectorConfig) -> None:
    if events_only:
        for event in events:
            _print_confirmed_event(event)
        return
    order = np.argsort(scores)[::-1][:2]
    candidates = ", ".join(LABELS[i] for i in order)
    best = int(order[0])
    status = decision["statuses"][LABELS[best]]
    print(
        f"{now} | {candidates} | {scores[best]:.3f} | "
        f"{config.classes[LABELS[best]].start_threshold:.3f} | {status}"
    )


def _flush_live_events(engine: TemporalDecisionEngine, elapsed_ms: int, event_log: str | Path,
                       events_only: bool) -> list[dict]:
    events = engine.flush(elapsed_ms)
    append_jsonl(event_log, events)
    if events_only:
        for event in events:
            _print_confirmed_event(event)
    return events


def _write_pcm16(path: str | Path, chunks: list[np.ndarray], sample_rate: int) -> None:
    data = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(sample_rate); handle.writeframes(pcm.tobytes())


def run_live(scorer: Scorer, config: DetectorConfig, event_log: str | Path, raw_log: str | Path | None = None,
             device: int | str | None = None, duration_seconds: float | None = None,
             record_wav: str | Path | None = None, events_only: bool = False) -> None:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("sounddevice is required for live mode") from exc
    info = sd.query_devices(device, "input")
    device_rate = int(info["default_samplerate"])
    block = max(1, int(device_rate * config.hop_ms / 1000))
    received: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
    recorded: list[np.ndarray] = []

    def callback(indata, frames, timing, status):
        if status:
            print(f"audio_status={status}")
        try:
            received.put_nowait(np.asarray(indata, np.float32).copy())
        except queue.Full:
            print("audio_status=input_queue_overflow")

    ring = RingBuffer(config.long_window_ms * 16)
    engine = TemporalDecisionEngine(config, {"architecture": scorer.architecture, "checkpoint": scorer.checkpoint,
                                             "classifier_version": scorer.classifier_version})
    started, elapsed_ms = time.monotonic(), 0
    if not events_only:
        print("time | top candidates | score | threshold | status")
    try:
        with sd.InputStream(device=device, channels=1, samplerate=device_rate, blocksize=block, dtype="float32", callback=callback):
            while duration_seconds is None or time.monotonic() - started < duration_seconds:
                block_data = received.get(timeout=2.0)
                mono = resample(block_data, device_rate, config.sample_rate)
                if record_wav is not None: recorded.append(mono)
                ring.append(mono); elapsed_ms = int((time.monotonic() - started) * 1000)
                long_window = ring.latest(config.long_window_ms * 16)
                short_window = ring.latest(config.short_window_ms * 16)
                quality = inspect_quality(long_window, config.min_rms, config.max_abs_dc, config.max_clipped_fraction)
                scores = scorer.scores(short_window, long_window)
                events, decision = engine.update(scores, elapsed_ms, quality, getattr(scorer, "last_ood_distance", None))
                append_jsonl(event_log, events)
                now = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
                _print_live_update(events_only, events, now, scores, decision, config)
                if raw_log is not None:
                    append_jsonl(raw_log, [{"schema_version": "1.0", "at": now, "scores": dict(zip(LABELS, map(float, scores))),
                                            "decision": decision, "quality": quality.__dict__}])
    except KeyboardInterrupt:
        pass
    finally:
        _flush_live_events(engine, elapsed_ms, event_log, events_only)
        if record_wav is not None:
            _write_pcm16(record_wav, recorded, config.sample_rate)
