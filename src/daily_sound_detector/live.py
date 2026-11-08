from __future__ import annotations

import queue
import json
import sys
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


def _print_confirmed_event(event: dict, stdout_jsonl: bool = False) -> None:
    """Print one compact, human-readable line for a finalized event."""
    if stdout_jsonl:
        print(json.dumps({key: value for key, value in event.items() if not key.startswith("_")}, ensure_ascii=False), flush=True)
        return
    print(
        f"{event['ended_at']} | EVENT | label={event['label']} | "
        f"confidence={float(event['confidence']):.3f} | "
        f"duration_ms={int(event['duration_ms'])} | "
        f"model={event['model']['architecture']}"
    )


def _print_live_update(events_only: bool, events: list[dict], now: str, scores: np.ndarray,
                       decision: dict, config: DetectorConfig, stdout_jsonl: bool = False) -> None:
    if events_only or stdout_jsonl:
        for event in events:
            _print_confirmed_event(event, stdout_jsonl)
        return
    enabled_indices = np.asarray([LABELS.index(label) for label in config.enabled_labels], np.int64)
    order = enabled_indices[np.argsort(scores[enabled_indices])[::-1][:2]]
    candidates = ", ".join(LABELS[int(i)] for i in order)
    best = int(order[0])
    status = decision["statuses"][LABELS[best]]
    print(
        f"{now} | {candidates} | {scores[best]:.3f} | "
        f"{config.classes[LABELS[best]].start_threshold:.3f} | {status}"
    )


def _flush_live_events(engine: TemporalDecisionEngine, elapsed_ms: int, event_log: str | Path,
                       events_only: bool, stdout_jsonl: bool = False) -> list[dict]:
    events = engine.flush(elapsed_ms)
    append_jsonl(event_log, events)
    if events_only or stdout_jsonl:
        for event in events:
            _print_confirmed_event(event, stdout_jsonl)
    return events


class _StreamingPcm16Writer:
    """Write capture blocks immediately so long recordings do not accumulate in RAM."""

    def __init__(self, path: str | Path, sample_rate: int):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._handle = wave.open(str(target), "wb")
        self._handle.setnchannels(1)
        self._handle.setsampwidth(2)
        self._handle.setframerate(sample_rate)

    def write(self, data: np.ndarray) -> None:
        pcm = (np.clip(np.asarray(data, np.float32), -1, 1) * 32767).astype("<i2")
        self._handle.writeframesraw(pcm.tobytes())

    def close(self) -> None:
        self._handle.close()


def list_input_devices() -> dict:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("sounddevice is required to enumerate input devices") from exc
    devices = sd.query_devices()
    host_apis = sd.query_hostapis()
    try:
        default_input = int(sd.default.device[0])
    except (TypeError, IndexError):
        default_input = int(sd.default.device)
    inputs = []
    for index, value in enumerate(devices):
        info = dict(value)
        if int(info.get("max_input_channels", 0)) <= 0:
            continue
        host_index = int(info.get("hostapi", -1))
        host_name = host_apis[host_index]["name"] if 0 <= host_index < len(host_apis) else None
        inputs.append({
            "index": index,
            "name": info.get("name"),
            "host_api": host_name,
            "max_input_channels": int(info["max_input_channels"]),
            "default_sample_rate": float(info["default_samplerate"]),
            "is_default": index == default_input,
        })
    return {"default_input_index": default_input, "input_devices": inputs}


def run_live(scorer: Scorer, config: DetectorConfig, event_log: str | Path, raw_log: str | Path | None = None,
             device: int | str | None = None, duration_seconds: float | None = None,
             record_wav: str | Path | None = None, events_only: bool = False,
             stdout_jsonl: bool = False) -> None:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("sounddevice is required for live mode") from exc
    if isinstance(device, str) and device.strip().isdigit():
        device = int(device.strip())
    info = sd.query_devices(device, "input")
    device_rate = int(info["default_samplerate"])
    block = max(1, int(device_rate * config.hop_ms / 1000))
    received: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
    recorder: _StreamingPcm16Writer | None = None

    def callback(indata, frames, timing, status):
        if status:
            print(f"audio_status={status}", file=sys.stderr, flush=True)
        try:
            received.put_nowait(np.asarray(indata, np.float32).copy())
        except queue.Full:
            print("audio_status=input_queue_overflow", file=sys.stderr, flush=True)

    ring = RingBuffer(config.long_window_ms * 16)
    engine = TemporalDecisionEngine(config, {"architecture": scorer.architecture, "checkpoint": scorer.checkpoint,
                                             "classifier_version": scorer.classifier_version})
    started, elapsed_ms = time.monotonic(), 0
    if not events_only and not stdout_jsonl:
        print("time | top candidates | score | threshold | status")
    try:
        with sd.InputStream(device=device, channels=1, samplerate=device_rate, blocksize=block, dtype="float32", callback=callback):
            if record_wav is not None:
                recorder = _StreamingPcm16Writer(record_wav, config.sample_rate)
            while duration_seconds is None or time.monotonic() - started < duration_seconds:
                block_data = received.get(timeout=2.0)
                mono = resample(block_data, device_rate, config.sample_rate)
                if recorder is not None:
                    recorder.write(mono)
                ring.append(mono); elapsed_ms = int((time.monotonic() - started) * 1000)
                long_window = ring.latest(config.long_window_ms * 16)
                short_window = ring.latest(config.short_window_ms * 16)
                quality = inspect_quality(long_window, config.min_rms, config.max_abs_dc, config.max_clipped_fraction)
                scores = scorer.scores(short_window, long_window)
                events, decision = engine.update(scores, elapsed_ms, quality, getattr(scorer, "last_ood_distance", None))
                append_jsonl(event_log, events)
                now = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
                _print_live_update(events_only, events, now, scores, decision, config, stdout_jsonl)
                if raw_log is not None:
                    append_jsonl(raw_log, [{"schema_version": "1.0", "at": now, "scores": dict(zip(LABELS, map(float, scores))),
                                            "decision": decision, "quality": quality.__dict__}])
    except KeyboardInterrupt:
        pass
    finally:
        try:
            _flush_live_events(engine, elapsed_ms, event_log, events_only, stdout_jsonl)
        finally:
            if recorder is not None:
                recorder.close()
