from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import LABELS

VALID_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class ManifestItem:
    path: Path
    split: str
    labels: tuple[str, ...]
    source_id: str
    session_id: str
    person_id: str | None = None
    device_id: str | None = None
    recorded_at: str | None = None
    is_hard_negative: bool = False
    playback: bool = False
    events: tuple[dict, ...] = ()


def read_manifest(path: str | Path) -> list[ManifestItem]:
    manifest_path = Path(path).resolve()
    items = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            missing = {"path", "split", "labels", "source_id", "session_id"} - raw.keys()
            if missing:
                raise ValueError(f"line {line_number}: missing {sorted(missing)}")
            if raw["split"] not in VALID_SPLITS:
                raise ValueError(f"line {line_number}: invalid split {raw['split']!r}")
            unknown = set(raw["labels"]) - set(LABELS)
            if unknown:
                raise ValueError(f"line {line_number}: unknown labels {sorted(unknown)}")
            wav_path = Path(raw["path"])
            if not wav_path.is_absolute():
                wav_path = (manifest_path.parent / wav_path).resolve()
            items.append(ManifestItem(
                path=wav_path, split=raw["split"], labels=tuple(raw["labels"]),
                source_id=str(raw["source_id"]), session_id=str(raw["session_id"]),
                person_id=raw.get("person_id"), device_id=raw.get("device_id"),
                recorded_at=raw.get("recorded_at"), is_hard_negative=bool(raw.get("is_hard_negative", False)),
                playback=bool(raw.get("playback", False)), events=tuple(raw.get("events", ())),
            ))
    if not items:
        raise ValueError("manifest is empty")
    return items


def leakage_report(items: Iterable[ManifestItem]) -> dict:
    groups: dict[tuple[str, str], set[str]] = {}
    for item in items:
        identities = {"source_id": item.source_id, "session_id": item.session_id}
        if item.person_id:
            identities["person_id"] = item.person_id
        if item.device_id:
            identities["device_id"] = item.device_id
        for kind, value in identities.items():
            groups.setdefault((kind, value), set()).add(item.split)
    leaks = [
        {"field": kind, "value": value, "splits": sorted(splits)}
        for (kind, value), splits in groups.items() if len(splits) > 1
    ]
    return {"ok": not leaks, "leaks": sorted(leaks, key=lambda x: (x["field"], x["value"]))}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_prepared_index(items: list[ManifestItem], output: str | Path) -> dict:
    report = leakage_report(items)
    if not report["ok"]:
        raise ValueError(f"recording-source leakage detected: {report['leaks']}")
    rows = []
    for item in items:
        if not item.path.is_file():
            raise FileNotFoundError(item.path)
        rows.append({
            "path": str(item.path), "split": item.split, "labels": list(item.labels),
            "source_id": item.source_id, "session_id": item.session_id,
            "sha256": sha256_file(item.path), "is_hard_negative": item.is_hard_negative,
            "playback": item.playback, "events": list(item.events),
        })
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {split: sum(x.split == split for x in items) for split in sorted(VALID_SPLITS)}
    return {"items": len(items), "split_counts": counts, "leakage": report, "output": str(target)}

