"""Combine detector JSONL manifests while preserving resolvable audio paths."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path


VALID_SPLITS = {"train", "validation", "test"}
TARGET_LABELS = {"cough", "sneeze", "laughter", "alarm", "timer_or_ringtone"}


def combine(input_paths: list[Path], output_path: Path) -> dict:
    if len(input_paths) < 2:
        raise ValueError("at least two input manifests are required")
    inputs = [path.resolve() for path in input_paths]
    output = output_path.resolve()
    if output in inputs:
        raise ValueError("output manifest must not overwrite an input manifest")

    rows: list[dict] = []
    identities: dict[tuple[str, str], set[str]] = {}
    for source_manifest in inputs:
        if not source_manifest.is_file():
            raise FileNotFoundError(source_manifest)
        with source_manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                split = row.get("split")
                if split not in VALID_SPLITS:
                    raise ValueError(
                        f"{source_manifest}:{line_number}: invalid split {split!r}"
                    )
                labels = set(row.get("labels", []))
                unknown = labels - TARGET_LABELS
                if unknown:
                    raise ValueError(
                        f"{source_manifest}:{line_number}: unknown labels {sorted(unknown)}"
                    )
                audio_path = Path(row["path"])
                if not audio_path.is_absolute():
                    audio_path = (source_manifest.parent / audio_path).resolve()
                if not audio_path.is_file():
                    raise FileNotFoundError(audio_path)
                row["path"] = Path(os.path.relpath(audio_path, output.parent)).as_posix()
                rows.append(row)

                for field in ("source_id", "session_id", "person_id", "device_id"):
                    value = row.get(field)
                    if value:
                        identities.setdefault((field, str(value)), set()).add(split)

    leaks = [
        {"field": field, "value": value, "splits": sorted(splits)}
        for (field, value), splits in identities.items()
        if len(splits) > 1
    ]
    if leaks:
        raise ValueError(
            "refusing to combine manifests with cross-split identity leakage: "
            + repr(sorted(leaks, key=lambda item: (item["field"], item["value"])))
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False,
        dir=output.parent, prefix=output.name + ".", suffix=".tmp",
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, output)

    split_counts = Counter(row["split"] for row in rows)
    target_counts = {
        split: {
            label: sum(row["split"] == split and label in row["labels"] for row in rows)
            for label in sorted(TARGET_LABELS)
        }
        for split in ("train", "validation", "test")
    }
    return {
        "inputs": [str(path) for path in inputs],
        "output": str(output),
        "items": len(rows),
        "split_counts": dict(split_counts),
        "target_counts": target_counts,
        "identity_leakage": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = combine(args.input, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
