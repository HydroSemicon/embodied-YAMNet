"""Prepare a focused, source-disjoint FSD50K development subset.

The official split ZIP is first converted to a normal ZIP with Info-ZIP. Only
the five detector targets, confusable hard negatives, and a balanced sample of
general negatives are extracted. Train/validation assignment is deterministic
and grouped by Freesound uploader.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


GROUND_TRUTH_MD5 = "ca27382c195e37d2269c4c866dd73485"
METADATA_MD5 = "b9ea0c829a411c1d42adb9da539ed237"
DOC_MD5 = "3516162b82dc2945d3e7feba0904e800"
DEV_PART_MD5 = {
    "FSD50K.dev_audio.z01": "faa7cf4cc076fc34a44a479a5ed862a3",
    "FSD50K.dev_audio.z02": "8f9b66153e68571164fb1315d00bc7bc",
    "FSD50K.dev_audio.z03": "1196ef47d267a993d30fa98af54b7159",
    "FSD50K.dev_audio.z04": "d088ac4e11ba53daf9f7574c11cccac9",
    "FSD50K.dev_audio.z05": "81356521aa159accd3c35de22da28c7f",
    "FSD50K.dev_audio.zip": "c480d119b8f7a7e32fdb58f3ea4d6c5a",
}

TARGET_MAP = {
    "Cough": "cough",
    "Sneeze": "sneeze",
    "Laughter": "laughter",
    "Siren": "alarm",
    "Ringtone": "timer_or_ringtone",
}

HARD_NEGATIVE_LABELS = {
    "Bell",
    "Chime",
    "Clock",
    "Doorbell",
    "Microwave_oven",
    "Telephone",
}

AMBIGUOUS_LABELS = {"Alarm"}
SPLIT_SALT = "fsd50k-uploader-split-v1:"
GENERAL_SAMPLE_SALT = "fsd50k-general-negative-v1:"


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # nosec: official archive integrity checksum
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_md5(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = md5_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"MD5 mismatch for {path}: {actual}; expected {expected}")


def ensure_unsplit_archive(
    parts_directory: Path,
    output_archive: Path,
    *,
    verify_parts: bool = True,
    zip_command: str | None = None,
) -> bool:
    """Return True when a new normal ZIP was created."""
    parts_directory = parts_directory.resolve()
    output_archive = output_archive.resolve()
    tail = parts_directory / "FSD50K.dev_audio.zip"

    if output_archive.is_file():
        with zipfile.ZipFile(output_archive) as archive:
            if not any(name.lower().endswith(".wav") for name in archive.namelist()):
                raise zipfile.BadZipFile(f"no WAV members in {output_archive}")
        return False

    if verify_parts:
        for filename, expected in DEV_PART_MD5.items():
            _verify_md5(parts_directory / filename, expected)

    executable = zip_command or shutil.which("zip")
    if not executable:
        raise FileNotFoundError(
            "Info-ZIP 'zip' command was not found; install it or pass --zip-command"
        )
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [executable, "-q", "-s", "0", str(tail), "--out", str(output_archive)],
        check=True,
    )
    with zipfile.ZipFile(output_archive) as archive:
        if not any(name.lower().endswith(".wav") for name in archive.namelist()):
            raise zipfile.BadZipFile(f"no WAV members in {output_archive}")
    return True


def _read_csv_member(archive_path: Path, member: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(archive_path) as archive:
        text = archive.read(member).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def _read_json_member(archive_path: Path, member: str) -> dict:
    with zipfile.ZipFile(archive_path) as archive:
        return json.loads(archive.read(member))


def _stable_digest(salt: str, value: str) -> bytes:
    return hashlib.sha256((salt + value).encode("utf-8")).digest()


def _validation_split(uploader: str, validation_percent: int) -> str:
    bucket = int.from_bytes(_stable_digest(SPLIT_SALT, uploader)[:8], "big") % 100
    return "validation" if bucket < validation_percent else "train"


def select_rows(
    rows: list[dict[str, str]],
    metadata: dict,
    *,
    general_per_primary_label: int = 20,
    validation_percent: int = 25,
) -> list[dict]:
    if not 1 <= validation_percent <= 50:
        raise ValueError("validation_percent must be between 1 and 50")
    if general_per_primary_label < 0:
        raise ValueError("general_per_primary_label must be non-negative")

    positives: list[dict] = []
    hard_negatives: list[dict] = []
    general_groups: dict[str, list[dict]] = defaultdict(list)

    for source in rows:
        clip_id = str(source["fname"])
        if clip_id not in metadata:
            raise KeyError(f"metadata is missing for clip {clip_id}")
        source_labels = source["labels"].split(",")
        source_label_set = set(source_labels)
        detector_labels = sorted({
            detector_label
            for fsd_label, detector_label in TARGET_MAP.items()
            if fsd_label in source_label_set
        })
        base = {
            "clip_id": clip_id,
            "source_labels": source_labels,
            "detector_labels": detector_labels,
            "uploader": str(metadata[clip_id]["uploader"]),
            "license": str(metadata[clip_id]["license"]),
            "title": str(metadata[clip_id].get("title", "")),
        }
        if detector_labels:
            base["selection_role"] = "positive"
            positives.append(base)
        elif source_label_set & HARD_NEGATIVE_LABELS:
            base["selection_role"] = "hard_negative"
            hard_negatives.append(base)
        elif source_label_set & AMBIGUOUS_LABELS:
            # Generic Alarm combines safety alarms, telephones, clocks, and
            # doorbells in the source ontology, so it is unsafe for either class.
            continue
        elif general_per_primary_label:
            base["selection_role"] = "general_negative"
            general_groups[source_labels[0]].append(base)

    selected = positives + hard_negatives
    for primary_label in sorted(general_groups):
        candidates = sorted(
            general_groups[primary_label],
            key=lambda row: _stable_digest(GENERAL_SAMPLE_SALT, row["clip_id"]),
        )
        selected.extend(candidates[:general_per_primary_label])

    for row in selected:
        row["split"] = _validation_split(row["uploader"], validation_percent)
    return sorted(selected, key=lambda row: (row["split"], int(row["clip_id"])))


def _safe_destination(root: Path, member: str) -> Path:
    pure = PurePosixPath(member)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe ZIP member: {member!r}")
    destination = root.joinpath(*pure.parts)
    try:
        destination.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"ZIP member escapes output root: {member!r}") from exc
    return destination


def _extract_member(archive: zipfile.ZipFile, member: str, destination: Path) -> bool:
    info = archive.getinfo(member)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == info.file_size:
        return False
    if destination.exists():
        raise FileExistsError(
            f"existing file has a different size: {destination}; remove it and run again"
        )
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()
    try:
        with archive.open(info) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()
    return True


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False,
        dir=path.parent, prefix=path.name + ".", suffix=".tmp",
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def prepare(
    ground_truth_archive: Path,
    metadata_archive: Path,
    audio_archive: Path,
    output_root: Path,
    manifest_path: Path,
    attribution_path: Path,
    *,
    general_per_primary_label: int = 20,
    validation_percent: int = 25,
    verify_metadata_archives: bool = True,
) -> dict:
    ground_truth_archive = ground_truth_archive.resolve()
    metadata_archive = metadata_archive.resolve()
    audio_archive = audio_archive.resolve()
    output_root = output_root.resolve()
    manifest_path = manifest_path.resolve()
    attribution_path = attribution_path.resolve()

    if verify_metadata_archives:
        _verify_md5(ground_truth_archive, GROUND_TRUTH_MD5)
        _verify_md5(metadata_archive, METADATA_MD5)

    source_rows = _read_csv_member(
        ground_truth_archive, "FSD50K.ground_truth/dev.csv"
    )
    metadata = _read_json_member(
        metadata_archive, "FSD50K.metadata/dev_clips_info_FSD50K.json"
    )
    selected = select_rows(
        source_rows,
        metadata,
        general_per_primary_label=general_per_primary_label,
        validation_percent=validation_percent,
    )

    manifest_rows: list[dict] = []
    attribution_rows: list[dict] = []
    extracted = 0
    skipped = 0
    with zipfile.ZipFile(audio_archive) as archive:
        wav_by_basename = {
            PurePosixPath(name).name: name
            for name in archive.namelist()
            if name.lower().endswith(".wav")
        }
        for selected_row in selected:
            filename = selected_row["clip_id"] + ".wav"
            try:
                member = wav_by_basename[filename]
            except KeyError as exc:
                raise FileNotFoundError(f"audio archive is missing {filename}") from exc
            audio_path = _safe_destination(output_root, f"audio/{filename}")
            if _extract_member(archive, member, audio_path):
                extracted += 1
            else:
                skipped += 1

            relative_audio = Path(
                os.path.relpath(audio_path, manifest_path.parent)
            ).as_posix()
            is_hard_negative = selected_row["selection_role"] == "hard_negative"
            manifest_rows.append({
                "path": relative_audio,
                "split": selected_row["split"],
                "labels": selected_row["detector_labels"],
                "source_id": "fsd50k:uploader:" + selected_row["uploader"],
                "session_id": "fsd50k:clip:" + selected_row["clip_id"],
                "person_id": None,
                "device_id": None,
                "is_hard_negative": is_hard_negative,
                "playback": False,
                "events": [],
            })
            attribution_rows.append({
                "clip_id": selected_row["clip_id"],
                "uploader": selected_row["uploader"],
                "license": selected_row["license"],
                "title": selected_row["title"],
                "source_labels": selected_row["source_labels"],
                "detector_labels": selected_row["detector_labels"],
                "selection_role": selected_row["selection_role"],
                "split": selected_row["split"],
            })

    _atomic_jsonl(manifest_path, manifest_rows)
    _atomic_jsonl(attribution_path, attribution_rows)

    split_counts = Counter(row["split"] for row in selected)
    role_counts = Counter(row["selection_role"] for row in selected)
    target_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation"):
        target_counts[split] = {
            label: sum(
                row["split"] == split and label in row["detector_labels"]
                for row in selected
            )
            for label in TARGET_MAP.values()
        }
    uploader_sets = {
        split: {row["uploader"] for row in selected if row["split"] == split}
        for split in ("train", "validation")
    }
    overlap = uploader_sets["train"] & uploader_sets["validation"]
    if overlap:
        raise AssertionError(f"uploader leakage: {sorted(overlap)[:5]}")

    return {
        "clips": len(selected),
        "split_counts": dict(split_counts),
        "role_counts": dict(role_counts),
        "target_counts": target_counts,
        "uploader_counts": {key: len(value) for key, value in uploader_sets.items()},
        "uploader_leakage": False,
        "extracted_files": extracted,
        "already_present_files": skipped,
        "manifest": str(manifest_path),
        "attribution": str(attribution_path),
        "output_root": str(output_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=Path("downloads/fsd50k"))
    parser.add_argument(
        "--unsplit-archive",
        type=Path,
        default=Path("downloads/fsd50k/FSD50K.dev_audio.unsplit.zip"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/public/fsd50k"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/fsd50k-dev.jsonl"))
    parser.add_argument(
        "--attribution",
        type=Path,
        default=Path("data/manifests/fsd50k-dev-attribution.jsonl"),
    )
    parser.add_argument("--general-negatives-per-class", type=int, default=20)
    parser.add_argument("--validation-percent", type=int, default=25)
    parser.add_argument("--zip-command")
    parser.add_argument("--skip-part-md5", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    downloads = args.downloads
    created = ensure_unsplit_archive(
        downloads,
        args.unsplit_archive,
        verify_parts=not args.skip_part_md5,
        zip_command=args.zip_command,
    )
    result = prepare(
        downloads / "FSD50K.ground_truth.zip",
        downloads / "FSD50K.metadata.zip",
        args.unsplit_archive,
        args.output_root,
        args.manifest,
        args.attribution,
        general_per_primary_label=args.general_negatives_per_class,
        validation_percent=args.validation_percent,
    )
    result["created_unsplit_archive"] = created
    result["unsplit_archive"] = str(args.unsplit_archive.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
