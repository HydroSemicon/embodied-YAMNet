"""Safely extract VocalSound 16 kHz and create the detector manifest.

The official train/validation/test files bundled in the release are preserved.
Only cough, sneeze, and laughter are detector targets; the other three vocal
sound classes are useful hard negatives for those targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


EXPECTED_ARCHIVE_SIZE = 1_783_406_544
EXPECTED_ARCHIVE_SHA256 = "66b83ee62e79c059b051c6f4afac06497945931c10c7f643a928b92f1ff9b9f2"

SPLIT_MEMBERS = {
    "train": "datafiles/tr.json",
    "validation": "datafiles/val.json",
    "test": "datafiles/te.json",
}

MID_TO_SOURCE_LABEL = {
    "/m/01j3sz": "laughter",
    "/m/07plz5l": "sigh",
    "/m/01b_21": "cough",
    "/m/0dl9sf8": "throatclearing",
    "/m/01hsr_": "sneeze",
    "/m/07ppn3j": "sniff",
}

TARGET_LABELS = {"cough", "sneeze", "laughter"}
HARD_NEGATIVE_LABELS = {"sigh", "sniff", "throatclearing"}

METADATA_MEMBERS = (
    "LICENSE",
    "class_labels_indices_vs.csv",
    "datafiles/tr.json",
    "datafiles/val.json",
    "datafiles/te.json",
    "meta/tr_meta.csv",
    "meta/val_meta.csv",
    "meta/te_meta.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _extract_member(archive: zipfile.ZipFile, member: str, root: Path) -> bool:
    info = archive.getinfo(member)
    destination = _safe_destination(root, member)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and destination.stat().st_size == info.file_size:
        return False
    if destination.exists():
        raise FileExistsError(
            f"existing file has a different size: {destination}; "
            "remove the incomplete file and run again"
        )

    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        temporary.unlink()
    try:
        with archive.open(info, "r") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def _read_official_splits(archive: zipfile.ZipFile) -> dict[str, list[dict]]:
    splits: dict[str, list[dict]] = {}
    for split, member in SPLIT_MEMBERS.items():
        payload = json.loads(archive.read(member))
        rows = payload.get("data")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{member} does not contain a non-empty data list")
        splits[split] = rows

    speaker_sets = {
        split: {str(row["spk_id"]) for row in rows}
        for split, rows in splits.items()
    }
    split_names = list(speaker_sets)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = speaker_sets[left] & speaker_sets[right]
            if overlap:
                sample = sorted(overlap)[:5]
                raise ValueError(
                    f"official splits have speaker leakage between {left} and {right}: {sample}"
                )
    return splits


def _relative_manifest_path(audio_path: Path, manifest_path: Path) -> str:
    return Path(os.path.relpath(audio_path, manifest_path.parent)).as_posix()


def prepare(
    archive_path: Path,
    output_root: Path,
    manifest_path: Path,
    *,
    expected_sha256: str | None = EXPECTED_ARCHIVE_SHA256,
) -> dict:
    archive_path = archive_path.resolve()
    output_root = output_root.resolve()
    manifest_path = manifest_path.resolve()

    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    archive_size = archive_path.stat().st_size
    if expected_sha256 == EXPECTED_ARCHIVE_SHA256 and archive_size != EXPECTED_ARCHIVE_SIZE:
        raise ValueError(
            f"unexpected archive size: {archive_size}; expected {EXPECTED_ARCHIVE_SIZE}"
        )
    archive_sha256 = sha256_file(archive_path)
    if expected_sha256 and archive_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"archive SHA-256 mismatch: {archive_sha256}; expected {expected_sha256}"
        )

    extracted = 0
    skipped = 0
    manifest_rows: list[dict] = []
    label_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    speaker_counts: dict[str, int] = {}

    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"CRC failure in ZIP member: {bad_member}")

        splits = _read_official_splits(archive)
        available = set(archive.namelist())
        for member in METADATA_MEMBERS:
            if member not in available:
                raise FileNotFoundError(f"required ZIP member is missing: {member}")
            if _extract_member(archive, member, output_root):
                extracted += 1
            else:
                skipped += 1

        for split, rows in splits.items():
            speaker_counts[split] = len({str(row["spk_id"]) for row in rows})
            for source_row in rows:
                speaker_id = str(source_row["spk_id"])
                mid = str(source_row["labels"])
                try:
                    source_label = MID_TO_SOURCE_LABEL[mid]
                except KeyError as exc:
                    raise ValueError(f"unknown VocalSound label MID: {mid}") from exc

                filename = PurePosixPath(str(source_row["wav"])).name
                member = f"audio_16k/{filename}"
                if member not in available:
                    raise FileNotFoundError(f"audio referenced by {split} split is missing: {member}")
                audio_path = _safe_destination(output_root, member)
                if _extract_member(archive, member, output_root):
                    extracted += 1
                else:
                    skipped += 1

                detector_labels = [source_label] if source_label in TARGET_LABELS else []
                manifest_rows.append(
                    {
                        "path": _relative_manifest_path(audio_path, manifest_path),
                        "split": split,
                        "labels": detector_labels,
                        "source_id": f"vocalsound:{speaker_id}",
                        "session_id": f"vocalsound:{Path(filename).stem}",
                        "person_id": f"vocalsound:{speaker_id}",
                        "device_id": None,
                        "is_hard_negative": source_label in HARD_NEGATIVE_LABELS,
                        "playback": False,
                        "events": [],
                    }
                )
                label_counts[source_label] += 1
                split_counts[split] += 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False,
        dir=manifest_path.parent, prefix=manifest_path.name + ".", suffix=".tmp",
    ) as handle:
        temporary_manifest = Path(handle.name)
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary_manifest, manifest_path)

    return {
        "archive": str(archive_path),
        "archive_size": archive_size,
        "archive_sha256": archive_sha256,
        "output_root": str(output_root),
        "manifest": str(manifest_path),
        "clips": len(manifest_rows),
        "split_counts": dict(split_counts),
        "speaker_counts": speaker_counts,
        "source_label_counts": dict(sorted(label_counts.items())),
        "extracted_files": extracted,
        "already_present_files": skipped,
        "speaker_leakage": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("downloads/vs_release_16k.zip"))
    parser.add_argument("--output-root", type=Path, default=Path("data/public/vocalsound"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/vocalsound.jsonl"))
    parser.add_argument(
        "--skip-archive-hash",
        action="store_true",
        help="Skip the known SHA-256 comparison (ZIP CRC is still fully checked).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = prepare(
        args.archive,
        args.output_root,
        args.manifest,
        expected_sha256=None if args.skip_archive_hash else EXPECTED_ARCHIVE_SHA256,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
