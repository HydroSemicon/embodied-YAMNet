"""Cache frozen BEATs embeddings in resumable chunks."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from daily_sound_detector import LABELS
from daily_sound_detector.audio import read_wav
from daily_sound_detector.manifest import ManifestItem, leakage_report, read_manifest
from daily_sound_detector.models import BeatsExtractor


SAMPLE_RATE = 16_000
WINDOW_SECONDS = 2.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
CACHE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class WindowSample:
    waveform: np.ndarray
    target: np.ndarray
    split: str
    source_id: str
    session_id: str
    path: str
    window_index: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_window_samples(items: list[ManifestItem]) -> Iterator[WindowSample]:
    for item in items:
        waveform, _ = read_wav(item.path, SAMPLE_RATE)
        if len(waveform) < WINDOW_SAMPLES:
            windows = [np.pad(waveform, (0, WINDOW_SAMPLES - len(waveform)))]
        else:
            windows = [
                waveform[start : start + WINDOW_SAMPLES]
                for start in range(0, len(waveform) - WINDOW_SAMPLES + 1, WINDOW_SAMPLES)
            ]
        target = np.asarray([label in item.labels for label in LABELS], np.uint8)
        for window_index, window in enumerate(windows):
            yield WindowSample(
                waveform=np.asarray(window, np.float32),
                target=target,
                split=item.split,
                source_id=item.source_id,
                session_id=item.session_id,
                path=str(item.path),
                window_index=window_index,
            )


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False,
        dir=path.parent, prefix=path.name + ".", suffix=".tmp",
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _chunk_paths(cache_dir: Path) -> list[Path]:
    chunks = sorted((cache_dir / "chunks").glob("chunk-*.npz"))
    for index, path in enumerate(chunks):
        if path.name != f"chunk-{index:06d}.npz":
            raise ValueError(f"non-contiguous cache chunks at {path}")
    return chunks


def _completed_samples(chunks: list[Path]) -> int:
    total = 0
    for path in chunks:
        with np.load(path, allow_pickle=False) as data:
            embeddings = data["embeddings"]
            targets = data["targets"]
            if embeddings.ndim != 2 or embeddings.shape[1] != 768:
                raise ValueError(f"invalid embedding shape in {path}: {embeddings.shape}")
            if targets.shape != (len(embeddings), len(LABELS)):
                raise ValueError(f"invalid target shape in {path}: {targets.shape}")
            total += len(embeddings)
    return total


def _write_chunk(
    path: Path,
    embeddings: list[np.ndarray],
    samples: list[WindowSample],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = [
        json.dumps({
            "split": sample.split,
            "source_id": sample.source_id,
            "session_id": sample.session_id,
            "path": sample.path,
            "window_index": sample.window_index,
        }, ensure_ascii=False, separators=(",", ":"))
        for sample in samples
    ]
    with tempfile.NamedTemporaryFile(
        "wb", delete=False, dir=path.parent, prefix=path.name + ".", suffix=".tmp"
    ) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                embeddings=np.stack(embeddings).astype(np.float32),
                targets=np.stack([sample.target for sample in samples]).astype(np.uint8),
                splits=np.asarray([sample.split for sample in samples]),
                metadata=np.asarray(metadata),
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _extract_batch(extractor: BeatsExtractor, samples: list[WindowSample]) -> np.ndarray:
    torch = extractor._torch
    waveforms = np.stack([sample.waveform for sample in samples]).astype(np.float32)
    source = torch.from_numpy(waveforms).to(extractor.device)
    padding = torch.zeros_like(source, dtype=torch.bool)
    with torch.inference_mode():
        features, _ = extractor._model.extract_features(source, padding_mask=padding)
        embeddings = features.mean(dim=1)
    result = embeddings.cpu().numpy().astype(np.float32)
    if result.shape != (len(samples), extractor.dimension):
        raise ValueError(f"unexpected BEATs output shape: {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("BEATs produced non-finite embeddings")
    return result


def cache_embeddings(
    manifest_path: Path,
    checkpoint_path: Path,
    beats_source: Path,
    cache_dir: Path,
    *,
    device: str = "cuda",
    batch_size: int = 8,
    chunk_size: int = 256,
) -> dict:
    if batch_size <= 0 or chunk_size <= 0:
        raise ValueError("batch_size and chunk_size must be positive")
    if chunk_size < batch_size:
        raise ValueError("chunk_size must be at least batch_size")

    manifest_path = manifest_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    beats_source = beats_source.resolve()
    cache_dir = cache_dir.resolve()
    state_path = cache_dir / "cache.json"
    identity = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "beats_source": str(beats_source),
        "beats_source_sha256": sha256_file(beats_source),
        "labels": list(LABELS),
        "sample_rate": SAMPLE_RATE,
        "window_seconds": WINDOW_SECONDS,
        "embedding_dimension": 768,
        "augmentation": None,
    }
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        for key, value in identity.items():
            if previous.get(key) != value:
                raise ValueError(
                    f"cache identity mismatch for {key}: {previous.get(key)!r} != {value!r}"
                )
        if previous.get("complete"):
            return previous
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(state_path, {
            **identity,
            "complete": False,
            "completed_samples": 0,
            "last_batch_size": batch_size,
            "last_chunk_size": chunk_size,
        })

    items = read_manifest(manifest_path)
    leak = leakage_report(items)
    if not leak["ok"]:
        raise ValueError(f"manifest identity leakage: {leak['leaks']}")

    chunks = _chunk_paths(cache_dir)
    completed = _completed_samples(chunks)
    sample_iterator = itertools.islice(iter_window_samples(items), completed, None)
    extractor = BeatsExtractor(checkpoint_path, beats_source, device)
    torch = extractor._torch
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    pending_embeddings: list[np.ndarray] = []
    pending_samples: list[WindowSample] = []
    chunk_index = len(chunks)
    started = time.perf_counter()
    processed_this_run = 0
    last_report = started

    while True:
        batch = list(itertools.islice(sample_iterator, batch_size))
        if not batch:
            break
        batch_embeddings = _extract_batch(extractor, batch)
        pending_embeddings.extend(batch_embeddings)
        pending_samples.extend(batch)
        processed_this_run += len(batch)

        while len(pending_samples) >= chunk_size:
            _write_chunk(
                cache_dir / "chunks" / f"chunk-{chunk_index:06d}.npz",
                pending_embeddings[:chunk_size],
                pending_samples[:chunk_size],
            )
            del pending_embeddings[:chunk_size]
            del pending_samples[:chunk_size]
            chunk_index += 1
            completed += chunk_size
            _atomic_json(
                state_path,
                {
                    **identity,
                    "complete": False,
                    "completed_samples": completed,
                    "last_batch_size": batch_size,
                    "last_chunk_size": chunk_size,
                },
            )

        now = time.perf_counter()
        if now - last_report >= 5:
            elapsed = max(now - started, 1e-9)
            rate = processed_this_run / elapsed
            print(
                f"cached={completed + len(pending_samples)} "
                f"run_rate={rate:.2f} windows/s chunks={chunk_index}",
                flush=True,
            )
            last_report = now

    if pending_samples:
        _write_chunk(
            cache_dir / "chunks" / f"chunk-{chunk_index:06d}.npz",
            pending_embeddings,
            pending_samples,
        )
        completed += len(pending_samples)
        chunk_index += 1

    elapsed = time.perf_counter() - started
    split_counts: Counter[str] = Counter()
    target_counts = {split: Counter() for split in ("train", "validation", "test")}
    for path in _chunk_paths(cache_dir):
        with np.load(path, allow_pickle=False) as data:
            for split, target in zip(data["splits"], data["targets"]):
                split_name = str(split)
                split_counts[split_name] += 1
                for index, present in enumerate(target):
                    if present:
                        target_counts[split_name][LABELS[index]] += 1

    result = {
        **identity,
        "complete": True,
        "completed_samples": completed,
        "chunks": chunk_index,
        "split_window_counts": dict(split_counts),
        "target_window_counts": {
            split: dict(counts) for split, counts in target_counts.items()
        },
        "last_run_seconds": elapsed,
        "last_run_windows_per_second": processed_this_run / max(elapsed, 1e-9),
        "device": device,
        "last_batch_size": batch_size,
        "last_chunk_size": chunk_size,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else None
        ),
    }
    _atomic_json(state_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/public-combined.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/BEATs_iter3+_AS2M.pt"))
    parser.add_argument("--beats-source", type=Path, default=Path("vendor/unilm/beats/BEATs.py"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/beats-cache/public-combined"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = cache_embeddings(
        args.manifest,
        args.checkpoint,
        args.beats_source,
        args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
