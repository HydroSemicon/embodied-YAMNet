from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from daily_sound_detector.models import BeatsExtractor


EXPECTED_SHA256 = "d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the pinned BEATs checkpoint and CUDA embedding extraction")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/BEATs_iter3+_AS2M.pt"))
    parser.add_argument("--beats-source", type=Path, default=Path("vendor/unilm/beats/BEATs.py"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    actual_hash = sha256_file(args.checkpoint)
    if actual_hash != EXPECTED_SHA256:
        raise RuntimeError(f"checkpoint SHA-256 mismatch: {actual_hash}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    extractor = BeatsExtractor(args.checkpoint, args.beats_source, args.device)
    loaded = time.perf_counter()
    embedding = extractor.extract(np.zeros(32000, dtype=np.float32))
    finished = time.perf_counter()

    print(f"checkpoint_sha256: {actual_hash}")
    print(f"device: {extractor.device}")
    print(f"embedding_shape: {embedding.shape}")
    print(f"finite: {bool(np.isfinite(embedding).all())}")
    print(f"load_seconds: {loaded - started:.3f}")
    print(f"inference_seconds: {finished - loaded:.3f}")
    if args.device == "cuda":
        print(f"peak_gpu_memory_mb: {torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
