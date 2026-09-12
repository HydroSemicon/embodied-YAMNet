"""Run the resumable public-data BEATs cache, training, calibration, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cache_beats_embeddings import cache_embeddings
from train_beats_from_cache import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/public-combined.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/BEATs_iter3+_AS2M.pt"))
    parser.add_argument("--beats-source", type=Path, default=Path("vendor/unilm/beats/BEATs.py"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/beats-cache/public-combined"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/beats-public"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_result = cache_embeddings(
        args.manifest,
        args.checkpoint,
        args.beats_source,
        args.cache_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    pipeline_result = run_pipeline(
        args.cache_dir,
        args.config,
        args.output_dir,
        device=args.device,
        epochs=args.epochs,
    )
    print(json.dumps({
        "cache": {
            "completed_samples": cache_result["completed_samples"],
            "chunks": cache_result["chunks"],
            "split_window_counts": cache_result["split_window_counts"],
        },
        "pipeline": pipeline_result,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
