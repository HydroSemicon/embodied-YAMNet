from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from . import LABELS
from .calibration import calibrate_head, choose_threshold
from .config import load_config
from .evaluation import evaluate_clips, evaluate_continuous, score_recording
from .live import run_live
from .manifest import read_manifest, write_prepared_index
from .models import LinearHead, TransferScorer, YamnetBaseline
from .reporting import export_report
from .training import collect_embeddings, make_extractor, run_training
from .tuning import tune_temporal


def _common_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=["yamnet_standard", "yamnet_transfer", "beats_transfer", "mock_transfer"], required=True)
    parser.add_argument("--head", help="Path to trained head.npz for transfer models")
    parser.add_argument("--checkpoint", help="Local BEATs checkpoint path")
    parser.add_argument("--beats-source", help="Official BEATs.py from microsoft/unilm")
    parser.add_argument("--inference-device", choices=["cpu", "cuda"], default="cpu")


def _load_scorer(args):
    if args.model == "yamnet_standard": return YamnetBaseline()
    if not args.head: raise ValueError("transfer models require --head")
    head, metadata = LinearHead.load(args.head)
    extractor_name = {"yamnet_transfer": "yamnet", "beats_transfer": "beats", "mock_transfer": "mock"}[args.model]
    extractor = make_extractor(extractor_name, args.checkpoint, args.beats_source, args.inference_device)
    checkpoint = args.checkpoint or metadata.get("checkpoint", extractor_name)
    version = hashlib.sha256(Path(args.head).read_bytes()).hexdigest()
    return TransferScorer(extractor, head, str(checkpoint), version, metadata)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sound-detector")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare-dataset"); p.add_argument("--manifest", required=True); p.add_argument("--output", required=True)
    p = commands.add_parser("train"); p.add_argument("--manifest", required=True); p.add_argument("--extractor", choices=["yamnet", "beats", "mock"], required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--checkpoint"); p.add_argument("--beats-source"); p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--inference-device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--learning-rate", type=float, default=.05); p.add_argument("--augmentation-config")
    p = commands.add_parser("calibrate"); p.add_argument("--manifest", required=True); p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True); _common_model(p)
    p = commands.add_parser("evaluate"); p.add_argument("--manifest", required=True); p.add_argument("--config", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="test"); p.add_argument("--raw-scores", action="store_true"); _common_model(p)
    p = commands.add_parser("evaluate-continuous"); p.add_argument("--manifest", required=True); p.add_argument("--config", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--split", default="test"); _common_model(p)
    p = commands.add_parser("live"); p.add_argument("--config", required=True); p.add_argument("--event-log", required=True); p.add_argument("--raw-log")
    p.add_argument("--device"); p.add_argument("--duration", type=float); p.add_argument("--record-wav")
    p.add_argument("--events-only", action="store_true", help="Print one line only when an event is finalized")
    _common_model(p)
    p = commands.add_parser("export-report"); p.add_argument("--run-dir", action="append", required=True); p.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-dataset":
        result = write_prepared_index(read_manifest(args.manifest), args.output)
    elif args.command == "train":
        items = read_manifest(args.manifest); extractor = make_extractor(args.extractor, args.checkpoint, args.beats_source, args.inference_device)
        augmentation = json.loads(Path(args.augmentation_config).read_text(encoding="utf-8")) if args.augmentation_config else None
        checkpoint = args.checkpoint or ("yamnet/1" if args.extractor == "yamnet" else args.extractor)
        result = run_training(items, extractor, args.output_dir, args.epochs, args.learning_rate, augmentation, str(checkpoint))
    elif args.command == "calibrate":
        items, config = read_manifest(args.manifest), load_config(args.config)
        if args.model == "yamnet_standard":
            scorer, scores, targets = _load_scorer(args), [], []
            for item in items:
                if item.split != "validation": continue
                from .audio import read_wav
                clip_scores, _, _ = score_recording(scorer, read_wav(item.path, config.sample_rate)[0], config)
                scores.append(clip_scores); targets.append([label in item.labels for label in LABELS])
            if not scores: raise ValueError("no validation items")
            selections, classes = {}, {}
            for i, label in enumerate(LABELS):
                selections[label] = choose_threshold(np.asarray(scores)[:, i], np.asarray(targets)[:, i])
                old = config.classes[label]; start = float(np.clip(selections[label]["threshold"], .05, .99))
                classes[label] = replace(old, start_threshold=start, end_threshold=min(old.end_threshold, start * .65))
            updated, result = replace(config, classes=classes), {"thresholds": selections, "source_split": "validation", "temperature": None}
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            updated, temporal = tune_temporal(items, scorer, updated); result["temporal_search"] = temporal
            Path(args.output_dir, "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        else:
            if not args.head: raise ValueError("transfer model calibration requires --head")
            head, metadata = LinearHead.load(args.head); extractor = make_extractor(metadata["extractor"], args.checkpoint, args.beats_source, args.inference_device)
            x, y = collect_embeddings(items, "validation", extractor)
            calibrated, updated, result = calibrate_head(head, x, y, config, args.output_dir)
            center, scale = np.asarray(metadata.get("ood_center", [])), np.asarray(metadata.get("ood_scale", []))
            if center.shape == (x.shape[1],) and scale.shape == (x.shape[1],):
                distances = np.sqrt(np.mean(((x - center) / np.maximum(scale, 1e-4)) ** 2, axis=1))
                updated = replace(updated, ood_max_distance=float(np.percentile(distances, 99.5)))
                result["ood_max_distance"] = updated.ood_max_distance
            metadata["classifier_version"] = calibrated.save(Path(args.output_dir) / "head_calibrated.npz", metadata)
            temporal_scorer = TransferScorer(extractor, calibrated, str(args.checkpoint or metadata.get("checkpoint", metadata["extractor"])), metadata["classifier_version"], metadata)
            updated, temporal = tune_temporal(items, temporal_scorer, updated); result["temporal_search"] = temporal
            Path(args.output_dir, "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        Path(args.output_dir, "config_calibrated.json").write_text(json.dumps(asdict(updated), indent=2), encoding="utf-8")
    elif args.command in {"evaluate", "evaluate-continuous"}:
        items, config, scorer = read_manifest(args.manifest), load_config(args.config), _load_scorer(args)
        result = (evaluate_clips(items, scorer, config, args.output_dir, args.split, getattr(args, "raw_scores", False))
                  if args.command == "evaluate" else evaluate_continuous(items, scorer, config, args.output_dir, args.split))
    elif args.command == "live":
        run_live(_load_scorer(args), load_config(args.config), args.event_log, args.raw_log, args.device,
                 args.duration, args.record_wav, args.events_only)
        result = {"event_log": args.event_log}
    else:
        result = export_report(args.run_dir, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
