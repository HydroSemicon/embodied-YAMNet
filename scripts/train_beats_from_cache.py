"""Train, calibrate, and evaluate a five-output linear head from a BEATs cache."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from daily_sound_detector import LABELS
from daily_sound_detector.calibration import fit_temperature, precision_recall_points, choose_threshold
from daily_sound_detector.config import load_config
from daily_sound_detector.models import LinearHead, sigmoid


def load_cache(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], dict]:
    cache_dir = cache_dir.resolve()
    state = json.loads((cache_dir / "cache.json").read_text(encoding="utf-8"))
    if not state.get("complete"):
        raise ValueError("embedding cache is not complete")
    chunks = sorted((cache_dir / "chunks").glob("chunk-*.npz"))
    if not chunks:
        raise ValueError("embedding cache has no chunks")

    embeddings, targets, splits, metadata = [], [], [], []
    for chunk in chunks:
        with np.load(chunk, allow_pickle=False) as data:
            embeddings.append(np.asarray(data["embeddings"], np.float32))
            targets.append(np.asarray(data["targets"], np.uint8))
            splits.append(np.asarray(data["splits"]).astype(str))
            metadata.extend(json.loads(str(value)) for value in data["metadata"])
    x = np.concatenate(embeddings)
    y = np.concatenate(targets)
    split_array = np.concatenate(splits)
    if not (len(x) == len(y) == len(split_array) == len(metadata)):
        raise ValueError("cache arrays have inconsistent lengths")
    if x.shape[1] != 768 or y.shape[1] != len(LABELS):
        raise ValueError(f"unexpected cache shapes: embeddings={x.shape}, targets={y.shape}")
    return x, y, split_array, metadata, state


def clip_groups(
    targets: np.ndarray,
    splits: np.ndarray,
    metadata: list[dict],
    split: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    indices = np.flatnonzero(splits == split)
    group_by_session: dict[str, int] = {}
    group_indices = np.empty(len(indices), np.int64)
    group_targets: list[np.ndarray] = []
    group_metadata: list[dict] = []
    for local_index, source_index in enumerate(indices):
        row = metadata[int(source_index)]
        session_id = str(row["session_id"])
        group = group_by_session.get(session_id)
        if group is None:
            group = len(group_targets)
            group_by_session[session_id] = group
            group_targets.append(targets[source_index].copy())
            group_metadata.append(row)
        elif not np.array_equal(group_targets[group], targets[source_index]):
            raise ValueError(f"inconsistent targets for session {session_id}")
        group_indices[local_index] = group
    if not group_targets:
        return indices, group_indices, []
    return indices, group_indices, group_metadata


def aggregate_logits(logits: np.ndarray, group_indices: np.ndarray, groups: int) -> np.ndarray:
    result = np.full((groups, logits.shape[1]), -np.inf, np.float32)
    np.maximum.at(result, group_indices, logits)
    return result


def _group_targets(
    targets: np.ndarray,
    indices: np.ndarray,
    group_indices: np.ndarray,
    groups: int,
) -> np.ndarray:
    result = np.zeros((groups, targets.shape[1]), np.float32)
    for local_index, source_index in enumerate(indices):
        group = int(group_indices[local_index])
        value = targets[source_index]
        if result[group].any() and not np.array_equal(result[group], value):
            raise ValueError("inconsistent targets within a clip group")
        result[group] = value
    return result


def train_linear_mil(
    x: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    metadata: list[dict],
    *,
    device: str = "cuda",
    epochs: int = 200,
    learning_rate: float = 0.01,
    patience: int = 25,
    seed: int = 17,
) -> tuple[LinearHead, dict, np.ndarray, np.ndarray]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to train the cached BEATs head") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_indices, train_groups, train_group_meta = clip_groups(y, splits, metadata, "train")
    val_indices, val_groups, val_group_meta = clip_groups(y, splits, metadata, "validation")
    if not len(train_indices) or not len(val_indices):
        raise ValueError("cache must contain train and validation windows")
    train_group_y = _group_targets(y, train_indices, train_groups, len(train_group_meta))
    val_group_y = _group_targets(y, val_indices, val_groups, len(val_group_meta))

    center = x[train_indices].mean(axis=0).astype(np.float32)
    scale = np.maximum(x[train_indices].std(axis=0), 1e-4).astype(np.float32)
    train_x = torch.from_numpy((x[train_indices] - center) / scale).to(device)
    val_x = torch.from_numpy((x[val_indices] - center) / scale).to(device)
    train_group_index = torch.from_numpy(train_groups).to(device)
    val_group_index = torch.from_numpy(val_groups).to(device)
    train_targets = torch.from_numpy(train_group_y).to(device)
    val_targets = torch.from_numpy(val_group_y).to(device)

    weights = torch.nn.Parameter(torch.empty(x.shape[1], len(LABELS), device=device))
    bias = torch.nn.Parameter(torch.zeros(len(LABELS), device=device))
    torch.nn.init.normal_(weights, mean=0.0, std=0.01)
    positives = train_targets.sum(dim=0)
    negatives = len(train_targets) - positives
    pos_weight = torch.clamp(negatives / torch.clamp(positives, min=1), 1, 100)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW([weights, bias], lr=learning_rate, weight_decay=1e-4)

    def grouped_logits(window_x, group_index, group_count):
        window_logits = window_x @ weights + bias
        output = torch.full(
            (group_count, len(LABELS)), -torch.inf,
            dtype=window_logits.dtype, device=window_logits.device,
        )
        return output.scatter_reduce(
            0,
            group_index[:, None].expand(-1, len(LABELS)),
            window_logits,
            reduce="amax",
            include_self=True,
        )

    best_loss = float("inf")
    best_weights = None
    best_bias = None
    history: list[dict] = []
    stale = 0
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        train_logits = grouped_logits(train_x, train_group_index, len(train_targets))
        train_loss = criterion(train_logits, train_targets)
        train_loss.backward()
        optimizer.step()
        with torch.inference_mode():
            val_logits = grouped_logits(val_x, val_group_index, len(val_targets))
            val_loss = float(criterion(val_logits, val_targets).cpu())
        train_loss_value = float(train_loss.detach().cpu())
        history.append({"epoch": epoch, "train_loss": train_loss_value, "validation_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_weights = weights.detach().cpu().numpy().copy()
            best_bias = bias.detach().cpu().numpy().copy()
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch} train_loss={train_loss_value:.6f} "
                f"validation_loss={val_loss:.6f} best={best_loss:.6f}",
                flush=True,
            )
        if stale >= patience:
            break

    assert best_weights is not None and best_bias is not None
    raw_weights = best_weights / scale[:, None]
    raw_bias = best_bias - (center / scale) @ best_weights
    head = LinearHead(raw_weights.astype(np.float32), raw_bias.astype(np.float32))
    training = {
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_validation_loss": best_loss,
        "learning_rate": learning_rate,
        "patience": patience,
        "seed": seed,
        "device": device,
        "positive_weight": pos_weight.detach().cpu().numpy().astype(float).tolist(),
        "train_clips": len(train_group_meta),
        "validation_clips": len(val_group_meta),
        "history": history,
    }
    return head, training, center, scale


def _clip_logits_for_split(
    head: LinearHead,
    x: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    metadata: list[dict],
    split: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    indices, groups, group_metadata = clip_groups(y, splits, metadata, split)
    if not len(indices):
        return np.empty((0, len(LABELS))), np.empty((0, len(LABELS))), []
    targets = _group_targets(y, indices, groups, len(group_metadata))
    logits = aggregate_logits(head.logits(x[indices]), groups, len(group_metadata))
    return logits, targets, group_metadata


def _binary_metrics(target: np.ndarray, predicted: np.ndarray) -> dict:
    tp = int(np.sum((target == 1) & predicted))
    fp = int(np.sum((target == 0) & predicted))
    fn = int(np.sum((target == 1) & ~predicted))
    tn = int(np.sum((target == 0) & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if recall is not None and precision + recall else None
    return {
        "evaluable": bool(np.sum(target)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def run_pipeline(
    cache_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    device: str = "cuda",
    epochs: int = 200,
    learning_rate: float = 0.01,
    patience: int = 25,
) -> dict:
    x, y, splits, metadata_rows, cache_state = load_cache(cache_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    head, training, center, scale = train_linear_mil(
        x, y, splits, metadata_rows,
        device=device, epochs=epochs, learning_rate=learning_rate, patience=patience,
    )
    model_metadata = {
        "architecture": "beats_transfer",
        "extractor": "beats",
        "checkpoint": cache_state["checkpoint"],
        "checkpoint_sha256": cache_state["checkpoint_sha256"],
        "beats_source": cache_state["beats_source"],
        "beats_source_sha256": cache_state["beats_source_sha256"],
        "labels": list(LABELS),
        "embedding_dimension": int(x.shape[1]),
        "training_windows": int(np.sum(splits == "train")),
        "training_clips": training["train_clips"],
        "seed": training["seed"],
        "ood_center": center.astype(float).tolist(),
        "ood_scale": scale.astype(float).tolist(),
        "cache_manifest_sha256": cache_state["manifest_sha256"],
    }
    head_version = head.save(output_dir / "head.npz", model_metadata)

    validation_logits, validation_targets, _ = _clip_logits_for_split(
        head, x, y, splits, metadata_rows, "validation"
    )
    temperature = fit_temperature(validation_logits, validation_targets)
    calibrated = LinearHead(head.weights, head.bias, temperature)
    validation_scores = sigmoid(validation_logits / temperature)
    config = load_config(config_path)
    class_configs = {}
    selections = {}
    for index, label in enumerate(LABELS):
        selection = choose_threshold(validation_scores[:, index], validation_targets[:, index])
        selections[label] = selection
        old = config.classes[label]
        start = float(np.clip(selection["threshold"], 0.05, 0.99))
        class_configs[label] = replace(
            old,
            start_threshold=start,
            end_threshold=min(old.end_threshold, max(0.01, start * 0.65)),
        )
    distances = np.sqrt(np.mean(((x[splits == "validation"] - center) / scale) ** 2, axis=1))
    ood_threshold = float(np.percentile(distances, 99.5))
    calibrated_config = replace(
        config,
        classes=class_configs,
        ood_max_distance=ood_threshold,
    )
    model_metadata["parent_classifier_version"] = head_version
    calibrated_version = calibrated.save(output_dir / "head_calibrated.npz", model_metadata)
    calibration = {
        "source_split": "validation",
        "validation_clips": len(validation_targets),
        "temperature": temperature.astype(float).tolist(),
        "thresholds": selections,
        "ood_max_distance": ood_threshold,
        "temporal_parameters": "defaults retained; tune only with strongly timestamped real-environment validation audio",
    }
    (output_dir / "training.json").write_text(json.dumps(training, indent=2), encoding="utf-8")
    (output_dir / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    (output_dir / "config_calibrated.json").write_text(
        json.dumps(asdict(calibrated_config), indent=2), encoding="utf-8"
    )

    test_logits, test_targets, test_metadata = _clip_logits_for_split(
        calibrated, x, y, splits, metadata_rows, "test"
    )
    test_scores = sigmoid(test_logits / temperature) if len(test_logits) else test_logits
    thresholds = np.asarray(
        [calibrated_config.classes[label].start_threshold for label in LABELS], np.float32
    )
    predicted = test_scores >= thresholds if len(test_scores) else np.empty_like(test_scores, bool)
    per_class = {
        label: _binary_metrics(test_targets[:, index], predicted[:, index])
        for index, label in enumerate(LABELS)
    }
    evaluable_f1 = [metric["f1"] for metric in per_class.values() if metric["f1"] is not None]
    false_negatives = []
    false_positives = []
    for row, truth, prediction, scores in zip(test_metadata, test_targets, predicted, test_scores):
        for index, label in enumerate(LABELS):
            record = {"path": row["path"], "label": label, "score": float(scores[index])}
            if truth[index] and not prediction[index]:
                false_negatives.append(record)
            if not truth[index] and prediction[index]:
                false_positives.append(record)
    evaluation = {
        "model": {
            "architecture": "beats_transfer",
            "checkpoint": cache_state["checkpoint"],
            "classifier_version": calibrated_version,
        },
        "split": "test",
        "clips": len(test_targets),
        "per_class": per_class,
        "macro_f1_evaluable_classes": float(np.mean(evaluable_f1)) if evaluable_f1 else None,
        "not_evaluable_classes": [label for label, metric in per_class.items() if not metric["evaluable"]],
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "note": "alarm and timer_or_ringtone require a held-out public or real-environment test set",
    }
    (output_dir / "metrics.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    (output_dir / "false_negatives.json").write_text(json.dumps(false_negatives, indent=2), encoding="utf-8")
    (output_dir / "false_positives.json").write_text(json.dumps(false_positives, indent=2), encoding="utf-8")
    for index, label in enumerate(LABELS):
        with (output_dir / f"pr_curve_{label}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("threshold", "precision", "recall"))
            writer.writeheader()
            writer.writerows(precision_recall_points(test_scores[:, index], test_targets[:, index]))

    result = {
        "head": str(output_dir / "head.npz"),
        "head_calibrated": str(output_dir / "head_calibrated.npz"),
        "config_calibrated": str(output_dir / "config_calibrated.json"),
        "classifier_version": calibrated_version,
        "training": {key: value for key, value in training.items() if key != "history"},
        "calibration": calibration,
        "public_test": {
            "clips": evaluation["clips"],
            "per_class": evaluation["per_class"],
            "macro_f1_evaluable_classes": evaluation["macro_f1_evaluable_classes"],
            "not_evaluable_classes": evaluation["not_evaluable_classes"],
        },
    }
    (output_dir / "pipeline_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/beats-cache/public-combined"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/beats-public"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_pipeline(
        args.cache_dir,
        args.config,
        args.output_dir,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
