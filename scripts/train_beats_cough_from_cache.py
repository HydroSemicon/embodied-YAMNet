"""Train, calibrate, and evaluate a cough-only head from the frozen BEATs cache."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from daily_sound_detector import LABELS
from daily_sound_detector.calibration import choose_threshold, fit_temperature, precision_recall_points
from daily_sound_detector.config import load_config
from daily_sound_detector.models import LinearHead, sigmoid

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_beats_from_cache import _group_targets, clip_groups, load_cache  # noqa: E402

COUGH_INDEX = LABELS.index("cough")


def train_cough_mil(
    x: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    metadata: list[dict],
    *,
    device: str = "cuda",
    epochs: int = 400,
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

    train_indices, train_groups, train_meta = clip_groups(y, splits, metadata, "train")
    val_indices, val_groups, val_meta = clip_groups(y, splits, metadata, "validation")
    if not len(train_indices) or not len(val_indices):
        raise ValueError("cache must contain train and validation windows")
    train_group_y = _group_targets(y, train_indices, train_groups, len(train_meta))[:, COUGH_INDEX]
    val_group_y = _group_targets(y, val_indices, val_groups, len(val_meta))[:, COUGH_INDEX]

    center = x[train_indices].mean(axis=0).astype(np.float32)
    scale = np.maximum(x[train_indices].std(axis=0), 1e-4).astype(np.float32)
    train_x = torch.from_numpy((x[train_indices] - center) / scale).to(device)
    val_x = torch.from_numpy((x[val_indices] - center) / scale).to(device)
    train_group_index = torch.from_numpy(train_groups).to(device)
    val_group_index = torch.from_numpy(val_groups).to(device)
    train_targets = torch.from_numpy(train_group_y).to(device)
    val_targets = torch.from_numpy(val_group_y).to(device)

    weights = torch.nn.Parameter(torch.empty(x.shape[1], device=device))
    bias = torch.nn.Parameter(torch.zeros((), device=device))
    torch.nn.init.normal_(weights, mean=0.0, std=0.01)
    positives = train_targets.sum()
    negatives = len(train_targets) - positives
    pos_weight = torch.clamp(negatives / torch.clamp(positives, min=1), 1, 100)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW([weights, bias], lr=learning_rate, weight_decay=1e-4)

    def grouped_logits(window_x, group_index, group_count):
        window_logits = window_x @ weights + bias
        output = torch.full((group_count,), -torch.inf, dtype=window_logits.dtype, device=window_logits.device)
        return output.scatter_reduce(0, group_index, window_logits, reduce="amax", include_self=True)

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
            best_bias = float(bias.detach().cpu())
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

    if best_weights is None or best_bias is None:
        raise RuntimeError("training did not produce a checkpoint")
    raw_weights = best_weights / scale
    raw_bias = best_bias - float((center / scale) @ best_weights)
    packaged_weights = np.zeros((x.shape[1], len(LABELS)), np.float32)
    packaged_bias = np.full(len(LABELS), -20.0, np.float32)
    packaged_weights[:, COUGH_INDEX] = raw_weights.astype(np.float32)
    packaged_bias[COUGH_INDEX] = np.float32(raw_bias)
    training = {
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_validation_loss": best_loss,
        "learning_rate": learning_rate,
        "patience": patience,
        "seed": seed,
        "device": device,
        "positive_weight": float(pos_weight.detach().cpu()),
        "train_clips": len(train_meta),
        "validation_clips": len(val_meta),
        "history": history,
    }
    return LinearHead(packaged_weights, packaged_bias), training, center, scale


def _cough_logits_for_split(
    head: LinearHead,
    x: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    metadata: list[dict],
    split: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    indices, groups, group_metadata = clip_groups(y, splits, metadata, split)
    if not len(indices):
        return np.empty(0, np.float32), np.empty(0, np.float32), []
    targets = _group_targets(y, indices, groups, len(group_metadata))[:, COUGH_INDEX]
    window_logits = head.logits(x[indices])[:, COUGH_INDEX]
    logits = np.full(len(group_metadata), -np.inf, np.float32)
    np.maximum.at(logits, groups, window_logits)
    return logits, targets, group_metadata


def _binary_metrics(target: np.ndarray, predicted: np.ndarray) -> dict:
    target = np.asarray(target, bool)
    predicted = np.asarray(predicted, bool)
    tp = int(np.sum(target & predicted))
    fp = int(np.sum(~target & predicted))
    fn = int(np.sum(target & ~predicted))
    tn = int(np.sum(~target & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def run_pipeline(
    cache_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    device: str = "cuda",
    epochs: int = 400,
    learning_rate: float = 0.01,
    patience: int = 25,
) -> dict:
    x, y, splits, metadata_rows, cache_state = load_cache(cache_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    head, training, center, scale = train_cough_mil(
        x,
        y,
        splits,
        metadata_rows,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=patience,
    )
    model_metadata = {
        "architecture": "beats_cough_transfer",
        "extractor": "beats",
        "checkpoint": cache_state["checkpoint"],
        "checkpoint_sha256": cache_state["checkpoint_sha256"],
        "beats_source": cache_state["beats_source"],
        "beats_source_sha256": cache_state["beats_source_sha256"],
        "labels": list(LABELS),
        "enabled_labels": ["cough"],
        "embedding_dimension": int(x.shape[1]),
        "training_windows": int(np.sum(splits == "train")),
        "training_clips": training["train_clips"],
        "seed": training["seed"],
        "ood_center": center.astype(float).tolist(),
        "ood_scale": scale.astype(float).tolist(),
        "cache_manifest_sha256": cache_state["manifest_sha256"],
    }
    head_version = head.save(output_dir / "head.npz", model_metadata)

    validation_logits, validation_targets, _ = _cough_logits_for_split(
        head, x, y, splits, metadata_rows, "validation"
    )
    temperature = float(fit_temperature(validation_logits[:, None], validation_targets[:, None])[0])
    temperatures = np.ones(len(LABELS), np.float32)
    temperatures[COUGH_INDEX] = temperature
    calibrated = LinearHead(head.weights, head.bias, temperatures)
    validation_scores = sigmoid(validation_logits / temperature)
    selection = choose_threshold(validation_scores, validation_targets)

    config = replace(load_config(config_path), enabled_labels=("cough",))
    cough_config = config.classes["cough"]
    start = float(np.clip(selection["threshold"], 0.05, 1.0))
    calibrated_config = replace(
        config,
        classes={
            **config.classes,
            "cough": replace(
                cough_config,
                start_threshold=start,
                end_threshold=min(cough_config.end_threshold, max(0.01, start * 0.65)),
            ),
        },
        ood_max_distance=float(
            np.percentile(
                np.sqrt(np.mean(((x[splits == "validation"] - center) / scale) ** 2, axis=1)),
                99.5,
            )
        ),
    )
    model_metadata["parent_classifier_version"] = head_version
    calibrated_version = calibrated.save(output_dir / "head_calibrated.npz", model_metadata)
    calibration = {
        "source_split": "validation",
        "validation_clips": len(validation_targets),
        "temperature": temperature,
        "threshold": selection,
        "ood_max_distance": calibrated_config.ood_max_distance,
        "temporal_parameters": "defaults retained pending timestamped real-environment validation audio",
    }

    test_logits, test_targets, test_metadata = _cough_logits_for_split(
        calibrated, x, y, splits, metadata_rows, "test"
    )
    test_scores = sigmoid(test_logits / temperature)
    predicted = test_scores >= calibrated_config.classes["cough"].start_threshold
    metrics = _binary_metrics(test_targets, predicted)
    false_negatives = [
        {"path": row["path"], "label": "cough", "score": float(score)}
        for row, truth, hit, score in zip(test_metadata, test_targets, predicted, test_scores)
        if truth and not hit
    ]
    false_positives = [
        {"path": row["path"], "label": "cough", "score": float(score)}
        for row, truth, hit, score in zip(test_metadata, test_targets, predicted, test_scores)
        if not truth and hit
    ]
    evaluation = {
        "model": {
            "architecture": "beats_cough_transfer",
            "checkpoint": cache_state["checkpoint"],
            "classifier_version": calibrated_version,
        },
        "split": "test",
        "clips": len(test_targets),
        "enabled_labels": ["cough"],
        "per_class": {"cough": metrics},
        "macro_f1": metrics["f1"],
        "macro_recall": metrics["recall"],
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "deployment_ready": False,
        "deployment_blocker": "real-environment continuous negative audio has not been evaluated",
    }
    (output_dir / "training.json").write_text(json.dumps(training, indent=2), encoding="utf-8")
    (output_dir / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    (output_dir / "config_calibrated.json").write_text(
        json.dumps(asdict(calibrated_config), indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    (output_dir / "false_negatives.json").write_text(json.dumps(false_negatives, indent=2), encoding="utf-8")
    (output_dir / "false_positives.json").write_text(json.dumps(false_positives, indent=2), encoding="utf-8")
    with (output_dir / "pr_curve_cough.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("threshold", "precision", "recall"))
        writer.writeheader()
        writer.writerows(precision_recall_points(test_scores, test_targets))

    result = {
        "head": str(output_dir / "head.npz"),
        "head_calibrated": str(output_dir / "head_calibrated.npz"),
        "config_calibrated": str(output_dir / "config_calibrated.json"),
        "classifier_version": calibrated_version,
        "training": {key: value for key, value in training.items() if key != "history"},
        "calibration": calibration,
        "public_test": {"clips": evaluation["clips"], "per_class": evaluation["per_class"]},
        "deployment_ready": False,
        "next_gate": "evaluate timestamped cough events and long negative audio recorded in the target room",
    }
    (output_dir / "pipeline_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/beats-cache/public-combined"))
    parser.add_argument("--config", type=Path, default=Path("configs/cough_only.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/beats-cough-public"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=400)
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
