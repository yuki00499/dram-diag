import json
from pathlib import Path

import numpy as np

from .compat import file_sha256


def experiment_checkpoints(experiment_dir):
    summary_path = Path(experiment_dir) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = sorted(summary["runs"], key=lambda row: (row["fold"], row["seed"]))
    folds = [row["fold"] for row in runs]
    if folds != list(range(5)):
        raise ValueError("正式候选必须恰好包含 fold 0-4 各一个模型")
    paths = [Path(row["checkpoint"]) for row in runs]
    if any(not path.exists() for path in paths):
        raise FileNotFoundError("候选实验缺少 checkpoint")
    return summary, paths


def label_matrix(items, labels):
    index = {int(label): column for column, label in enumerate(labels)}
    matrix = np.zeros((len(items), len(labels)), dtype=int)
    for row, item in enumerate(items):
        for label in item.get("labels", []):
            matrix[row, index[int(label)]] = 1
    return matrix


def prediction_records(image_names, probabilities, labels, thresholds, model_version, margin=.1):
    thresholds = np.asarray(thresholds, dtype=float)
    records = []
    for image_name, scores in zip(image_names, probabilities):
        predicted = [int(label) for label, score, threshold in zip(labels, scores, thresholds)
                     if score >= threshold]
        low_confidence = bool(np.any(np.abs(scores - thresholds) <= margin))
        records.append({
            "image_name": image_name,
            "probabilities": {str(label): float(score) for label, score in zip(labels, scores)},
            "predicted_labels": predicted,
            "thresholds": {str(label): float(value) for label, value in zip(labels, thresholds)},
            "low_confidence": low_confidence,
            "model_version": model_version,
        })
    return records


def lock_payload(experiment_dir, manifest, thresholds, model_version):
    summary, checkpoints = experiment_checkpoints(experiment_dir)
    oof_path = Path(experiment_dir) / "oof_evaluation.json"
    return {
        "model_version": model_version,
        "experiment": summary["experiment"],
        "experiment_dir": str(Path(experiment_dir)),
        "protocol_version": manifest["protocol_version"],
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "thresholds": {str(key): float(value) for key, value in thresholds.items()},
        "checkpoints": [{"path": str(path), "sha256": file_sha256(path)} for path in checkpoints],
        "oof_evaluation": {"path": str(oof_path), "sha256": file_sha256(oof_path)},
    }
