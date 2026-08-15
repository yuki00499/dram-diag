import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, recall_score, roc_auc_score, roc_curve)


def expected_calibration_error(y_true, probabilities, bins=15):
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == np.asarray(y_true)
    result = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            result += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(result)


def closed_set_metrics(y_true, probabilities, labels):
    y_true = np.asarray(y_true)
    prediction = probabilities.argmax(axis=1)
    top = np.argsort(probabilities, axis=1)[:, -min(5, probabilities.shape[1]):]
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(recall_score(y_true, prediction, labels=labels, average="macro", zero_division=0)),
        "top5_accuracy": float(np.mean([value in row for value, row in zip(y_true, top)])),
        "ece": expected_calibration_error(y_true, probabilities),
        "per_class_recall": {str(label): float(value) for label, value in zip(labels, recall_score(y_true, prediction, labels=labels, average=None, zero_division=0))},
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=labels).tolist(),
    }


def stratified_bootstrap(y_true, values, metric, iterations=2000, seed=42):
    y_true = np.asarray(y_true)
    values = np.asarray(values)
    groups = [np.flatnonzero(y_true == label) for label in np.unique(y_true)]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        indexes = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        samples.append(float(metric(y_true[indexes], values[indexes])))
    return {"iterations": iterations, "seed": seed, "lower": float(np.quantile(samples, .025)), "upper": float(np.quantile(samples, .975))}


def open_set_metrics(known_scores, unknown_scores, known_classes=None, unknown_classes=None, threshold=None):
    known_scores, unknown_scores = np.asarray(known_scores), np.asarray(unknown_scores)
    truth = np.r_[np.zeros(len(known_scores)), np.ones(len(unknown_scores))]
    scores = np.r_[known_scores, unknown_scores]
    fpr, tpr, _ = roc_curve(truth, scores)
    index = int(np.argmin(np.abs(tpr - .95)))
    result = {
        "auroc": float(roc_auc_score(truth, scores)),
        "aupr": float(average_precision_score(truth, scores)),
        "fpr_at_95_tpr": float(fpr[index]),
    }
    if threshold is not None:
        result["known_false_reject_rate"] = float(np.mean(known_scores >= threshold))
        result["unknown_false_accept_rate"] = float(np.mean(unknown_scores < threshold))
        result["unknown_recall"] = float(np.mean(unknown_scores >= threshold))
        if unknown_classes is not None:
            result["macro_unknown_recall"] = float(np.mean([np.mean(unknown_scores[np.asarray(unknown_classes) == value] >= threshold) for value in np.unique(unknown_classes)]))
    return result
