import numpy as np
from sklearn.metrics import (average_precision_score, f1_score,
                             multilabel_confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, recall_score, roc_auc_score)


def multilabel_diagnostics(y_true, probabilities, labels, thresholds=.5, max_pr_points=101):
    """Return per-label confusion counts and compact precision-recall curves."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.ndim != 2 or probabilities.shape != y_true.shape:
        raise ValueError("标签与概率的维度不一致")
    thresholds = np.broadcast_to(np.asarray(thresholds, dtype=float), (y_true.shape[1],))
    predictions = (probabilities >= thresholds[None, :]).astype(int)
    matrix = multilabel_confusion_matrix(y_true, predictions, labels=range(y_true.shape[1]))
    confusion = {}
    pr_curves = {}
    for column, label in enumerate(labels):
        tn, fp, fn, tp = (int(value) for value in matrix[column].ravel())
        support = int(y_true[:, column].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        confusion[str(label)] = {
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "support": support,
        }
        if len(np.unique(y_true[:, column])) < 2:
            pr_curves[str(label)] = []
            continue
        curve_precision, curve_recall, curve_thresholds = precision_recall_curve(
            y_true[:, column], probabilities[:, column])
        # sklearn returns recall from 1 to 0; expose ascending recall for plotting.
        curve_threshold_values = list(curve_thresholds) + [None]
        points = [{"recall": float(recall_value), "precision": float(precision_value),
                   "threshold": None if threshold_value is None else float(threshold_value)}
                  for recall_value, precision_value, threshold_value in zip(
                      curve_recall[::-1], curve_precision[::-1], curve_threshold_values[::-1])]
        if len(points) > max_pr_points:
            indexes = np.linspace(0, len(points) - 1, max_pr_points, dtype=int)
            points = [points[int(index)] for index in indexes]
        pr_curves[str(label)] = points
    return {"confusion": confusion, "pr_curves": pr_curves}


def multilabel_metrics(y_true, probabilities, labels, threshold=.5, core_labels=None, tracked_pairs=None):
    """多标签评估：per-label F1/AUC、macro-F1、完全匹配率、Hamming 损失。

    y_true: (n, k) 0/1 矩阵；probabilities: (n, k) sigmoid 概率；labels: 类型 id 列表。
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    if y_true.shape[1] != probabilities.shape[1]:
        raise ValueError("标签与概率的维度不一致")
    thresholds = np.broadcast_to(np.asarray(threshold, dtype=float), (y_true.shape[1],))
    predictions = (probabilities >= thresholds[None, :]).astype(int)
    per_label_f1 = f1_score(y_true, predictions, labels=range(y_true.shape[1]), average=None, zero_division=0)
    per_label_auc = []
    for column in range(y_true.shape[1]):
        if len(np.unique(y_true[:, column])) < 2:
            per_label_auc.append(float("nan"))
        else:
            per_label_auc.append(float(roc_auc_score(y_true[:, column], probabilities[:, column])))
    exact_match = float(np.mean((predictions == y_true).all(axis=1)))
    hamming = float(np.mean(predictions != y_true))
    macro_f1 = float(np.nanmean(per_label_f1))
    micro_f1 = float(f1_score(y_true, predictions, average="micro", zero_division=0))
    sample_f1 = float(f1_score(y_true, predictions, average="samples", zero_division=0))
    macro_recall = float(recall_score(y_true, predictions, average="macro", zero_division=0))
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, predictions, average=None, zero_division=0)
    core_labels = list(core_labels or labels)
    label_to_column = {int(label): column for column, label in enumerate(labels)}
    core_columns = [label_to_column[int(label)] for label in core_labels]
    core_macro_f1 = float(np.mean(per_label_f1[core_columns]))
    per_label_ap = []
    for column in range(y_true.shape[1]):
        per_label_ap.append(float(average_precision_score(y_true[:, column], probabilities[:, column])))
    pair_metrics = {}
    for left, right in tracked_pairs or []:
        left_column, right_column = label_to_column[int(left)], label_to_column[int(right)]
        pair_true = y_true[:, left_column].astype(bool) & y_true[:, right_column].astype(bool)
        pair_pred = predictions[:, left_column] & predictions[:, right_column]
        pair_precision, pair_recall, pair_f1, _ = precision_recall_fscore_support(
            pair_true, pair_pred, average="binary", zero_division=0)
        pair_metrics[f"{left}+{right}"] = {
            "precision": float(pair_precision), "recall": float(pair_recall),
            "f1": float(pair_f1), "support": int(pair_true.sum()),
        }
    diagnostics = multilabel_diagnostics(y_true, probabilities, labels, thresholds)
    return {
        "macro_f1": macro_f1,
        "core_macro_f1": core_macro_f1,
        "micro_f1": micro_f1,
        "sample_f1": sample_f1,
        "map": float(np.mean(per_label_ap)),
        "macro_recall": macro_recall,
        "exact_match": exact_match,
        "hamming_loss": hamming,
        "per_label_f1": {str(label): float(value) for label, value in zip(labels, per_label_f1)},
        "per_label_auc": {str(label): value for label, value in zip(labels, per_label_auc)},
        "per_label_pr_auc": {str(label): value for label, value in zip(labels, per_label_ap)},
        "per_label_precision": {str(label): float(value) for label, value in zip(labels, precision)},
        "per_label_recall": {str(label): float(value) for label, value in zip(labels, recall)},
        "support": {str(label): int(y_true[:, column].sum()) for column, label in enumerate(labels)},
        "thresholds": {str(label): float(value) for label, value in zip(labels, thresholds)},
        "pair_metrics": pair_metrics,
        **diagnostics,
    }


def optimize_multilabel_thresholds(y_true, probabilities, labels, rare_labels=(), bootstrap_iterations=500,
                                   seed=42, shrinkage=20):
    """Calibrate label thresholds on OOF predictions; shrink unstable rare labels to 0.5."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    grid = np.linspace(.05, .95, 91)
    rng = np.random.default_rng(seed)
    rare_labels = {int(value) for value in rare_labels}
    thresholds, stability = [], {}

    def best_threshold(truth, scores):
        predicted = scores[:, None] >= grid[None, :]
        positive = truth[:, None].astype(bool)
        true_positive = np.sum(predicted & positive, axis=0)
        false_positive = np.sum(predicted & ~positive, axis=0)
        false_negative = np.sum(~predicted & positive, axis=0)
        denominator = 2 * true_positive + false_positive + false_negative
        values = np.divide(2 * true_positive, denominator, out=np.zeros_like(denominator, dtype=float),
                           where=denominator > 0)
        best = float(values.max())
        candidates = [float(threshold) for threshold, value in zip(grid, values) if value == best]
        return min(candidates, key=lambda value: (abs(value - .5), value))

    for column, label in enumerate(labels):
        truth, scores = y_true[:, column], probabilities[:, column]
        raw = best_threshold(truth, scores)
        positive = np.flatnonzero(truth == 1)
        negative = np.flatnonzero(truth == 0)
        boot = []
        if len(positive) and len(negative):
            for _ in range(bootstrap_iterations):
                indexes = np.r_[rng.choice(positive, len(positive), replace=True),
                                rng.choice(negative, len(negative), replace=True)]
                boot.append(best_threshold(truth[indexes], scores[indexes]))
        median = float(np.median(boot)) if boot else raw
        calibrated = median
        if int(label) in rare_labels:
            calibrated = (len(positive) * median + shrinkage * .5) / (len(positive) + shrinkage)
        thresholds.append(calibrated)
        stability[str(label)] = {
            "raw": raw, "bootstrap_median": median,
            "bootstrap_p05": float(np.quantile(boot, .05)) if boot else raw,
            "bootstrap_p95": float(np.quantile(boot, .95)) if boot else raw,
            "calibrated": float(calibrated), "positive_support": int(len(positive)),
        }
    return np.asarray(thresholds), stability


def multilabel_bootstrap_ci(y_true, probabilities, labels, thresholds, core_labels,
                            iterations=2000, seed=42):
    """Bootstrap by exact label combination to preserve multi-label structure."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    groups = {}
    for index, row in enumerate(y_true):
        groups.setdefault(tuple(row.tolist()), []).append(index)
    rng = np.random.default_rng(seed)
    values = []
    label_to_column = {int(label): column for column, label in enumerate(labels)}
    core_columns = [label_to_column[int(label)] for label in core_labels]
    predictions = probabilities >= np.asarray(thresholds)[None, :]
    for _ in range(iterations):
        indexes = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups.values()])
        truth_sample = y_true[indexes][:, core_columns].astype(bool)
        prediction_sample = predictions[indexes][:, core_columns]
        true_positive = np.sum(truth_sample & prediction_sample, axis=0)
        false_positive = np.sum(~truth_sample & prediction_sample, axis=0)
        false_negative = np.sum(truth_sample & ~prediction_sample, axis=0)
        denominator = 2 * true_positive + false_positive + false_negative
        per_label = np.divide(2 * true_positive, denominator, out=np.zeros_like(denominator, dtype=float),
                              where=denominator > 0)
        values.append(float(np.mean(per_label)))
    return {"iterations": iterations, "seed": seed, "lower": float(np.quantile(values, .025)),
            "upper": float(np.quantile(values, .975))}
