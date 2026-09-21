"""Auditable image-bootstrap metrics for dram-det-v3 ablations."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def box_iou(left, right) -> float:
    lx1, ly1, lx2, ly2 = map(float, left)
    rx1, ry1, rx2, ry2 = map(float, right)
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(0.0, min(ly2, ry2) - max(ly1, ry1))
    union = max(0.0, (lx2 - lx1) * (ly2 - ly1)) + max(0.0, (rx2 - rx1) * (ry2 - ry1)) - intersection
    return intersection / union if union else 0.0


def _class_events(records, class_id: int, iou_threshold: float, small_area: float | None = None,
                  min_confidence: float | None = None):
    predictions, positives = [], 0
    for record_index, record in enumerate(records):
        truths = [item for item in record.get("truth", []) if int(item["class_id"]) == class_id]
        if small_area is not None:
            width, height = record.get("image_size", [640, 640])
            scale = 640 / max(float(width), float(height))
            truths = [item for item in truths if
                      (item["bbox_xyxy"][2] - item["bbox_xyxy"][0])
                      * (item["bbox_xyxy"][3] - item["bbox_xyxy"][1]) * scale * scale < small_area]
        positives += len(truths)
        matched = set()
        candidates = sorted(
            (item for item in record.get("predictions", []) if int(item["class_id"]) == class_id),
            key=lambda item: float(item["confidence"]), reverse=True)
        for prediction in candidates:
            if min_confidence is not None and float(prediction["confidence"]) < min_confidence:
                continue
            if small_area is not None:
                width, height = record.get("image_size", [640, 640])
                scale = 640 / max(float(width), float(height))
                area = ((prediction["bbox_xyxy"][2] - prediction["bbox_xyxy"][0])
                        * (prediction["bbox_xyxy"][3] - prediction["bbox_xyxy"][1])) * scale * scale
                if area >= small_area:
                    continue
            choices = [(box_iou(prediction["bbox_xyxy"], truth["bbox_xyxy"]), index)
                       for index, truth in enumerate(truths) if index not in matched]
            best_iou, best_index = max(choices, default=(0.0, -1))
            is_true = best_iou >= iou_threshold
            if is_true:
                matched.add(best_index)
            predictions.append((float(prediction["confidence"]), int(is_true), record_index))
    return sorted(predictions, reverse=True), positives


def average_precision(events, positives: int) -> float:
    if positives == 0:
        return float("nan")
    if not events:
        return 0.0
    true_positive = np.cumsum([event[1] for event in events])
    false_positive = np.cumsum([1 - event[1] for event in events])
    recall = true_positive / positives
    precision = true_positive / np.maximum(true_positive + false_positive, 1)
    interpolated = []
    for level in np.linspace(0, 1, 101):
        eligible = precision[recall >= level]
        interpolated.append(float(eligible.max()) if len(eligible) else 0.0)
    return float(np.mean(interpolated))


def detection_metrics(records: list[dict], class_ids: list[int], small_area: float = 1024.0) -> dict:
    thresholds = np.linspace(.5, .95, 10)
    per_class = {}
    all_ap = []
    ap50 = []
    small_ap = []
    total_tp = total_fp = total_fn = 0
    for class_id in class_ids:
        aps = []
        for threshold in thresholds:
            events, positives = _class_events(records, class_id, float(threshold))
            aps.append(average_precision(events, positives))
        events50, positives = _class_events(records, class_id, .5, min_confidence=.25)
        tp = sum(event[1] for event in events50)
        fp = len(events50) - tp
        fn = positives - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives if positives else 0.0
        small_aps = []
        small_positives = 0
        for threshold in thresholds:
            small_events, small_positives = _class_events(records, class_id, float(threshold), small_area)
            small_aps.append(average_precision(small_events, small_positives))
        class_small_ap = float(np.nanmean(small_aps)) if any(not np.isnan(v) for v in small_aps) else float("nan")
        values = [value for value in aps if not np.isnan(value)]
        class_map = float(np.mean(values)) if values else float("nan")
        per_class[str(class_id)] = {
            "map50_95": class_map,
            "map50": aps[0],
            "precision50": precision,
            "recall50": recall,
            "support": positives,
            "ap_small50_95": class_small_ap,
            "small_support": small_positives,
        }
        if values:
            all_ap.extend(values)
            ap50.append(aps[0])
        if not np.isnan(class_small_ap):
            small_ap.append(class_small_ap)
        total_tp += tp; total_fp += fp; total_fn += fn
    return {
        "map50_95": float(np.mean(all_ap)) if all_ap else 0.0,
        "map50": float(np.mean(ap50)) if ap50 else 0.0,
        "ap_small50_95": float(np.mean(small_ap)) if small_ap else None,
        "precision50": total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0,
        "recall50": total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0,
        "false_negatives50": int(total_fn),
        "per_class": per_class,
    }


def hierarchical_classification_metrics(records: list[dict], class_ids: list[int],
                                        quality_ids: list[str] | None = None) -> dict:
    """Compute image-level defect/quality F1 and usability AUROC when records provide them."""
    quality_ids = list(quality_ids or [])
    result = {"global_defect_macro_f1": None, "quality_macro_f1": None, "usability_auroc": None}
    if records and all("global_truth" in row and "global_probabilities" in row for row in records):
        truth = np.asarray([[int(row["global_truth"].get(str(cid), row["global_truth"].get(cid, 0)))
                             for cid in class_ids] for row in records])
        pred = np.asarray([[float(row["global_probabilities"].get(str(cid), row["global_probabilities"].get(cid, 0))) >= .5
                            for cid in class_ids] for row in records])
        result["global_defect_macro_f1"] = float(f1_score(truth, pred, average="macro", zero_division=0))
    if quality_ids and records and all("quality_truth" in row and "quality_probabilities" in row for row in records):
        truth = np.asarray([[int(row["quality_truth"].get(qid, 0)) for qid in quality_ids] for row in records])
        pred = np.asarray([[float(row["quality_probabilities"].get(qid, 0)) >= .5 for qid in quality_ids]
                           for row in records])
        result["quality_macro_f1"] = float(f1_score(truth, pred, average="macro", zero_division=0))
    states = ("usable", "review", "unusable")
    if records and all("usability_truth" in row and "usability_probabilities" in row for row in records):
        truth = np.asarray([[int(row["usability_truth"] == state) for state in states] for row in records])
        prob = np.asarray([[float(row["usability_probabilities"].get(state, 0)) for state in states]
                           for row in records])
        valid = [index for index in range(3) if len(np.unique(truth[:, index])) == 2]
        if valid:
            result["usability_auroc"] = float(roc_auc_score(truth[:, valid], prob[:, valid], average="macro"))
    return result


def paired_bootstrap(left: list[dict], right: list[dict], class_ids: list[int],
                     iterations: int = 2000, seed: int = 20260920) -> dict:
    if [item["image_name"] for item in left] != [item["image_name"] for item in right]:
        raise ValueError("配对 bootstrap 要求两组记录的图像顺序完全一致")
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(iterations):
        indexes = rng.integers(0, len(left), len(left))
        left_value = detection_metrics([left[int(index)] for index in indexes], class_ids)["map50_95"]
        right_value = detection_metrics([right[int(index)] for index in indexes], class_ids)["map50_95"]
        differences.append(right_value - left_value)
    return {
        "iterations": iterations,
        "seed": seed,
        "mean_difference": float(np.mean(differences)),
        "lower": float(np.quantile(differences, .025)),
        "upper": float(np.quantile(differences, .975)),
    }


def review_budget_metrics(records: list[dict], budget: float = .2) -> dict:
    if not records:
        return {"budget": budget, "reviewed": 0, "false_negatives": 0, "captured": 0, "reduction": 0.0}
    count = max(1, round(len(records) * budget))
    ranked = sorted(records, key=lambda item: float(item.get("review_score", 0)), reverse=True)
    false_negatives = sum(int(item.get("false_negatives", 0)) for item in records)
    captured = sum(int(item.get("false_negatives", 0)) for item in ranked[:count])
    return {"budget": budget, "reviewed": count, "false_negatives": false_negatives,
            "captured": captured, "reduction": captured / false_negatives if false_negatives else 0.0}
