from dram_diag.detection_metrics import (box_iou, detection_metrics, hierarchical_classification_metrics,
                                         paired_bootstrap, review_budget_metrics)


def records(confidence=.9, offset=0):
    return [{
        "image_name": "a.jpg",
        "truth": [{"class_id": 0, "bbox_xyxy": [0, 0, 20, 20]}],
        "predictions": [{"class_id": 0, "bbox_xyxy": [offset, 0, 20 + offset, 20],
                         "confidence": confidence}],
        "review_score": .9, "false_negatives": int(offset > 20),
    }, {
        "image_name": "b.jpg", "truth": [], "predictions": [],
        "review_score": .1, "false_negatives": 0,
    }]


def test_detection_metrics_and_pairing_are_image_based():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    good = detection_metrics(records(), [0])
    bad = detection_metrics(records(offset=30), [0])
    assert good["map50_95"] == 1
    assert good["ap_small50_95"] == 1
    assert bad["map50_95"] == 0
    paired = paired_bootstrap(records(offset=30), records(), [0], iterations=20, seed=1)
    assert paired["mean_difference"] > 0


def test_review_budget_reports_captured_false_negatives():
    rows = records(offset=30)
    result = review_budget_metrics(rows, .5)
    assert result["reviewed"] == 1
    assert result["reduction"] == 1


def test_hierarchical_classification_metrics_cover_all_heads():
    rows = [
        {"global_truth": {"0": 1}, "global_probabilities": {"0": .9},
         "quality_truth": {"blur": 1}, "quality_probabilities": {"blur": .8},
         "usability_truth": "usable", "usability_probabilities": {"usable": .9, "review": .1, "unusable": 0}},
        {"global_truth": {"0": 0}, "global_probabilities": {"0": .1},
         "quality_truth": {"blur": 0}, "quality_probabilities": {"blur": .2},
         "usability_truth": "unusable", "usability_probabilities": {"usable": .1, "review": .1, "unusable": .8}},
    ]
    metrics = hierarchical_classification_metrics(rows, [0], ["blur"])
    assert metrics["global_defect_macro_f1"] == 1
    assert metrics["quality_macro_f1"] == 1
    assert metrics["usability_auroc"] == 1
