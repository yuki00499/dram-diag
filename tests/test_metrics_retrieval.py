import numpy as np

from dram_diag.metrics import (multilabel_diagnostics, multilabel_metrics,
                               optimize_multilabel_thresholds)
from dram_diag.retrieval import RetrievalIndex


def test_retrieval_excludes_query_itself():
    records = [{"image_name": "a.jpg", "labels": [2]}, {"image_name": "b.jpg", "labels": [2]}]
    index = RetrievalIndex(np.array([[1., 0.], [.9, .1]]), records)
    assert index.search(np.array([1., 0.]), 1, "a.jpg")[0]["image_name"] == "b.jpg"


def test_multilabel_thresholds_and_core_metrics_are_label_aware():
    truth = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    probabilities = np.array([[.8, .2], [.7, .8], [.2, .7], [.1, .1]])
    thresholds, stability = optimize_multilabel_thresholds(
        truth, probabilities, [2, 7], rare_labels=[7], bootstrap_iterations=20, seed=3)
    result = multilabel_metrics(truth, probabilities, [2, 7], thresholds, core_labels=[2], tracked_pairs=[[2, 7]])
    assert result["core_macro_f1"] == 1.0
    assert result["pair_metrics"]["2+7"]["f1"] == 1.0
    assert stability["7"]["positive_support"] == 2


def test_multilabel_diagnostics_reports_confusion_and_bounded_pr_curve():
    truth = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    probabilities = np.array([[.9, .1], [.8, .7], [.2, .8], [.1, .2]])
    result = multilabel_diagnostics(truth, probabilities, [2, 7], [.5, .6], max_pr_points=3)
    assert result["confusion"]["2"] == {
        "tn": 2, "fp": 0, "fn": 0, "tp": 2,
        "precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2,
    }
    assert len(result["pr_curves"]["2"]) <= 3
    recalls = [point["recall"] for point in result["pr_curves"]["2"]]
    assert recalls == sorted(recalls)
    assert result["pr_curves"]["2"][0]["threshold"] is None


def test_multilabel_diagnostics_handles_single_class_label():
    truth = np.array([[0, 1], [0, 1], [0, 1]])
    probabilities = np.array([[.1, .8], [.2, .7], [.3, .9]])
    result = multilabel_diagnostics(truth, probabilities, [2, 7])
    assert result["confusion"]["2"]["support"] == 0
    assert result["pr_curves"]["2"] == []
