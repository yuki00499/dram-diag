import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from dram_diag.compat import file_sha256
from dram_diag.competition import label_matrix, prediction_records
from dram_diag.inference import ModelRunner
from dram_diag.metrics import multilabel_bootstrap_ci, multilabel_metrics
from dram_diag.protocol import validate_multilabel_manifest


def main():
    parser = argparse.ArgumentParser(description="一次性评估锁定的 dram-ml-v2 测试集")
    parser.add_argument("--selection", default="artifacts/dram_ml_v2/model_selection.json")
    parser.add_argument("--manifest", default="artifacts/dram_ml_v2/split_manifest.json")
    parser.add_argument("--data-root", default="晶圆缺陷分类数据集")
    parser.add_argument("--out", default="artifacts/dram_ml_v2/test_report.json")
    args = parser.parse_args()
    destination = Path(args.out)
    predictions_path = destination.with_name("test_predictions.json")
    if destination.exists() or predictions_path.exists():
        raise FileExistsError("锁定测试报告已存在；不得覆盖或重复消费测试集")
    manifest = validate_multilabel_manifest(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    lock = selection["lock"]
    if lock["manifest_fingerprint"] != manifest["manifest_fingerprint"]:
        raise ValueError("模型选择清单与 manifest 不一致")
    for checkpoint in lock["checkpoints"]:
        if file_sha256(checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError(f"checkpoint 哈希不匹配: {checkpoint['path']}")
    runner = ModelRunner([item["path"] for item in lock["checkpoints"]], manifest, args.data_root)
    items = manifest["splits"]["test_known"]
    probabilities, _ = runner.infer_multilabel_items(items)
    labels = manifest["types"]
    thresholds = np.asarray([lock["thresholds"][str(label)] for label in labels])
    truth = label_matrix(items, labels)
    metrics = multilabel_metrics(truth, probabilities, labels, thresholds,
                                 manifest["core_types"], manifest["tracked_pairs"])
    ci = multilabel_bootstrap_ci(truth, probabilities, labels, thresholds, manifest["core_types"])
    records = prediction_records([item["image_name"] for item in items], probabilities, labels, thresholds,
                                 lock["model_version"])
    predictions = probabilities >= thresholds[None, :]
    exact = np.all(predictions == truth, axis=1)
    errors = np.mean(predictions != truth, axis=1)
    report_data = {"pr_curves": {}, "typical_successes": [], "typical_failures": []}
    for column, label in enumerate(labels):
        precision, recall, _ = precision_recall_curve(truth[:, column], probabilities[:, column])
        report_data["pr_curves"][str(label)] = {"precision": precision.tolist(), "recall": recall.tolist()}
    report_data["typical_successes"] = [records[index] for index in np.flatnonzero(exact)[:20]]
    report_data["typical_failures"] = [records[index] for index in np.argsort(errors)[::-1][:20] if errors[index] > 0]
    report = {"consumed_test_set": True, "selection_sha256": file_sha256(args.selection),
              "protocol_version": manifest["protocol_version"],
              "manifest_fingerprint": manifest["manifest_fingerprint"],
              "model_version": lock["model_version"], "metrics": metrics,
              "core_macro_f1_ci": ci, "rare_type_results": {
                  str(label): {"support": metrics["support"][str(label)],
                               "precision": metrics["per_label_precision"][str(label)],
                               "recall": metrics["per_label_recall"][str(label)],
                               "f1": metrics["per_label_f1"][str(label)]}
                  for label in manifest["rare_types"]}, "report_data": report_data}
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "report_data"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
