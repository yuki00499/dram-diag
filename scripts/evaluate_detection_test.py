"""Consume the locked dram-det-v3 test set once and emit an immutable report."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dram_diag.detection_inference import DetectionRuntime, validate_v3_deployment
from dram_diag.detection_metrics import (box_iou, detection_metrics,
                                         hierarchical_classification_metrics,
                                         review_budget_metrics)
from dram_diag.detection_data import validate_detection_manifest


def false_negative_count(truth, predictions, confidence=.25, iou=.5):
    count = 0
    for class_id in {int(item["class_id"]) for item in truth}:
        targets = [item for item in truth if int(item["class_id"]) == class_id]
        candidates = sorted(
            (item for item in predictions if int(item["class_id"]) == class_id
             and float(item["confidence"]) >= confidence),
            key=lambda item: float(item["confidence"]), reverse=True)
        matched = set()
        for candidate in candidates:
            choices = [(box_iou(candidate["bbox_xyxy"], target["bbox_xyxy"]), index)
                       for index, target in enumerate(targets) if index not in matched]
            best, index = max(choices, default=(0, -1))
            if best >= iou:
                matched.add(index)
        count += len(targets) - len(matched)
    return count


def main():
    parser = argparse.ArgumentParser(description="一次性评估冻结的 dram-det-v3 测试集")
    parser.add_argument("--deployment", default="artifacts/deployment-v3.json")
    parser.add_argument("--images", default="晶圆缺陷分类数据集/images")
    parser.add_argument("--out", default="artifacts/dram_det_v3/test_report.json")
    parser.add_argument("--lock", default="artifacts/dram_det_v3/test_consumed.lock")
    args = parser.parse_args()
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"v3 测试报告已存在，拒绝再次消费测试集: {destination}")
    lock_path = Path(args.lock)
    if lock_path.exists():
        raise FileExistsError(f"v3 测试集已被消费，锁文件为: {lock_path}")
    deployment_path = Path(args.deployment)
    deployment = validate_v3_deployment(
        json.loads(deployment_path.read_text(encoding="utf-8")), deployment_path.parent)
    if not deployment.get("ready"):
        raise ValueError("部署清单未锁定为 ready")
    manifest_path = deployment_path.parent / deployment["manifest"]
    manifest = validate_detection_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    if deployment.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("部署清单与测试 manifest 哈希不一致")
    if deployment["taxonomy_sha256"] != manifest["taxonomy_sha256"]:
        raise ValueError("部署清单与测试 taxonomy 哈希不一致")

    runtime = DetectionRuntime(deployment_path)
    if not runtime.ready:
        raise RuntimeError("v3 推理运行时未就绪")
    image_root = Path(args.images)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None
    rows, latencies = [], []
    class_ids = [int(value) for value in deployment["class_ids"]]
    quality_ids = [str(value) for value in deployment.get("quality_attributes", [])]
    for item in manifest["splits"]["test_known"]:
        path = image_root / item["image_name"]
        started = time.perf_counter()
        result = runtime.diagnose_path(path, item["image_name"])
        latencies.append((time.perf_counter() - started) * 1000)
        predictions = result["detections"]
        truth = item.get("objects", [])
        quality_probability = result.get("quality_probabilities", {})
        usability_probability = result.get("usability_probabilities", {})
        review_score = max(
            float(result.get("max_disagreement", 0)),
            1 - float(usability_probability.get("usable", 0)),
            max((float(value) for value in quality_probability.values()), default=0),
        )
        present = {int(obj["class_id"]) for obj in truth}
        rows.append({
            "image_name": item["image_name"],
            "image_size": [int(item["width"]), int(item["height"])],
            "truth": truth,
            "predictions": predictions,
            "global_truth": {str(class_id): int(class_id in present) for class_id in class_ids},
            "global_probabilities": result["global_defect_probabilities"],
            "quality_truth": {quality_id: int(quality_id in item.get("quality_attributes", []))
                              for quality_id in quality_ids},
            "quality_probabilities": quality_probability,
            "usability_truth": item["usability"],
            "usability_probabilities": usability_probability,
            "review_required": result["review_required"],
            "review_reasons": result["review_reasons"],
            "review_score": review_score,
            "false_negatives": false_negative_count(truth, predictions),
        })
    checkpoint = deployment_path.parent / deployment["checkpoint"]["path"]
    runtime_stats = {
        "latency_ms_per_image": sum(latencies) / len(latencies) if latencies else None,
        "latency_ms_p95": sorted(latencies)[round((len(latencies) - 1) * .95)] if latencies else None,
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 1024 ** 2
                         if torch is not None and torch.cuda.is_available() else None),
        "model_size_mb": checkpoint.stat().st_size / 1024 ** 2,
    }
    metrics = detection_metrics(rows, class_ids)
    metrics.update(hierarchical_classification_metrics(rows, class_ids, quality_ids))
    metrics["review_budget_20pct"] = review_budget_metrics(rows, .2)
    report = {
        "protocol_version": "dram-det-v3",
        "model_version": deployment["model_version"],
        "variant": deployment["variant"],
        "taxonomy_sha256": deployment["taxonomy_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_sha256": deployment["checkpoint"]["sha256"],
        "review_policy": deployment.get("review_policy", {}),
        "runtime": runtime_stats,
        "metrics": metrics,
        "records": rows,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({
        "report": str(destination.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_sha256": deployment["checkpoint"]["sha256"],
        "review_policy": deployment.get("review_policy", {}),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
