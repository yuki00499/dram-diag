"""Compare B0-B3 prediction records with paired image bootstrap."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dram_diag.detection_metrics import (detection_metrics, hierarchical_classification_metrics,
                                         paired_bootstrap, review_budget_metrics)


def load(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, {}
    return payload["records"], payload.get("runtime", {})


def main():
    parser = argparse.ArgumentParser(description="评估 dram-det-v3 消融实验")
    parser.add_argument("--b0", required=True)
    parser.add_argument("--b1")
    parser.add_argument("--b2")
    parser.add_argument("--b3")
    parser.add_argument("--classes", required=True, help="逗号分隔的类别 ID")
    parser.add_argument("--quality-attributes", default="", help="逗号分隔的质量属性 ID")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    class_ids = [int(value) for value in args.classes.split(",")]
    loaded = {key.upper(): load(value) for key, value in vars(args).items()
              if key in {"b0", "b1", "b2", "b3"} and value}
    records = {key: value[0] for key, value in loaded.items()}
    runtime = {key: value[1] for key, value in loaded.items()}
    quality_ids = [value for value in args.quality_attributes.split(",") if value]
    names = [item["image_name"] for item in records["B0"]]
    for variant, rows in records.items():
        if [item["image_name"] for item in rows] != names:
            raise ValueError(f"{variant} 与 B0 不是同一组配对图像")
    result = {"metrics": {}, "paired_vs_b0": {}, "acceptance": {}}
    for variant, rows in records.items():
        result["metrics"][variant] = detection_metrics(rows, class_ids)
        result["metrics"][variant].update(hierarchical_classification_metrics(rows, class_ids, quality_ids))
        result["metrics"][variant]["review_budget_20pct"] = review_budget_metrics(rows, .2)
        result["metrics"][variant]["runtime"] = runtime[variant]
        if variant != "B0":
            result["paired_vs_b0"][variant] = paired_bootstrap(
                records["B0"], rows, class_ids, args.bootstrap)
    base = result["metrics"]["B0"]
    for variant, metrics in result["metrics"].items():
        if variant == "B0":
            continue
        decision = {"retained": True, "reasons": []}
        if metrics["map50_95"] < base["map50_95"] - .005:
            decision["retained"] = False; decision["reasons"].append("mAP50-95 下降超过0.5个百分点")
        if variant in {"B1", "B3"}:
            base_small, value = base.get("ap_small50_95"), metrics.get("ap_small50_95")
            if base_small is None or value is None or value < base_small + .02:
                decision["retained"] = False; decision["reasons"].append("AP-small 未提升2个百分点")
            base_latency = base.get("runtime", {}).get("latency_ms_per_image")
            latency = metrics.get("runtime", {}).get("latency_ms_per_image")
            if base_latency is None or latency is None or latency > base_latency * 1.5:
                decision["retained"] = False; decision["reasons"].append("推理时延超过B0的1.5倍或未报告")
        if variant in {"B2", "B3"} and metrics["review_budget_20pct"]["reduction"] < .2:
            decision["retained"] = False; decision["reasons"].append("20%复核预算下漏检降低不足20%")
        result["acceptance"][variant] = decision
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
