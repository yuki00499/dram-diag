import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from dram_diag.config import load_yaml
from dram_diag.protocol import build_multilabel_manifest, validate_multilabel_manifest


def main():
    parser = argparse.ArgumentParser(description="审计多标签标注数据并生成冻结协议")
    parser.add_argument("--data-root", default="晶圆缺陷分类数据集")
    parser.add_argument("--protocol", default="configs/protocols/ge20_ml.yaml")
    parser.add_argument("--out", default="artifacts/ge20_ml")
    parser.add_argument("--labels", default="artifacts/ge20_ml/labels.csv", help="多标签 CSV 路径")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = load_yaml(args.protocol)
    with Path(args.labels).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = [row["IMAGE_NAME"] for row in rows if not (Path(args.data_root) / "images" / row["IMAGE_NAME"]).exists()]
    if missing:
        raise ValueError(f"labels.csv 中存在缺失图片: {missing[:10]}")
    manifest = build_multilabel_manifest(rows, protocol)
    validate_multilabel_manifest(manifest, rows)
    report = {
        "task": "multilabel",
        "protocol_version": manifest["protocol_version"],
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "total_images": manifest["role_stats"]["total_images"],
        "types": manifest["types"],
        "type_counts": manifest["role_stats"]["types"],
        "split_samples": manifest["role_stats"]["split_samples"],
    }
    (out / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
