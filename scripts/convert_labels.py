"""将标注站导出的 label_v2.json 转换为多标签 CSV，并生成统计报告。

用法:
    python scripts/convert_labels.py --labels label_v2.json --out artifacts/ge20_ml
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from dram_diag.data import load_labels

KEEP_TYPES = (1, 2, 3, 4, 5, 7, 8)  # 6=Dark spot、9=Unknown 未被标注使用，删除


def main():
    parser = argparse.ArgumentParser(description="转换多标签标注为 CSV")
    parser.add_argument("--labels", default="label_v2.json")
    parser.add_argument("--data-root", default="晶圆缺陷分类数据集")
    parser.add_argument("--out", default="artifacts/ge20_ml")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    labels = payload.get("labels", {})
    unlabeled = payload.get("unlabeled", [])
    rows = load_labels(args.data_root)
    image_root = Path(args.data_root) / "images"
    all_names = [row["IMAGE_NAME"] for row in rows]

    dropped_types = sorted(set(payload.get("types", []) and []))
    used = Counter()
    per_count = Counter()
    kept, excluded = [], []
    for name in all_names:
        if name not in labels:
            excluded.append({"image_name": name, "reason": "unlabeled"})
            continue
        type_list = [int(value) for value in labels[name]]
        dropped = sorted(set(type_list) - set(KEEP_TYPES))
        if dropped:
            excluded.append({"image_name": name, "reason": "contains_dropped_types", "types": dropped})
            continue
        if not (image_root / name).exists():
            excluded.append({"image_name": name, "reason": "image_missing"})
            continue
        kept.append((name, type_list))
        used.update(type_list)
        per_count[len(type_list)] += 1

    with (out / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["IMAGE_NAME", "LABELS"])
        for name, type_list in sorted(kept, key=lambda item: item[0]):
            writer.writerow([name, ",".join(str(value) for value in type_list)])

    report = {
        "source": str(Path(args.labels)),
        "total_images": len(all_names),
        "kept_images": len(kept),
        "excluded": excluded,
        "excluded_by_reason": {reason: sum(1 for item in excluded if item["reason"] == reason) for reason in sorted({item["reason"] for item in excluded})},
        "types_used": {str(tid): used[tid] for tid in KEEP_TYPES},
        "labels_per_image": {str(k): v for k, v in sorted(per_count.items())},
        "dropped_types": [6, 9],
    }
    (out / "convert_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
