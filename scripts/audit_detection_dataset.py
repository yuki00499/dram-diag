"""Audit or freeze the canonical dram-det-v3 annotation source."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dram_diag.detection_data import audit_annotations, freeze_annotations, load_taxonomy


def main():
    parser = argparse.ArgumentParser(description="审计 dram-det-v3 框标注")
    parser.add_argument("--taxonomy", default="configs/taxonomy_v3.yaml")
    parser.add_argument("--annotations", default="annotations/dram_det_v3.json")
    parser.add_argument("--images", default="晶圆缺陷分类数据集/images")
    parser.add_argument("--report", default="artifacts/dram_det_v3/annotation_audit.json")
    parser.add_argument("--freeze-out", help="通过完整审计后写入新的冻结标注文件")
    args = parser.parse_args()

    taxonomy = load_taxonomy(args.taxonomy, require_frozen=bool(args.freeze_out))
    payload = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    if args.freeze_out:
        payload, report = freeze_annotations(payload, taxonomy, args.images)
        destination = Path(args.freeze_out)
        if destination.exists():
            raise FileExistsError(f"冻结标注已存在，拒绝覆盖: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        report = audit_annotations(payload, taxonomy, args.images)
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
