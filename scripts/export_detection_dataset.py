"""Export one frozen dram-det-v3 fold to an Ultralytics YOLO dataset."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dram_diag.detection_data import export_yolo_fold, load_taxonomy, load_yaml


def main():
    parser = argparse.ArgumentParser(description="导出 dram-det-v3 YOLO 派生数据")
    parser.add_argument("--protocol", default="configs/protocols/dram_det_v3.yaml")
    parser.add_argument("--manifest", default="artifacts/dram_det_v3/split_manifest.json")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    protocol = load_yaml(args.protocol)
    taxonomy = load_taxonomy(protocol["taxonomy"], require_frozen=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    destination = Path(args.out or f"artifacts/dram_det_v3/yolo/fold-{args.fold}")
    if destination.exists():
        raise FileExistsError(f"导出目录已存在，拒绝覆盖: {destination}")
    report = export_yolo_fold(
        manifest, taxonomy, Path(protocol["data_root"]) / "images", destination, args.fold)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
