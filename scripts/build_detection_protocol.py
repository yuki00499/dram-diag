"""Build the frozen dram-det-v3 test split and five development folds."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dram_diag.detection_data import build_detection_manifest, load_taxonomy, load_yaml


def main():
    parser = argparse.ArgumentParser(description="构建 dram-det-v3 数据协议")
    parser.add_argument("--protocol", default="configs/protocols/dram_det_v3.yaml")
    parser.add_argument("--annotations", help="覆盖协议中的冻结标注路径")
    parser.add_argument("--out", default="artifacts/dram_det_v3/split_manifest.json")
    args = parser.parse_args()
    protocol = load_yaml(args.protocol)
    taxonomy = load_taxonomy(protocol["taxonomy"], require_frozen=True)
    annotations = json.loads(Path(args.annotations or protocol["annotations"]).read_text(encoding="utf-8"))
    manifest = build_detection_manifest(annotations, taxonomy, protocol)
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"manifest 已存在，拒绝覆盖: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "protocol_version": manifest["protocol_version"],
        "development": len(manifest["splits"]["development"]),
        "test_known": len(manifest["splits"]["test_known"]),
        "folds": len(manifest["cv_folds"]),
        "manifest_sha256": manifest["manifest_sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
