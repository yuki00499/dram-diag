"""Create a hash-locked dram-det-v3 deployment manifest."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dram_diag.detection_data import load_taxonomy, validate_detection_manifest
from dram_diag.detection_inference import sha256_file


def main():
    parser = argparse.ArgumentParser(description="生成 dram-det-v3 部署清单")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="artifacts/dram_det_v3/split_manifest.json")
    parser.add_argument("--taxonomy", default="configs/taxonomy_v3.yaml")
    parser.add_argument("--evaluation")
    parser.add_argument("--variant", choices=["B0", "B1", "B2", "B3"], required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--out", default="artifacts/deployment-v3.json")
    args = parser.parse_args()
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"部署清单已存在，拒绝覆盖: {destination}")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    taxonomy = load_taxonomy(args.taxonomy, require_frozen=True)
    manifest = validate_detection_manifest(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    if manifest["taxonomy_sha256"] != taxonomy["taxonomy_sha256"]:
        raise ValueError("manifest 与 taxonomy 不匹配")
    deployment = {
        "ready": True,
        "task": "hierarchical_detection",
        "model_version": args.model_version,
        "protocol_version": "dram-det-v3",
        "variant": args.variant,
        "manifest": os.path.relpath(Path(args.manifest).resolve(), destination.parent.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "taxonomy": os.path.relpath(Path(args.taxonomy).resolve(), destination.parent.resolve()),
        "taxonomy_sha256": taxonomy["taxonomy_sha256"],
        "checkpoint": {"path": os.path.relpath(checkpoint.resolve(), destination.parent.resolve()),
                       "sha256": sha256_file(checkpoint)},
        "class_ids": [int(item["id"]) for item in taxonomy["object_classes"]],
        "class_names": {str(item["id"]): item["name_zh"] for item in taxonomy["object_classes"]},
        "quality_attributes": [str(item["id"]) for item in taxonomy.get("quality_attributes", [])],
        "hierarchical_head": args.variant in {"B2", "B3"},
        "p2_head": args.variant in {"B1", "B3"},
        "imgsz": 640,
        "confidence": .25,
        "review_policy": {
            "global_threshold": .5,
            "detection_threshold": .25,
            "disagreement_threshold": .35,
            "usability_review_threshold": .5,
        },
        "evaluation": (os.path.relpath(Path(args.evaluation).resolve(), destination.parent.resolve())
                       if args.evaluation else None),
        "upstream": {"package": "ultralytics", "version": "8.4.156", "license": "AGPL-3.0-or-later"},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(deployment, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(deployment, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
