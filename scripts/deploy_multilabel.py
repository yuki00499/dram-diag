"""为多标签模型生成部署清单和相似案例索引。"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from dram_diag.compat import file_sha256, validate_checkpoint
from dram_diag.inference import ModelRunner
from dram_diag.protocol import validate_multilabel_manifest

TYPE_NAMES = {
    1: "圆形颗粒/异物",
    2: "细长颗粒/异物",
    3: "方形颗粒/异物",
    4: "划痕/裂纹",
    5: "凹坑/空洞",
    7: "块状/块斑类",
    8: "背景/低信号",
}

TYPE_NAMES_FULL = {
    1: "Round Particle（圆形颗粒/异物）",
    2: "Elongated Particle（细长颗粒/异物）",
    3: "Polygonal Particle（方形颗粒/异物）",
    4: "Scratch / Crack（划痕/裂纹）",
    5: "Pit / Void（凹坑/空洞）",
    7: "Blob（块状/块斑类）",
    8: "Low-signal（背景/低信号）",
}


def main():
    parser = argparse.ArgumentParser(description="生成多标签部署清单")
    parser.add_argument("--checkpoint", nargs="+")
    parser.add_argument("--selection", help="模型选择清单；自动读取5折 checkpoint 和 OOF 阈值")
    parser.add_argument("--manifest", default="artifacts/dram_ml_v2/split_manifest.json")
    parser.add_argument("--data-root", default="晶圆缺陷分类数据集")
    parser.add_argument("--out", default="artifacts/deployment-v2.json")
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--thresholds", help="OOF evaluation JSON；存在时使用逐标签阈值")
    parser.add_argument("--model-version", default="dram-ml-v2-ensemble")
    args = parser.parse_args()

    selection_lock = None
    if args.selection:
        selection_lock = json.loads(Path(args.selection).read_text(encoding="utf-8"))["lock"]
        args.checkpoint = [item["path"] for item in selection_lock["checkpoints"]]
        args.model_version = selection_lock["model_version"]
    if not args.checkpoint:
        parser.error("必须提供 --checkpoint 或 --selection")

    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"部署清单已存在: {destination}")

    manifest = validate_multilabel_manifest(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    if selection_lock and selection_lock["manifest_fingerprint"] != manifest["manifest_fingerprint"]:
        raise ValueError("模型选择清单与 manifest 不一致")
    checkpoint_paths = [Path(value) for value in args.checkpoint]
    for checkpoint_path in checkpoint_paths:
        checkpoint = __import__("torch").load(checkpoint_path, map_location="cpu", weights_only=False)
        validate_checkpoint(checkpoint, manifest)

    runner = ModelRunner([str(path) for path in checkpoint_paths], manifest, args.data_root)
    retrieval_items = manifest["splits"].get("development", manifest["splits"].get("train", []))
    _, train_embeddings = runner.infer_items(retrieval_items)

    retrieval_dir = destination.with_suffix(".multilabel")
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    records = retrieval_items
    np.savez_compressed(retrieval_dir / "retrieval.npz", embeddings=train_embeddings)
    (retrieval_dir / "retrieval.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")

    deployment = {
        "model_version": args.model_version,
        "mode": "multilabel_classification",
        "ready": True,
        "task": "multilabel",
        "protocol_version": manifest["protocol_version"],
        "manifest": str(Path(args.manifest)),
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "checkpoints": [{"path": str(path), "sha256": file_sha256(path)} for path in checkpoint_paths],
        "types": manifest["types"],
        "type_names": {str(key): TYPE_NAMES[key] for key in manifest["types"] if key in TYPE_NAMES},
        "type_names_full": {str(key): TYPE_NAMES_FULL[key] for key in manifest["types"] if key in TYPE_NAMES_FULL},
        "threshold": args.threshold,
        "thresholds": (selection_lock["thresholds"] if selection_lock else
                       json.loads(Path(args.thresholds).read_text(encoding="utf-8"))["thresholds"]
                       if args.thresholds else {str(value): args.threshold for value in manifest["types"]}),
        "review_margin": 0.1,
        "oof_evaluation": (selection_lock["oof_evaluation"]["path"] if selection_lock else args.thresholds),
        "retrieval_embeddings": str(retrieval_dir / "retrieval.npz"),
        "retrieval_records": str(retrieval_dir / "retrieval.json"),
    }
    destination.write_text(json.dumps(deployment, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(deployment, ensure_ascii=False, indent=2))
    print(f"\n部署清单已生成: {destination}")


if __name__ == "__main__":
    main()
