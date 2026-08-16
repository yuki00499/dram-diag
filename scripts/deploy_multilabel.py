"""为多标签模型生成部署清单（含检索索引），供 serve.py 加载。

用法:
    python scripts/deploy_multilabel.py --checkpoint runs/ge20_ml_baseline/seed-42/best.pt
"""

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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="artifacts/ge20_ml/split_manifest.json")
    parser.add_argument("--data-root", default="晶圆缺陷分类数据集")
    parser.add_argument("--out", default="artifacts/deployment.json")
    parser.add_argument("--threshold", type=float, default=.5)
    args = parser.parse_args()

    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"部署清单已存在: {destination}")

    manifest = validate_multilabel_manifest(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    checkpoint_path = Path(args.checkpoint)
    checkpoint = __import__("torch").load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint, manifest)

    runner = ModelRunner([str(checkpoint_path)], manifest, args.data_root)
    _, train_embeddings = runner.infer_items(manifest["splits"]["train"])

    prototypes = []
    for type_id in manifest["types"]:
        mask = np.asarray([type_id in item["labels"] for item in manifest["splits"]["train"]])
        if mask.any():
            center = train_embeddings[mask].mean(0)
            prototypes.append(center / np.linalg.norm(center))
        else:
            prototypes.append(np.zeros(train_embeddings.shape[1]))
    prototypes = np.asarray(prototypes)

    retrieval_dir = destination.with_suffix(".multilabel")
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    records = manifest["splits"]["train"]
    np.savez_compressed(retrieval_dir / "retrieval.npz", embeddings=train_embeddings)
    (retrieval_dir / "retrieval.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")

    deployment = {
        "model_version": "ge20-ml-baseline-v1",
        "mode": "multilabel_classification",
        "ready": True,
        "task": "multilabel",
        "protocol_version": manifest["protocol_version"],
        "manifest": str(Path(args.manifest)),
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "checkpoints": [{"path": str(checkpoint_path), "sha256": file_sha256(checkpoint_path)}],
        "types": manifest["types"],
        "type_names": {str(key): TYPE_NAMES[key] for key in manifest["types"] if key in TYPE_NAMES},
        "type_names_full": {str(key): TYPE_NAMES_FULL[key] for key in manifest["types"] if key in TYPE_NAMES_FULL},
        "threshold": args.threshold,
        "unknown_rejection_enabled": False,
        "unknown_alpha": 0.5,
        "unknown_threshold": 1.0,
        "temperature": 1.0,
        "retrieval_embeddings": str(retrieval_dir / "retrieval.npz"),
        "retrieval_records": str(retrieval_dir / "retrieval.json"),
    }
    destination.write_text(json.dumps(deployment, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(deployment, ensure_ascii=False, indent=2))
    print(f"\n部署清单已生成: {destination}")


if __name__ == "__main__":
    main()
