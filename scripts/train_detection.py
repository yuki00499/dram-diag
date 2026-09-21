"""Train one frozen dram-det-v3 ablation variant."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dram_diag.detection_data import load_experiment_config, load_taxonomy
from dram_diag.detection_training import create_hierarchical_trainer


TRAIN_KEYS = {
    "imgsz", "epochs", "patience", "batch", "device", "amp", "seed", "deterministic", "workers",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "perspective", "fliplr", "flipud",
    "mosaic", "mixup", "cutmix", "erasing",
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="训练 dram-det-v3 B0/B1/B2/B3")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, help="export_detection_dataset.py 生成的 data.yaml")
    parser.add_argument("--taxonomy", default="configs/taxonomy_v3.yaml")
    parser.add_argument("--project", default="artifacts/dram_det_v3/runs")
    parser.add_argument("--name")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != "8.4.156":
        raise RuntimeError(f"需要 ultralytics==8.4.156，当前为 {ultralytics.__version__}")
    config_path, data_path = Path(args.config), Path(args.data)
    config = load_experiment_config(config_path)
    taxonomy = load_taxonomy(args.taxonomy, require_frozen=True)
    if config.get("variant") not in {"B0", "B1", "B2", "B3"}:
        raise ValueError("实验 variant 必须为 B0/B1/B2/B3")
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    global_labels = data_path.parent / "global_labels.json"
    export_report = data_path.parent / "export_report.json"
    if not global_labels.exists() or not export_report.exists():
        raise FileNotFoundError("缺少同目录 global_labels.json 或 export_report.json")
    report = json.loads(export_report.read_text(encoding="utf-8"))
    if report["taxonomy_sha256"] != taxonomy["taxonomy_sha256"]:
        raise ValueError("训练数据与冻结 taxonomy 哈希不一致")

    name = args.name or config["experiment"]
    overrides = {key: config[key] for key in TRAIN_KEYS if key in config}
    overrides.update({"data": str(data_path), "project": args.project, "name": name, "resume": args.resume})
    model_source = config["model"]
    if config.get("pretrained_weights"):
        overrides["pretrained"] = config["pretrained_weights"]

    if config.get("hierarchical_head"):
        trainer = create_hierarchical_trainer(
            hierarchy_config=config,
            global_labels_path=global_labels,
            class_ids=[int(item["id"]) for item in taxonomy["object_classes"]],
            quality_ids=[str(item["id"]) for item in taxonomy.get("quality_attributes", [])],
            overrides={"model": model_source, **overrides},
        )
        trainer.train()
        save_dir = Path(trainer.save_dir)
    else:
        model = YOLO(model_source)
        if config.get("pretrained_weights"):
            model.load(config["pretrained_weights"])
        result = model.train(**overrides)
        save_dir = Path(result.save_dir)

    weights = {}
    for label in ("best", "last"):
        path = save_dir / "weights" / f"{label}.pt"
        if path.exists():
            weights[label] = {"path": str(path.resolve()), "sha256": file_hash(path)}
    provenance = {
        "protocol_version": "dram-det-v3",
        "variant": config["variant"],
        "experiment_config": str(config_path.resolve()),
        "experiment_config_sha256": file_hash(config_path),
        "taxonomy_sha256": taxonomy["taxonomy_sha256"],
        "manifest_sha256": report["manifest_sha256"],
        "ultralytics_version": ultralytics.__version__,
        "upstream_license": "AGPL-3.0-or-later",
        "weights": weights,
    }
    pretrained_source = config.get("pretrained_weights") or (
        model_source if str(model_source).lower().endswith(".pt") else None)
    if pretrained_source and Path(pretrained_source).exists():
        provenance["pretrained_weights"] = {
            "path": str(Path(pretrained_source).resolve()),
            "sha256": file_hash(Path(pretrained_source)),
        }
    (save_dir / "v3_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
