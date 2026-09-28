"""Train one frozen dram-det-v3 ablation variant."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dram_diag.detection_data import (canonical_hash, load_experiment_config,
                                      load_taxonomy, validate_detection_manifest)
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
    parser.add_argument("--manifest", help="冻结划分清单；批量训练时必须提供")
    parser.add_argument("--taxonomy", default="configs/taxonomy_v3.yaml")
    parser.add_argument("--project", default="artifacts/dram_det_v3/runs")
    parser.add_argument("--name")
    parser.add_argument("--epochs", type=int, help="仅用于独立命名的冒烟任务；正式训练使用配置值")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != "8.4.156":
        raise RuntimeError(f"需要 ultralytics==8.4.156，当前为 {ultralytics.__version__}")
    config_path, data_path = Path(args.config).resolve(), Path(args.data).resolve()
    config = load_experiment_config(config_path)
    config_hash = file_hash(config_path)
    resolved_config_hash = canonical_hash(config)
    taxonomy = load_taxonomy(args.taxonomy, require_frozen=True)
    if config.get("variant") not in {"B0", "B1", "B2", "B3"}:
        raise ValueError("实验 variant 必须为 B0/B1/B2/B3")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs 必须大于 0")
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    global_labels = data_path.parent / "global_labels.json"
    export_report = data_path.parent / "export_report.json"
    if not global_labels.exists() or not export_report.exists():
        raise FileNotFoundError("缺少同目录 global_labels.json 或 export_report.json")
    data_hash = file_hash(data_path)
    global_labels_hash = file_hash(global_labels)
    export_report_hash = file_hash(export_report)
    report = json.loads(export_report.read_text(encoding="utf-8"))
    if report["taxonomy_sha256"] != taxonomy["taxonomy_sha256"]:
        raise ValueError("训练数据与冻结 taxonomy 哈希不一致")
    manifest = None
    if args.manifest:
        manifest = validate_detection_manifest(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
        if (report["manifest_sha256"] != manifest["manifest_sha256"]
                or manifest["taxonomy_sha256"] != taxonomy["taxonomy_sha256"]):
            raise ValueError("训练数据、划分清单与 taxonomy 哈希不一致")

    name = args.name or config["experiment"]
    project = Path(args.project).resolve()
    expected_dir = project / name
    if expected_dir.exists() and not args.resume:
        raise FileExistsError(f"训练目录已存在，拒绝自动改名或覆盖: {expected_dir}")
    resume_checkpoint = expected_dir / "weights" / "last.pt"
    if args.resume and not resume_checkpoint.is_file():
        raise FileNotFoundError(f"无法续训，缺少 last.pt: {resume_checkpoint}")
    model_source = config["model"]
    pretrained_source = config.get("pretrained_weights") or (
        model_source if str(model_source).lower().endswith(".pt") else None)
    if not pretrained_source or not Path(pretrained_source).is_file():
        raise FileNotFoundError(f"预训练权重必须预先下载到本地: {pretrained_source}")
    pretrained_path = Path(pretrained_source).resolve()
    pretrained_hash = file_hash(pretrained_path)

    overrides = {key: config[key] for key in TRAIN_KEYS if key in config}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    overrides.update({"data": str(data_path), "project": str(project), "name": name,
                      "exist_ok": False, "resume": str(resume_checkpoint) if args.resume else False})
    if config.get("pretrained_weights"):
        overrides["pretrained"] = str(pretrained_path)

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
        model = YOLO(str(pretrained_path) if str(model_source).lower().endswith(".pt") else model_source)
        if config.get("pretrained_weights"):
            model.load(str(pretrained_path))
        model.train(**overrides)
        save_dir = Path(model.trainer.save_dir)

    if save_dir.resolve() != expected_dir:
        raise RuntimeError(f"实际训练目录不是预期目录: {save_dir} != {expected_dir}")
    if (file_hash(config_path) != config_hash or file_hash(pretrained_path) != pretrained_hash
            or file_hash(data_path) != data_hash or file_hash(global_labels) != global_labels_hash
            or file_hash(export_report) != export_report_hash):
        raise RuntimeError("训练期间配置、预训练权重或导出数据发生变化，拒绝写入有效 provenance")
    weights = {}
    for label in ("best", "last"):
        path = save_dir / "weights" / f"{label}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"训练结束但缺少 {label}.pt: {path}")
        weights[label] = {"path": str(path.resolve()), "sha256": file_hash(path)}
    provenance = {
        "protocol_version": "dram-det-v3",
        "variant": config["variant"],
        "fold": int(report["fold"]),
        "run_name": name,
        "data_yaml": str(data_path),
        "data_yaml_sha256": data_hash,
        "global_labels_sha256": global_labels_hash,
        "export_report_sha256": export_report_hash,
        "experiment_config": str(config_path.resolve()),
        "experiment_config_sha256": config_hash,
        "resolved_experiment_config_sha256": resolved_config_hash,
        "training_overrides": {"epochs": args.epochs} if args.epochs is not None else {},
        "taxonomy_sha256": taxonomy["taxonomy_sha256"],
        "manifest_sha256": report["manifest_sha256"],
        "annotation_sha256": manifest["annotation_sha256"] if manifest else None,
        "pretrained_weights": {"path": str(pretrained_path), "sha256": pretrained_hash},
        "ultralytics_version": ultralytics.__version__,
        "upstream_license": "AGPL-3.0-or-later",
        "weights": weights,
    }
    with (save_dir / "v3_provenance.json").open("x", encoding="utf-8") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
