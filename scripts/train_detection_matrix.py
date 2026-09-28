"""Preflight and sequentially run the 4 x 5 dram-det-v3 ablation matrix.

Without --execute this command only prints a validated plan. It never uses the
held-out test labels or runs model evaluation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dram_diag.detection_data import (canonical_hash, load_experiment_config,
                                      load_taxonomy, validate_detection_manifest)


VARIANTS = ("B0", "B1", "B2", "B3")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_completed(task: dict) -> bool:
    """A pre-existing run is reusable only when its artifacts match this plan."""
    run_dir = Path(task["run_dir"])
    if not run_dir.exists():
        return False
    provenance_path = run_dir / "v3_provenance.json"
    if not provenance_path.is_file():
        raise RuntimeError(f"已有不完整训练目录，需人工处理: {run_dir}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = {
        "variant": task["variant"],
        "fold": task["fold"],
        "run_name": task["name"],
        "manifest_sha256": task["manifest_sha256"],
        "annotation_sha256": task["annotation_sha256"],
        "taxonomy_sha256": task["taxonomy_sha256"],
        "experiment_config_sha256": task["config_sha256"],
        "resolved_experiment_config_sha256": task["resolved_config_sha256"],
        "training_overrides": {"epochs": task["epochs"]} if task["epochs"] is not None else {},
        "ultralytics_version": "8.4.156",
        "data_yaml_sha256": task["data_sha256"],
        "global_labels_sha256": task["global_labels_sha256"],
        "export_report_sha256": task["export_report_sha256"],
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"已有训练目录与当前协议或配置不符: {run_dir}")
    pretrained = provenance.get("pretrained_weights", {})
    if pretrained.get("sha256") != task["pretrained_sha256"]:
        raise RuntimeError(f"预训练权重哈希不符: {run_dir}")
    for label in ("best", "last"):
        weight = run_dir / "weights" / f"{label}.pt"
        if not weight.is_file() or provenance.get("weights", {}).get(label, {}).get("sha256") != file_hash(weight):
            raise RuntimeError(f"已有训练结果缺少有效 {label}.pt: {run_dir}")
    return True


def build_plan(args) -> list[dict]:
    manifest_path = Path(args.manifest).resolve()
    manifest = validate_detection_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    taxonomy_path = Path(args.taxonomy).resolve()
    taxonomy = load_taxonomy(taxonomy_path, require_frozen=True)
    if manifest["taxonomy_sha256"] != taxonomy["taxonomy_sha256"]:
        raise ValueError("manifest 与冻结 taxonomy 哈希不符")
    if len(args.variants) != len(set(args.variants)) or len(args.folds) != len(set(args.folds)):
        raise ValueError("变体或折号重复")
    available_folds = {int(item["fold"]) for item in manifest["cv_folds"]}
    if not set(args.folds) <= available_folds:
        raise ValueError(f"折号不在冻结 manifest 中: {sorted(set(args.folds) - available_folds)}")

    data_root = Path(args.data_root).resolve()
    config_root = Path(args.config_root).resolve()
    project = Path(args.project).resolve()
    expected_names = {int(item["id"]): item["name_zh"] for item in taxonomy["object_classes"]}
    datasets = {}
    for fold in args.folds:
        fold_dir = data_root / f"fold-{fold}"
        data_path = fold_dir / "data.yaml"
        report_path = fold_dir / "export_report.json"
        global_path = fold_dir / "global_labels.json"
        if any(not path.is_file() for path in (data_path, report_path, global_path)):
            raise FileNotFoundError(f"fold-{fold} 缺少 data.yaml、export_report.json 或 global_labels.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
        if (int(report["fold"]) != fold or report["manifest_sha256"] != manifest["manifest_sha256"]
                or report["taxonomy_sha256"] != taxonomy["taxonomy_sha256"]):
            raise ValueError(f"fold-{fold} 导出报告与冻结协议不符")
        if data["names"] != expected_names or Path(data["path"]).resolve() != fold_dir:
            raise ValueError(f"fold-{fold} data.yaml 类别或根路径不符")
        datasets[fold] = {
            "path": data_path,
            "data_sha256": file_hash(data_path),
            "global_labels_sha256": file_hash(global_path),
            "export_report_sha256": file_hash(report_path),
        }

    configs = {}
    for variant in args.variants:
        config_path = config_root / f"dram_det_v3_{variant.lower()}.yaml"
        config = load_experiment_config(config_path)
        if config.get("variant") != variant:
            raise ValueError(f"变体配置不匹配: {config_path}")
        source = config.get("pretrained_weights") or (
            config.get("model") if str(config.get("model", "")).lower().endswith(".pt") else None)
        if not source or not Path(source).is_file():
            raise FileNotFoundError(f"{variant} 缺少本地预训练权重: {source}")
        configs[variant] = {
            "path": config_path,
            "raw_sha256": file_hash(config_path),
            "resolved_sha256": canonical_hash(config),
            "pretrained_sha256": file_hash(Path(source)),
        }

    tasks = []
    for variant in args.variants:
        for fold in args.folds:
            name = f"dram_det_v3_{manifest['manifest_sha256'][:8]}_{variant.lower()}_fold{fold}"
            if args.epochs is not None:
                name += f"_smoke_e{args.epochs}"
            config = configs[variant]
            tasks.append({
                "variant": variant, "fold": fold, "name": name,
                "epochs": args.epochs,
                "config": str(config["path"]), "data": str(datasets[fold]["path"]),
                "run_dir": str(project / name),
                "data_sha256": datasets[fold]["data_sha256"],
                "global_labels_sha256": datasets[fold]["global_labels_sha256"],
                "export_report_sha256": datasets[fold]["export_report_sha256"],
                "config_sha256": config["raw_sha256"],
                "resolved_config_sha256": config["resolved_sha256"],
                "pretrained_sha256": config["pretrained_sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
                "annotation_sha256": manifest["annotation_sha256"],
                "taxonomy_sha256": taxonomy["taxonomy_sha256"],
                "manifest": str(manifest_path), "taxonomy": str(taxonomy_path),
                "project": str(project),
            })
    return tasks


def run_task(task: dict) -> None:
    project = Path(task["project"])
    log_dir = project / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task['name']}.log"
    command = [sys.executable, "-u", str(ROOT / "scripts" / "train_detection.py"),
               "--config", task["config"], "--data", task["data"],
               "--manifest", task["manifest"], "--taxonomy", task["taxonomy"],
               "--project", task["project"], "--name", task["name"]]
    if task["epochs"] is not None:
        command.extend(("--epochs", str(task["epochs"])))
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with log_path.open("x", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return_code = process.wait()
    if return_code:
        raise RuntimeError(f"训练失败 ({return_code})，查看日志: {log_path}")
    if not verify_completed(task):
        raise RuntimeError(f"训练进程结束，但缺少可验证产物: {task['run_dir']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="顺序执行 dram-det-v3 的 B0–B3 × 5 折")
    parser.add_argument("--manifest", default="artifacts/dram_det_v3/split_manifest_r2.json")
    parser.add_argument("--data-root", default="artifacts/dram_det_v3/yolo/r2")
    parser.add_argument("--config-root", default="configs/experiments")
    parser.add_argument("--taxonomy", default="configs/taxonomy_v3.yaml")
    parser.add_argument("--project", default="artifacts/dram_det_v3/runs")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--epochs", type=int, help="单独命名的短程冒烟；留空时按配置完成正式训练")
    parser.add_argument("--execute", action="store_true", help="显式启动训练；默认只预检并打印计划")
    parser.add_argument("--skip-completed", action="store_true", help="只跳过哈希和 best/last 均验证通过的任务")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs 必须大于 0")

    tasks = build_plan(args)
    completed = []
    for task in tasks:
        run_dir = Path(task["run_dir"])
        if run_dir.exists():
            if not args.skip_completed:
                raise FileExistsError(f"训练目录已存在，拒绝覆盖: {run_dir}；可用 --skip-completed 验证后跳过")
            if verify_completed(task):
                completed.append(task["name"])
    pending = [task for task in tasks if task["name"] not in completed]
    print(json.dumps({"mode": "execute" if args.execute else "preflight_only",
                      "total": len(tasks), "completed": len(completed), "pending": len(pending),
                      "manifest_sha256": tasks[0]["manifest_sha256"] if tasks else None,
                      "tasks": [{key: task[key] for key in ("variant", "fold", "name", "data", "run_dir")}
                                for task in pending]}, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return
    for index, task in enumerate(pending, 1):
        print(f"[{index}/{len(pending)}] 启动 {task['variant']} fold-{task['fold']}", flush=True)
        run_task(task)
    print(f"训练矩阵完成：{len(completed) + len(pending)}/{len(tasks)} 个任务", flush=True)


if __name__ == "__main__":
    main()
