import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageStat

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from dram_diag.config import load_yaml
from dram_diag.protocol import build_multilabel_manifest, validate_multilabel_manifest


def main():
    parser = argparse.ArgumentParser(description="审计多标签标注数据并生成冻结协议")
    parser.add_argument("--data-root", default="晶圆缺陷分类数据集")
    parser.add_argument("--protocol", default="configs/protocols/dram_ml_v2.yaml")
    parser.add_argument("--out", default="artifacts/dram_ml_v2")
    parser.add_argument("--labels", default="artifacts/dram_ml_v2/labels.csv", help="多标签 CSV 路径")
    parser.add_argument("--source-labels", default="label_v2.json", help="人工标注源文件，用于冻结哈希")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = load_yaml(args.protocol)
    source_labels = Path(args.source_labels)
    if source_labels.exists():
        protocol["label_file_sha256"] = hashlib.sha256(source_labels.read_bytes()).hexdigest()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], check=True,
                                    capture_output=True, text=True).stdout.strip())
        protocol["code_version"] = f"{commit}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        protocol["code_version"] = "unknown"
    code_digest = hashlib.sha256()
    code_paths = []
    for root in (Path("src"), Path("scripts"), Path("configs")):
        if root.exists():
            code_paths.extend(path for path in root.rglob("*") if path.is_file() and path.suffix in {".py", ".yaml", ".yml"})
    code_paths.extend(path for path in (Path("pyproject.toml"), Path("requirements.txt")) if path.exists())
    for path in sorted(code_paths, key=lambda value: value.as_posix()):
        code_digest.update(path.as_posix().encode("utf-8"))
        code_digest.update(path.read_bytes())
    protocol["code_fingerprint"] = code_digest.hexdigest()
    with Path(args.labels).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = [row["IMAGE_NAME"] for row in rows if not (Path(args.data_root) / "images" / row["IMAGE_NAME"]).exists()]
    if missing:
        raise ValueError(f"labels.csv 中存在缺失图片: {missing[:10]}")
    hashes = defaultdict(list)
    low_quality = []
    image_root = Path(args.data_root) / "images"
    for row in rows:
        path = image_root / row["IMAGE_NAME"]
        hashes[hashlib.sha256(path.read_bytes()).hexdigest()].append(row["IMAGE_NAME"])
        with Image.open(path) as image:
            gray = image.convert("L")
            if min(gray.size) < 32 or ImageStat.Stat(gray).stddev[0] < 2.0:
                low_quality.append(row["IMAGE_NAME"])
    duplicates = [names for names in hashes.values() if len(names) > 1]
    manifest = build_multilabel_manifest(rows, protocol)
    validate_multilabel_manifest(manifest, rows)
    report = {
        "task": "multilabel",
        "protocol_version": manifest["protocol_version"],
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "total_images": manifest["role_stats"]["total_images"],
        "types": manifest["types"],
        "type_counts": manifest["role_stats"]["types"],
        "label_file_sha256": manifest.get("label_file_sha256"),
        "code_version": manifest.get("code_version"),
        "code_fingerprint": manifest.get("code_fingerprint"),
        "label_combinations": manifest["role_stats"]["label_combinations"],
        "cooccurrence_matrix": manifest["role_stats"]["cooccurrence_matrix"],
        "split_samples": manifest["role_stats"]["split_samples"],
        "split_type_counts": manifest["role_stats"]["split_type_counts"],
        "fold_type_counts": [
            {
                "fold": fold["fold"],
                "train": {str(t): sum(t in item["labels"] for item in fold["train"]) for t in manifest["types"]},
                "validation": {str(t): sum(t in item["labels"] for item in fold["validation"]) for t in manifest["types"]},
            }
            for fold in manifest.get("cv_folds", [])
        ],
        "duplicate_image_groups": duplicates,
        "low_quality_images": low_quality,
    }
    (out / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
