"""Read-only spot check of one frozen dram-det-v3 YOLO fold.

The held-out test split is checked by filename only; its labels are not inspected.
"""

import argparse
import json
from pathlib import Path
import random
import sys

import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dram_diag.detection_data import annotation_fingerprint, load_taxonomy, validate_detection_manifest


def names_in(directory: Path, suffix: str) -> set[str]:
    return {path.name for path in directory.iterdir() if path.is_file() and path.suffix.lower() == suffix}


def check_box(line: str, obj: dict, width: int, height: int) -> None:
    parts = line.split()
    if len(parts) != 5 or int(parts[0]) != int(obj["class_id"]):
        raise ValueError(f"类别或字段数不匹配: {line}")
    actual = [float(value) for value in parts[1:]]
    x1, y1, x2, y2 = (float(value) for value in obj["bbox_xyxy"])
    expected = [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                (x2 - x1) / width, (y2 - y1) / height]
    if any(not 0 <= value <= 1 for value in actual):
        raise ValueError(f"归一化框越界: {line}")
    if any(abs(left - right) > 5.1e-9 for left, right in zip(actual, expected)):
        raise ValueError(f"像素框与归一化框不匹配: {line} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description="抽查 YOLO 导出与划分隔离")
    parser.add_argument("--manifest", default="artifacts/dram_det_v3/split_manifest.json")
    parser.add_argument("--taxonomy", default="configs/taxonomy_v3.yaml")
    parser.add_argument("--annotations", default="annotations/dram_det_v3.frozen.json")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--root", help="默认 artifacts/dram_det_v3/yolo/fold-<fold>")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--sample-size", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.root or f"artifacts/dram_det_v3/yolo/fold-{args.fold}")
    manifest = validate_detection_manifest(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    taxonomy = load_taxonomy(args.taxonomy, require_frozen=True)
    annotations = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    if annotations["annotation_sha256"] != annotation_fingerprint(annotations):
        raise ValueError("冻结标注自身哈希不匹配")
    if (manifest["annotation_sha256"] != annotations["annotation_sha256"]
            or manifest["taxonomy_sha256"] != taxonomy["taxonomy_sha256"]):
        raise ValueError("划分清单与冻结标注或类别定义不匹配")
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    expected_names = {int(item["id"]): item["name_zh"] for item in taxonomy["object_classes"]}
    if data["names"] != expected_names:
        raise ValueError(f"类别名称或编号不匹配: {data['names']}")
    if Path(data["path"]).resolve() != root.resolve():
        raise ValueError("data.yaml 指向了其他导出目录")

    fold = next(item for item in manifest["cv_folds"] if int(item["fold"]) == args.fold)
    splits = {"train": fold["train"], "val": fold["validation"],
              "test": manifest["splits"]["test_known"]}
    expected_sets = {split: {item["image_name"] for item in items} for split, items in splits.items()}
    if any(expected_sets[left] & expected_sets[right]
           for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("划分清单中 train/val/test 有重叠")

    counts = {}
    examples = []
    rng = random.Random(args.seed)
    for split, items in splits.items():
        global_names = names_in(root / "images" / "global" / split, ".jpg")
        detector_names = names_in(root / "images" / split, ".jpg")
        eligible = {item["image_name"] for item in items
                    if item.get("usability") == "usable" and not item.get("ignore_regions")}
        label_names = names_in(root / "labels" / split, ".txt")
        expected_labels = {f"{Path(name).stem}.txt" for name in eligible}
        if global_names != expected_sets[split] or detector_names != eligible or label_names != expected_labels:
            raise ValueError(f"{split} 导出图片、标签文件或排除掩码不匹配")
        counts[split] = {"global_images": len(global_names), "detector_images": len(detector_names),
                         "excluded": len(global_names - detector_names), "label_files": len(label_names)}
        # Do not inspect held-out test labels before the model and thresholds are locked.
        if split == "test":
            continue
        eligible_items = [item for item in items if item["image_name"] in eligible]
        negatives = []
        positives = []
        box_count = 0
        for item in eligible_items:
            source_record = annotations["images"][item["image_name"]]
            if any(item.get(key) != source_record.get(key)
                   for key in ("width", "height", "objects", "usability", "ignore_regions")):
                raise ValueError(f"manifest 与冻结源标注不一致: {item['image_name']}")
            label = root / "labels" / split / f"{Path(item['image_name']).stem}.txt"
            lines = label.read_text(encoding="utf-8").splitlines()
            objects = item.get("objects", [])
            if len(lines) != len(objects):
                raise ValueError(f"{label} 的框数与冻结标注不符")
            if objects:
                positives.append((item, lines))
            else:
                negatives.append(item)
                if label.stat().st_size != 0:
                    raise ValueError(f"真实负样本标签文件不是空文件: {label}")
            for line, obj in zip(lines, objects):
                check_box(line, obj, int(item["width"]), int(item["height"]))
                box_count += 1
        counts[split]["checked_boxes"] = box_count
        counts[split]["empty_labels"] = len(negatives)
        for item, lines in rng.sample(positives, min(args.sample_size, len(positives))):
            examples.append({"split": split, "image": item["image_name"],
                             "pixel_bbox": item["objects"][0]["bbox_xyxy"],
                             "yolo_label": lines[0]})
        if negatives:
            item = rng.choice(negatives)
            examples.append({"split": split, "image": item["image_name"], "empty_label": True})

    actual_sets = {split: names_in(root / "images" / split, ".jpg") for split in splits}
    if any(actual_sets[left] & actual_sets[right]
           for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("实际导出图片在 train/val/test 间重叠")
    print(json.dumps({"passed": True, "fold": args.fold, "seed": args.seed,
                      "manifest_sha256": manifest["manifest_sha256"],
                      "classes": expected_names, "counts": counts,
                      "random_examples": examples,
                      "test_labels_inspected": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
