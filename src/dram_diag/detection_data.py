"""Versioned annotation, audit, split, and export support for dram-det-v3."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

from PIL import Image
import yaml

from .protocol import _iterative_partition


SCHEMA_VERSION = 3
USABILITY_STATES = ("usable", "review", "unusable")
REVIEW_STATES = ("unreviewed", "primary_complete", "reviewed", "disputed")


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load_experiment_config(path: str | Path) -> dict:
    """Load a v3 experiment config with a single recursively resolved `_base_`."""
    path = Path(path)
    config = load_yaml(path)
    base = config.pop("_base_", None)
    if not base:
        return config
    base_path = Path(base)
    if not base_path.is_absolute():
        candidates = (path.parent / base_path, Path.cwd() / base_path)
        base_path = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
    resolved = load_experiment_config(base_path)
    resolved.update(config)
    return resolved


def load_taxonomy(path: str | Path, require_frozen: bool = False) -> dict:
    taxonomy = load_yaml(path)
    errors = validate_taxonomy(taxonomy, require_frozen=require_frozen)
    if errors:
        raise ValueError("taxonomy_v3 无效: " + "; ".join(errors))
    taxonomy = dict(taxonomy)
    taxonomy["taxonomy_sha256"] = canonical_hash(
        {key: value for key, value in taxonomy.items() if key != "taxonomy_sha256"}
    )
    return taxonomy


def validate_taxonomy(taxonomy: dict, require_frozen: bool = False) -> list[str]:
    errors: list[str] = []
    if int(taxonomy.get("schema_version", -1)) != SCHEMA_VERSION:
        errors.append(f"schema_version 必须为 {SCHEMA_VERSION}")
    status = taxonomy.get("status")
    if status not in {"draft", "frozen"}:
        errors.append("status 必须为 draft 或 frozen")
    if require_frozen and status != "frozen":
        errors.append("正式导出前 taxonomy status 必须为 frozen")

    classes = taxonomy.get("object_classes", [])
    class_ids = [item.get("id") for item in classes]
    if len(class_ids) != len(set(class_ids)):
        errors.append("缺陷类别 ID 重复")
    if class_ids and class_ids != list(range(len(class_ids))):
        errors.append("缺陷类别 ID 必须从 0 开始连续递增")
    required = ("id", "slug", "name_zh", "name_en", "definition", "boundary_rule")
    for index, item in enumerate(classes):
        missing = [key for key in required if item.get(key) in (None, "")]
        if missing:
            errors.append(f"object_classes[{index}] 缺少: {','.join(missing)}")
        if require_frozen:
            for key in ("positive_examples", "negative_examples", "confusable_with"):
                if key not in item:
                    errors.append(f"object_classes[{index}] 冻结前缺少 {key}")

    qualities = taxonomy.get("quality_attributes", [])
    quality_ids = [item.get("id") for item in qualities]
    if len(quality_ids) != len(set(quality_ids)):
        errors.append("质量属性 ID 重复")
    if taxonomy.get("usability_states") != list(USABILITY_STATES):
        errors.append(f"usability_states 必须为 {list(USABILITY_STATES)}")
    if require_frozen and not classes:
        errors.append("冻结 taxonomy 至少需要一个缺陷类别")
    return errors


def blank_annotation_payload(taxonomy: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "draft",
        "taxonomy_sha256": taxonomy["taxonomy_sha256"],
        "images": {},
        "change_log": [],
    }


def annotation_fingerprint(payload: dict) -> str:
    normalized = {key: value for key, value in payload.items() if key != "annotation_sha256"}
    return canonical_hash(normalized)


def _valid_box(box: Any, width: int, height: int) -> tuple[bool, str]:
    if not isinstance(box, list) or len(box) != 4:
        return False, "bbox_xyxy 必须是4个数字"
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError):
        return False, "bbox_xyxy 包含非数字"
    if x2 <= x1 or y2 <= y1:
        return False, "bbox_xyxy 宽高必须大于0"
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return False, f"bbox_xyxy 超出 {width}x{height} 图像边界"
    return True, ""


def _box_intersection(left: list[float], right: list[float]) -> float:
    return max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1]))


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    ordered = sorted(values)
    pick = lambda ratio: ordered[round((len(ordered) - 1) * ratio)]
    return {"min": ordered[0], "p25": pick(.25), "median": pick(.5), "p75": pick(.75),
            "max": ordered[-1], "mean": sum(ordered) / len(ordered)}


def audit_annotations(payload: dict, taxonomy: dict, image_root: str | Path | None = None,
                      require_complete: bool = False) -> dict:
    errors: list[dict] = []
    warnings: list[dict] = []
    classes = {int(item["id"]): item for item in taxonomy.get("object_classes", [])}
    quality_ids = {str(item["id"]) for item in taxonomy.get("quality_attributes", [])}
    images = payload.get("images", {})
    image_root = Path(image_root) if image_root else None
    class_support: Counter = Counter()
    quality_support: Counter = Counter()
    usability_support: Counter = Counter()
    object_areas: list[float] = []
    object_widths: list[float] = []
    object_heights: list[float] = []
    small_objects = 0
    review_support: Counter = Counter()

    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        errors.append({"scope": "root", "message": f"schema_version 必须为 {SCHEMA_VERSION}"})
    if payload.get("taxonomy_sha256") != taxonomy.get("taxonomy_sha256"):
        errors.append({"scope": "root", "message": "标注与 taxonomy 哈希不一致"})

    disk_names: set[str] = set()
    if image_root and image_root.exists():
        disk_names = {path.name for path in image_root.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}}
        for name in sorted(disk_names - set(images)):
            target = errors if require_complete else warnings
            target.append({"image_name": name, "message": "图像未标注"})
        for name in sorted(set(images) - disk_names):
            errors.append({"image_name": name, "message": "标注对应图像不存在"})

    for name, record in sorted(images.items()):
        width = int(record.get("width", 0))
        height = int(record.get("height", 0))
        if image_root and (image_root / name).exists():
            with Image.open(image_root / name) as image:
                actual = image.size
            if width and height and actual != (width, height):
                errors.append({"image_name": name, "message": f"记录尺寸 {width}x{height} 与图像 {actual[0]}x{actual[1]} 不符"})
            width, height = actual
        if width <= 0 or height <= 0:
            errors.append({"image_name": name, "message": "缺少有效图像尺寸"})
            continue

        usability = record.get("usability")
        if usability not in USABILITY_STATES:
            errors.append({"image_name": name, "message": f"无效 usability: {usability}"})
        else:
            usability_support[usability] += 1
        review_state = record.get("review", {}).get("status", "unreviewed")
        if review_state not in REVIEW_STATES:
            errors.append({"image_name": name, "message": f"无效 review.status: {review_state}"})
        review_support[review_state] += 1

        qualities = record.get("quality_attributes", [])
        unknown_quality = sorted(set(qualities) - quality_ids)
        if unknown_quality:
            errors.append({"image_name": name, "message": f"未定义质量属性: {unknown_quality}"})
        quality_support.update(qualities)

        instance_ids: set[str] = set()
        per_class: Counter = Counter()
        for obj in record.get("objects", []):
            instance_id = str(obj.get("instance_id", ""))
            if not instance_id or instance_id in instance_ids:
                errors.append({"image_name": name, "message": f"缺少或重复 instance_id: {instance_id}"})
            instance_ids.add(instance_id)
            class_id = obj.get("class_id")
            if class_id not in classes:
                errors.append({"image_name": name, "message": f"未定义缺陷类别: {class_id}"})
            ok, message = _valid_box(obj.get("bbox_xyxy"), width, height)
            if not ok:
                errors.append({"image_name": name, "instance_id": instance_id, "message": message})
            else:
                x1, y1, x2, y2 = map(float, obj["bbox_xyxy"])
                object_widths.append(x2 - x1)
                object_heights.append(y2 - y1)
                area = (x2 - x1) * (y2 - y1)
                object_areas.append(area)
                scale = 640 / max(width, height)
                small_objects += int(area * scale * scale < 32 * 32)
            if class_id in classes:
                class_support[class_id] += 1
                per_class[class_id] += 1
        for class_id, count in per_class.items():
            if count > 1:
                warnings.append({"image_name": name, "class_id": class_id,
                                 "message": f"同类别出现 {count} 个实例，请复核"})

        for index, region in enumerate(record.get("ignore_regions", [])):
            ok, message = _valid_box(region.get("bbox_xyxy"), width, height)
            if not ok:
                errors.append({"image_name": name, "ignore_region": index, "message": message})
            candidates = region.get("candidate_class_ids", [])
            unknown = sorted(set(candidates) - set(classes))
            if unknown:
                errors.append({"image_name": name, "ignore_region": index,
                               "message": f"争议区域候选类别未定义: {unknown}"})
            if ok:
                conflict_ids = [str(obj.get("instance_id")) for obj in record.get("objects", [])
                                if _valid_box(obj.get("bbox_xyxy"), width, height)[0]
                                and _box_intersection(region["bbox_xyxy"], obj["bbox_xyxy"]) > 0]
                if conflict_ids:
                    errors.append({"image_name": name, "ignore_region": index,
                                   "message": f"忽略区域与已标注实例重叠: {conflict_ids}"})

        secondary = record.get("review", {}).get("secondary_objects", [])
        secondary_per_class: Counter = Counter()
        for obj in secondary:
            instance_id = str(obj.get("instance_id", ""))
            if not instance_id or instance_id in instance_ids:
                errors.append({"image_name": name, "message": f"盲复核缺少或重复 instance_id: {instance_id}"})
            instance_ids.add(instance_id)
            class_id = obj.get("class_id")
            if class_id not in classes:
                errors.append({"image_name": name, "message": f"盲复核使用未定义缺陷类别: {class_id}"})
            ok, message = _valid_box(obj.get("bbox_xyxy"), width, height)
            if not ok:
                errors.append({"image_name": name, "instance_id": instance_id,
                               "message": "盲复核 " + message})
            if class_id in classes:
                secondary_per_class[class_id] += 1
        for class_id, count in secondary_per_class.items():
            if count > 1:
                warnings.append({"image_name": name, "class_id": class_id,
                                 "message": f"盲复核同类别出现 {count} 个实例，请复核"})
        for index, region in enumerate(record.get("review", {}).get("secondary_ignore_regions", [])):
            ok, message = _valid_box(region.get("bbox_xyxy"), width, height)
            if not ok:
                errors.append({"image_name": name, "secondary_ignore_region": index,
                               "message": "盲复核 " + message})
            unknown = sorted(set(region.get("candidate_class_ids", [])) - set(classes))
            if unknown:
                errors.append({"image_name": name, "secondary_ignore_region": index,
                               "message": f"盲复核候选类别未定义: {unknown}"})

    stats = {
        "image_count": len(images),
        "disk_image_count": len(disk_names) if image_root else None,
        "object_count": int(sum(class_support.values())),
        "class_support": {str(key): int(value) for key, value in sorted(class_support.items())},
        "quality_support": dict(sorted(quality_support.items())),
        "usability_support": dict(sorted(usability_support.items())),
        "review_support": dict(sorted(review_support.items())),
        "object_width": _quantiles(object_widths),
        "object_height": _quantiles(object_heights),
        "object_area": _quantiles(object_areas),
        "small_objects_at_640": small_objects,
        "small_object_ratio_at_640": small_objects / len(object_areas) if object_areas else None,
    }
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
        "annotation_sha256": annotation_fingerprint(payload),
        "taxonomy_sha256": taxonomy.get("taxonomy_sha256"),
    }


def freeze_annotations(payload: dict, taxonomy: dict, image_root: str | Path) -> tuple[dict, dict]:
    if taxonomy.get("status") != "frozen":
        raise ValueError("标注冻结前必须先冻结 taxonomy")
    report = audit_annotations(payload, taxonomy, image_root, require_complete=True)
    if not report["valid"]:
        raise ValueError(f"标注审计未通过，共 {len(report['errors'])} 个错误")
    frozen = json.loads(json.dumps(payload, ensure_ascii=False))
    frozen["source_annotation_sha256"] = annotation_fingerprint(payload)
    frozen["status"] = "frozen"
    frozen["taxonomy_sha256"] = taxonomy["taxonomy_sha256"]
    frozen.setdefault("change_log", []).append({
        "action": "freeze",
        "source_annotation_sha256": frozen["source_annotation_sha256"],
        "taxonomy_sha256": taxonomy["taxonomy_sha256"],
    })
    frozen["annotation_sha256"] = annotation_fingerprint(frozen)
    return frozen, report


def _record_features(record: dict, class_ids: list[int], quality_ids: list[str]) -> list[int]:
    present = {int(item["class_id"]) for item in record.get("objects", [])}
    qualities = set(record.get("quality_attributes", []))
    count = len(record.get("objects", []))
    return (
        [int(class_id in present) for class_id in class_ids]
        + [int(quality_id in qualities) for quality_id in quality_ids]
        + [int(record.get("usability") == state) for state in USABILITY_STATES]
        + [int(count >= 1), int(count >= 2), int(count >= 3)]
    )


def select_blind_review(payload: dict, taxonomy: dict, ratio: float = .2,
                        seed: int = 20260921) -> dict:
    """Select the stratified blind-review set plus all risk-mandatory images."""
    if not 0 < ratio <= 1:
        raise ValueError("review ratio 必须位于 (0,1]")
    records = [{"image_name": name, **record} for name, record in sorted(payload.get("images", {}).items())]
    if not records:
        return {"images": [], "reasons": {}, "ratio": ratio, "seed": seed}
    incomplete = [record["image_name"] for record in records
                  if record.get("review", {}).get("status") not in {"primary_complete", "reviewed", "disputed"}]
    if incomplete:
        raise ValueError(f"生成盲复核集前需完成全量主标，尚有 {len(incomplete)} 张未完成")
    class_ids = [int(item["id"]) for item in taxonomy.get("object_classes", [])]
    quality_ids = [str(item["id"]) for item in taxonomy.get("quality_attributes", [])]
    support = Counter(int(obj["class_id"]) for record in records for obj in record.get("objects", []))
    rare_limit = max(5, math.ceil(len(records) * .02))
    rare_ids = {class_id for class_id, count in support.items() if count <= rare_limit}
    reasons: dict[str, list[str]] = {}
    for record in records:
        current = []
        if any(int(obj["class_id"]) in rare_ids for obj in record.get("objects", [])):
            current.append("rare_class")
        if record.get("ignore_regions"):
            current.append("ambiguous_region")
        per_class = Counter(int(obj["class_id"]) for obj in record.get("objects", []))
        if any(count > 1 for count in per_class.values()):
            current.append("duplicate_same_class")
        if record.get("review", {}).get("status") == "disputed":
            current.append("disputed")
        if current:
            reasons[record["image_name"]] = current
    mandatory = set(reasons)
    target = max(round(len(records) * ratio), len(mandatory))
    candidates = [record for record in records if record["image_name"] not in mandatory]
    needed = min(target - len(mandatory), len(candidates))
    sampled: list[dict] = []
    if needed:
        features = [_record_features(record, class_ids, quality_ids) for record in candidates]
        _, sampled = _iterative_partition(
            candidates, features, [len(candidates) - needed, needed], int(seed))
        for record in sampled:
            reasons[record["image_name"]] = ["stratified_20_percent"]
    selected = sorted(mandatory | {record["image_name"] for record in sampled})
    return {"images": selected, "reasons": {name: reasons[name] for name in selected},
            "ratio": ratio, "seed": int(seed), "target_count": target,
            "rare_class_ids": sorted(rare_ids), "selection_sha256": canonical_hash(selected)}


def build_detection_manifest(payload: dict, taxonomy: dict, protocol: dict) -> dict:
    if payload.get("status") != "frozen":
        raise ValueError("只能使用冻结标注生成 manifest")
    if payload.get("annotation_sha256") != annotation_fingerprint(
            {key: value for key, value in payload.items() if key != "annotation_sha256"}):
        raise ValueError("标注哈希不匹配")
    class_ids = [int(item["id"]) for item in taxonomy["object_classes"]]
    quality_ids = [str(item["id"]) for item in taxonomy.get("quality_attributes", [])]
    items = [{"image_name": name, **record} for name, record in sorted(payload["images"].items())]
    features = [_record_features(item, class_ids, quality_ids) for item in items]
    test_size = max(1, round(len(items) * float(protocol.get("holdout_ratio", .1))))
    seed = int(protocol.get("split_seed", 20260920))
    development, test_known = _iterative_partition(
        items, features, [len(items) - test_size, test_size], seed)
    folds = int(protocol.get("cv_folds", 5))
    base, extra = divmod(len(development), folds)
    fold_sizes = [base + int(index < extra) for index in range(folds)]
    development_features = [_record_features(item, class_ids, quality_ids) for item in development]
    validation_folds = _iterative_partition(development, development_features, fold_sizes, seed + 1)
    cv_folds = []
    for fold, validation in enumerate(validation_folds):
        validation_names = {item["image_name"] for item in validation}
        train = [item for item in development if item["image_name"] not in validation_names]
        cv_folds.append({"fold": fold, "train": train, "validation": validation})
    manifest = {
        "protocol_version": protocol["protocol_version"],
        "task": "hierarchical_detection",
        "schema_version": SCHEMA_VERSION,
        "split_seed": seed,
        "holdout_ratio": float(protocol.get("holdout_ratio", .1)),
        "taxonomy_sha256": taxonomy["taxonomy_sha256"],
        "annotation_sha256": payload["annotation_sha256"],
        "class_ids": class_ids,
        "quality_ids": quality_ids,
        "splits": {"development": development, "test_known": test_known},
        "cv_folds": cv_folds,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    return validate_detection_manifest(manifest)


def validate_detection_manifest(manifest: dict) -> dict:
    expected = canonical_hash({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    if manifest.get("manifest_sha256") != expected:
        raise ValueError("dram-det-v3 manifest 哈希不匹配")
    development = {item["image_name"] for item in manifest["splits"]["development"]}
    test_known = {item["image_name"] for item in manifest["splits"]["test_known"]}
    if development & test_known:
        raise ValueError("development 与 test_known 存在泄漏")
    fold_validation = [item["image_name"] for fold in manifest["cv_folds"] for item in fold["validation"]]
    if set(fold_validation) != development or len(fold_validation) != len(set(fold_validation)):
        raise ValueError("CV validation 未恰好覆盖 development 一次")
    for fold in manifest["cv_folds"]:
        train = {item["image_name"] for item in fold["train"]}
        validation = {item["image_name"] for item in fold["validation"]}
        if train & validation or train | validation != development:
            raise ValueError(f"fold {fold['fold']} 存在泄漏或覆盖不完整")
    return manifest


def _yolo_line(obj: dict, width: int, height: int) -> str:
    x1, y1, x2, y2 = map(float, obj["bbox_xyxy"])
    xc, yc = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    bw, bh = (x2 - x1) / width, (y2 - y1) / height
    return f"{int(obj['class_id'])} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}"


def export_yolo_fold(manifest: dict, taxonomy: dict, image_root: str | Path,
                     destination: str | Path, fold: int) -> dict:
    """Materialize one fold. Ambiguous/unusable images stay in global labels only."""
    validate_detection_manifest(manifest)
    image_root, destination = Path(image_root), Path(destination)
    selected = next((item for item in manifest["cv_folds"] if int(item["fold"]) == int(fold)), None)
    if selected is None:
        raise ValueError(f"不存在 fold {fold}")
    split_items = {
        "train": selected["train"],
        "val": selected["validation"],
        "test": manifest["splits"]["test_known"],
    }
    excluded: list[dict] = []
    global_records: dict[str, dict] = {}
    exported = Counter()
    for split, items in split_items.items():
        (destination / "images" / split).mkdir(parents=True, exist_ok=True)
        (destination / "images" / "global" / split).mkdir(parents=True, exist_ok=True)
        (destination / "labels" / split).mkdir(parents=True, exist_ok=True)
        for item in items:
            name = item["image_name"]
            global_records[name] = {"split": split, **item}
            source = image_root / name
            if not source.exists():
                raise FileNotFoundError(source)
            # The global stream contains every image. In particular, review/unusable
            # samples supervise only quality/usability and never become detector
            # background examples.
            shutil.copy2(source, destination / "images" / "global" / split / name)
            if item.get("usability") != "usable" or item.get("ignore_regions"):
                excluded.append({"image_name": name, "split": split,
                                 "reason": "not_usable" if item.get("usability") != "usable" else "ignore_region"})
                continue
            shutil.copy2(source, destination / "images" / split / name)
            width, height = int(item["width"]), int(item["height"])
            lines = [_yolo_line(obj, width, height) for obj in item.get("objects", [])]
            (destination / "labels" / split / f"{Path(name).stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            exported[split] += 1
    names = {int(item["id"]): str(item.get("name_zh") or item.get("name_en"))
             for item in taxonomy["object_classes"]}
    dataset = {
        "path": str(destination.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    (destination / "data.yaml").write_text(
        yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (destination / "global_labels.json").write_text(
        json.dumps(global_records, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"fold": int(fold), "exported": dict(exported),
              "global_exported": {key: len(value) for key, value in split_items.items()},
              "excluded": excluded,
              "manifest_sha256": manifest["manifest_sha256"],
              "taxonomy_sha256": taxonomy["taxonomy_sha256"]}
    (destination / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
