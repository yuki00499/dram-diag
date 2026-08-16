from collections import defaultdict
import hashlib
import json
import random


SPLIT_NAMES = (
    "train",
    "validation",
    "calibration_known",
    "test_known",
    "calibration_unknown",
    "test_unknown_core",
    "test_unknown_stress",
    "case_library",
)

MULTILABEL_SPLIT_NAMES = ("train", "validation", "test_known")


def dataset_fingerprint(rows):
    records = sorted((row["IMAGE_NAME"], int(row["DEFECT_ID"])) for row in rows)
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def multilabel_dataset_fingerprint(rows):
    records = sorted((row["IMAGE_NAME"], tuple(sorted(int(value) for value in row["LABELS"].split(",")))) for row in rows)
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_fingerprint(manifest):
    payload = {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    normalized = json.loads(json.dumps(payload, ensure_ascii=False))
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _held_out_count(size, ratio):
    held_out = max(1, round(size * ratio))
    while 3 * held_out >= size:
        held_out -= 1
    if held_out < 1:
        raise ValueError(f"类别样本数 {size} 无法按留出比例 {ratio} 划分")
    return held_out


def _build_cv_folds(development_items, folds, seed):
    by_class = defaultdict(list)
    for item in development_items:
        by_class[int(item["defect_id"])].append(item)
    result = [[] for _ in range(folds)]
    for defect_id in sorted(by_class):
        items = list(by_class[defect_id])
        random.Random(seed + defect_id).shuffle(items)
        for index, item in enumerate(items):
            result[index % folds].append(item)
    return [{"fold": index, "validation": items} for index, items in enumerate(result)]


def validate_manifest(manifest, rows=None):
    if manifest.get("manifest_fingerprint") != manifest_fingerprint(manifest):
        raise ValueError("manifest 指纹不匹配，文件可能已被修改")
    if rows is not None and manifest["dataset_fingerprint"] != dataset_fingerprint(rows):
        raise ValueError("manifest 与当前 label.csv 不匹配")

    roles = [
        set(manifest["classification_classes"]),
        set(manifest["case_library_classes"]),
        set(manifest["calibration_unknown_classes"]),
        set(manifest["test_unknown_core_classes"]),
        set(manifest["test_unknown_stress_classes"]),
    ]
    for index, left in enumerate(roles):
        for right in roles[index + 1:]:
            if left & right:
                raise ValueError(f"类别角色重叠: {sorted(left & right)}")
    expected_classes = {int(value) for value in manifest["counts"]}
    if set().union(*roles) != expected_classes:
        raise ValueError("类别角色未覆盖全部类别")

    names = [item["image_name"] for split in manifest["splits"].values() for item in split]
    if len(names) != len(set(names)):
        raise ValueError("数据 split 中存在重复图片")
    if len(names) != sum(int(value) for value in manifest["counts"].values()):
        raise ValueError("数据 split 未覆盖全部图片")

    development_names = {
        item["image_name"]
        for split in ("train", "validation")
        for item in manifest["splits"][split]
    }
    fold_names = [
        item["image_name"]
        for fold in manifest["cv_folds"]
        for item in fold["validation"]
    ]
    if set(fold_names) != development_names or len(fold_names) != len(set(fold_names)):
        raise ValueError("5折索引未恰好覆盖 development pool")
    return manifest


def build_manifest(rows, protocol):
    seed = int(protocol.get("split_seed", 42))
    threshold = int(protocol["min_class_count"])
    ratio = float(protocol.get("holdout_ratio", .1))
    folds = int(protocol.get("cv_folds", 5))
    unknown = protocol["unknown_classes"]
    calibration_unknown = [int(value) for value in unknown["calibration"]]
    test_unknown_core = [int(value) for value in unknown["test_core"]]
    test_unknown_stress = [int(value) for value in unknown["test_stress"]]
    all_unknown = set(calibration_unknown + test_unknown_core + test_unknown_stress)

    by_class = defaultdict(list)
    for row in rows:
        by_class[int(row["DEFECT_ID"])].append(row["IMAGE_NAME"])
    counts = {key: len(value) for key, value in by_class.items()}
    missing_unknown = all_unknown - set(counts)
    if missing_unknown:
        raise ValueError(f"未知类不存在: {sorted(missing_unknown)}")
    min_unknown = int(protocol.get("min_core_unknown_count", 8))
    if any(counts[key] < min_unknown for key in test_unknown_core):
        raise ValueError("核心测试未知类存在样本数不足的类别")

    known = sorted(set(counts) - all_unknown)
    classification_classes = [key for key in known if counts[key] >= threshold]
    case_library_classes = [key for key in known if counts[key] < threshold]
    splits = {name: [] for name in SPLIT_NAMES}
    rng = random.Random(seed)

    for defect_id in sorted(by_class):
        images = list(by_class[defect_id])
        rng.shuffle(images)
        items = [{"image_name": name, "defect_id": defect_id} for name in images]
        if defect_id in calibration_unknown:
            splits["calibration_unknown"].extend(items)
        elif defect_id in test_unknown_core:
            splits["test_unknown_core"].extend(items)
        elif defect_id in test_unknown_stress:
            splits["test_unknown_stress"].extend(items)
        elif defect_id in case_library_classes:
            splits["case_library"].extend({**item, "review_required": True} for item in items)
        else:
            held_out = _held_out_count(len(items), ratio)
            train_end = len(items) - 3 * held_out
            validation_end = train_end + held_out
            calibration_end = validation_end + held_out
            splits["train"].extend(items[:train_end])
            splits["validation"].extend(items[train_end:validation_end])
            splits["calibration_known"].extend(items[validation_end:calibration_end])
            splits["test_known"].extend(items[calibration_end:])

    development = splits["train"] + splits["validation"]
    cv_folds = _build_cv_folds(development, folds, seed)
    role_stats = {
        "classification": {"classes": len(classification_classes), "samples": sum(counts[key] for key in classification_classes)},
        "case_library": {"classes": len(case_library_classes), "samples": sum(counts[key] for key in case_library_classes)},
        "unknown": {
            "calibration": {"classes": len(calibration_unknown), "samples": sum(counts[key] for key in calibration_unknown)},
            "test_core": {"classes": len(test_unknown_core), "samples": sum(counts[key] for key in test_unknown_core)},
            "test_stress": {"classes": len(test_unknown_stress), "samples": sum(counts[key] for key in test_unknown_stress)},
        },
        "split_samples": {name: len(items) for name, items in splits.items()},
    }
    manifest = {
        "protocol_version": protocol["protocol_version"],
        "min_class_count": threshold,
        "holdout_ratio": ratio,
        "split_seed": seed,
        "dataset_fingerprint": dataset_fingerprint(rows),
        "counts": counts,
        "classification_classes": classification_classes,
        "case_library_classes": case_library_classes,
        "calibration_unknown_classes": calibration_unknown,
        "test_unknown_core_classes": test_unknown_core,
        "test_unknown_stress_classes": test_unknown_stress,
        "role_stats": role_stats,
        "splits": splits,
        "cv_folds": cv_folds,
    }
    manifest["manifest_fingerprint"] = manifest_fingerprint(manifest)
    return validate_manifest(manifest, rows)


def build_multilabel_manifest(rows, protocol):
    """多标签协议：按图片随机划分 train/validation/test_known，无类别互斥。

    rows: [{"IMAGE_NAME": ..., "LABELS": "1,2"}]（labels.csv 的 DictReader 行）。
    """
    seed = int(protocol.get("split_seed", 42))
    ratio = float(protocol.get("holdout_ratio", .1))
    types = sorted(int(value) for value in protocol["types"])
    items = []
    for row in rows:
        labels = sorted(set(int(value) for value in row["LABELS"].split(",")))
        missing = set(labels) - set(types)
        if missing:
            raise ValueError(f"标签不在类型清单内: {sorted(missing)}")
        items.append({"image_name": row["IMAGE_NAME"], "labels": labels})
    names = [item["image_name"] for item in items]
    if len(names) != len(set(names)):
        raise ValueError("存在重复图片")
    rng = random.Random(seed)
    rng.shuffle(items)
    held = max(1, round(len(items) * ratio))
    splits = {
        "train": items[:-2 * held] if len(items) > 3 * held else items[:len(items) // 2],
        "validation": items[len(items) - 2 * held:len(items) - held] if len(items) > 3 * held else items[len(items) // 2:3 * len(items) // 4],
        "test_known": items[len(items) - held:] if len(items) > 3 * held else items[3 * len(items) // 4:],
    }
    counts = {str(tid): 0 for tid in types}
    for item in items:
        for label in item["labels"]:
            counts[str(label)] += 1
    split_samples = {name: len(value) for name, value in splits.items()}
    manifest = {
        "protocol_version": protocol["protocol_version"],
        "task": "multilabel",
        "types": types,
        "holdout_ratio": ratio,
        "split_seed": seed,
        "dataset_fingerprint": multilabel_dataset_fingerprint(rows),
        "role_stats": {"types": counts, "split_samples": split_samples, "total_images": len(items)},
        "splits": splits,
    }
    manifest["manifest_fingerprint"] = manifest_fingerprint(manifest)
    return validate_multilabel_manifest(manifest, rows)


def validate_multilabel_manifest(manifest, rows=None):
    if manifest.get("task") != "multilabel":
        raise ValueError("manifest 不是多标签协议")
    if manifest.get("manifest_fingerprint") != manifest_fingerprint(manifest):
        raise ValueError("manifest 指纹不匹配，文件可能已被修改")
    if rows is not None and manifest["dataset_fingerprint"] != multilabel_dataset_fingerprint(rows):
        raise ValueError("manifest 与当前 labels.csv 不匹配")
    names = [item["image_name"] for split in MULTILABEL_SPLIT_NAMES for item in manifest["splits"].get(split, [])]
    if len(names) != len(set(names)):
        raise ValueError("数据 split 中存在重复图片")
    if len(names) != manifest["role_stats"]["total_images"]:
        raise ValueError("数据 split 未覆盖全部图片")
    return manifest
