from collections import defaultdict
import hashlib
import json
import random


DEPLOYMENT_SPLIT_NAMES = ("train", "validation", "test_known")


def _label_features(items, types, tracked_pairs):
    """Return binary label and selected co-occurrence features for stratification."""
    features = []
    for item in items:
        labels = set(item["labels"])
        features.append(
            [int(value in labels) for value in types]
            + [int(left in labels and right in labels) for left, right in tracked_pairs]
        )
    return features


def _iterative_partition(items, feature_rows, sizes, seed):
    """Deterministic iterative multi-label stratification without an extra dependency."""
    if sum(sizes) != len(items) or any(size < 0 for size in sizes):
        raise ValueError("分层目标容量与样本数不一致")
    rng = random.Random(seed)
    feature_count = len(feature_rows[0]) if feature_rows else 0
    totals = [sum(row[column] for row in feature_rows) for column in range(feature_count)]
    desired = [[total * size / max(1, len(items)) for total in totals] for size in sizes]
    remaining_size = list(sizes)
    partitions = [[] for _ in sizes]
    unassigned = set(range(len(items)))

    while unassigned:
        remaining_counts = [sum(feature_rows[index][column] for index in unassigned)
                            for column in range(feature_count)]
        active = [column for column, count in enumerate(remaining_counts) if count > 0]
        if active:
            rarest = min(active, key=lambda column: (remaining_counts[column], column))
            candidates = [index for index in unassigned if feature_rows[index][rarest]]
        else:
            candidates = list(unassigned)
        rng.shuffle(candidates)
        for index in candidates:
            if index not in unassigned:
                continue
            available = [fold for fold, capacity in enumerate(remaining_size) if capacity > 0]
            if not available:
                raise ValueError("多标签分层分配失败：没有剩余容量")
            if active:
                best_feature = max(desired[fold][rarest] for fold in available)
                available = [fold for fold in available if abs(desired[fold][rarest] - best_feature) < 1e-12]
            best_size = max(remaining_size[fold] for fold in available)
            available = [fold for fold in available if remaining_size[fold] == best_size]
            chosen = rng.choice(available)
            partitions[chosen].append(items[index])
            remaining_size[chosen] -= 1
            for column, present in enumerate(feature_rows[index]):
                if present:
                    desired[chosen][column] -= 1
            unassigned.remove(index)
    return partitions


def _multilabel_counts(items, types):
    return {str(type_id): sum(type_id in item["labels"] for item in items) for type_id in types}


def _combination_counts(items):
    counts = defaultdict(int)
    for item in items:
        counts[",".join(str(value) for value in item["labels"])] += 1
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))


def _cooccurrence_matrix(items, types):
    return [[sum(left in item["labels"] and right in item["labels"] for item in items)
             for right in types] for left in types]


def multilabel_dataset_fingerprint(rows):
    records = sorted((row["IMAGE_NAME"], tuple(sorted(int(value) for value in row["LABELS"].split(",")))) for row in rows)
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_fingerprint(manifest):
    payload = {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    normalized = json.loads(json.dumps(payload, ensure_ascii=False))
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_multilabel_manifest(rows, protocol):
    """构建锁定测试集和迭代多标签分层交叉验证协议。"""
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
    folds = int(protocol.get("cv_folds", 0))
    tracked_pairs = [tuple(sorted(int(value) for value in pair)) for pair in protocol.get("tracked_pairs", [])]
    if folds < 2:
        raise ValueError("dram-ml-v2 至少需要2折交叉验证")
    test_size = max(1, round(len(items) * ratio))
    features = _label_features(items, types, tracked_pairs)
    development, test_known = _iterative_partition(
        items, features, [len(items) - test_size, test_size], seed)
    development_features = _label_features(development, types, tracked_pairs)
    base, extra = divmod(len(development), folds)
    fold_sizes = [base + int(index < extra) for index in range(folds)]
    validation_folds = _iterative_partition(development, development_features, fold_sizes, seed + 1)
    cv_folds = []
    for fold, validation in enumerate(validation_folds):
        validation_names = {item["image_name"] for item in validation}
        train = [item for item in development if item["image_name"] not in validation_names]
        cv_folds.append({"fold": fold, "train": train, "validation": validation})
    splits = {"development": development, "test_known": test_known}
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
        "label_file_sha256": protocol.get("label_file_sha256"),
        "code_version": protocol.get("code_version", "unknown"),
        "code_fingerprint": protocol.get("code_fingerprint"),
        "core_types": sorted(int(value) for value in protocol.get("core_types", types)),
        "rare_types": sorted(int(value) for value in protocol.get("rare_types", [])),
        "tracked_pairs": [list(pair) for pair in tracked_pairs],
        "role_stats": {
            "types": counts,
            "label_combinations": _combination_counts(items),
            "cooccurrence_matrix": _cooccurrence_matrix(items, types),
            "split_samples": split_samples,
            "split_type_counts": {name: _multilabel_counts(values, types) for name, values in splits.items()},
            "total_images": len(items),
        },
        "splits": splits,
        "cv_folds": cv_folds,
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
    if manifest.get("cv_folds"):
        names = [item["image_name"] for split in ("development", "test_known")
                 for item in manifest["splits"].get(split, [])]
        development_names = {item["image_name"] for item in manifest["splits"]["development"]}
        fold_validation = [item["image_name"] for fold in manifest["cv_folds"] for item in fold["validation"]]
        if set(fold_validation) != development_names or len(fold_validation) != len(set(fold_validation)):
            raise ValueError("交叉验证折未恰好覆盖 development")
        for fold in manifest["cv_folds"]:
            train_names = {item["image_name"] for item in fold["train"]}
            validation_names = {item["image_name"] for item in fold["validation"]}
            if train_names & validation_names or train_names | validation_names != development_names:
                raise ValueError(f"fold {fold['fold']} 存在泄漏或覆盖不完整")
    else:
        names = [item["image_name"] for split in DEPLOYMENT_SPLIT_NAMES
                 for item in manifest["splits"].get(split, [])]
    if len(names) != len(set(names)):
        raise ValueError("数据 split 中存在重复图片")
    if len(names) != manifest["role_stats"]["total_images"]:
        raise ValueError("数据 split 未覆盖全部图片")
    return manifest
