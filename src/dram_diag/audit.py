from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import load_yaml
from .data import load_labels
from .protocol import build_manifest


def _average_hash(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((8, 8)), dtype=np.float32)
    return "".join("1" if value >= pixels.mean() else "0" for value in pixels.ravel())


def audit_dataset(data_root, protocol_path, out_dir):
    root = Path(data_root)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_labels(root)
    image_dir = root / "images"
    files = {path.name: path for path in image_dir.glob("*.jpg")}
    missing = [row["IMAGE_NAME"] for row in rows if row["IMAGE_NAME"] not in files]
    extra = sorted(set(files) - {row["IMAGE_NAME"] for row in rows})
    invalid = []
    dimensions = Counter()
    exact_hashes = {}
    perceptual_hashes = {}
    for name, path in files.items():
        try:
            with Image.open(path) as image:
                dimensions[f"{image.width}x{image.height}/{image.mode}"] += 1
                image.verify()
            exact_hashes.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(), []).append(name)
            perceptual_hashes.setdefault(_average_hash(path), []).append(name)
        except Exception as exc:
            invalid.append({"image_name": name, "error": str(exc)})
    if missing or invalid:
        raise ValueError(f"数据审计失败: missing={len(missing)}, invalid={len(invalid)}")

    manifest = build_manifest(rows, load_yaml(protocol_path))
    report = {
        "rows": len(rows),
        "image_files": len(files),
        "missing": missing,
        "extra": extra,
        "invalid": invalid,
        "dimensions": dict(dimensions),
        "exact_duplicate_groups": [items for items in exact_hashes.values() if len(items) > 1],
        "perceptual_hash_groups": [items for items in perceptual_hashes.values() if len(items) > 1],
        "protocol_version": manifest["protocol_version"],
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "role_stats": manifest["role_stats"],
        "unknown_class_counts": {
            str(key): manifest["counts"][key]
            for key in (
                manifest["calibration_unknown_classes"]
                + manifest["test_unknown_core_classes"]
                + manifest["test_unknown_stress_classes"]
            )
        },
    }
    (output / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, manifest
