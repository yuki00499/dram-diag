"""Bulk-accept the current v3 annotations with an explicit provenance record.

This is an owner decision, not evidence that an independent blind review occurred.
The command preserves the exact original bytes before replacing the draft source.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dram_diag.detection_data import annotation_fingerprint, audit_annotations, load_taxonomy


def main() -> None:
    parser = argparse.ArgumentParser(description="批量确认 v3 标注复核状态")
    parser.add_argument("--annotations", default="annotations/dram_det_v3.json")
    parser.add_argument("--taxonomy", default="configs/taxonomy_v3.yaml")
    parser.add_argument("--images", default="晶圆缺陷分类数据集/images")
    parser.add_argument("--backup", required=True, help="不可覆盖的原始标注备份路径")
    parser.add_argument("--expected-source-hash", required=True)
    parser.add_argument("--expected-images", type=int, default=1150)
    args = parser.parse_args()

    source = Path(args.annotations)
    backup = Path(args.backup)
    temporary = source.with_name(source.name + ".reviewed.tmp")
    if backup.exists() or temporary.exists():
        raise FileExistsError("备份或临时文件已存在，拒绝覆盖")
    if source.resolve() == backup.resolve():
        raise ValueError("备份路径不能与源标注相同")

    original_bytes = source.read_bytes()
    payload = json.loads(original_bytes.decode("utf-8"))
    original_hash = annotation_fingerprint(payload)
    if original_hash != args.expected_source_hash:
        raise ValueError(f"源标注哈希不符: {original_hash}")
    if payload.get("status") != "draft" or len(payload.get("images", {})) != args.expected_images:
        raise ValueError("只允许处理预期数量的 draft 标注")

    taxonomy = load_taxonomy(args.taxonomy, require_frozen=True)
    previous_taxonomy_hash = payload["taxonomy_sha256"]
    previous_states = Counter()
    changed = 0
    for row in payload["images"].values():
        review = row.setdefault("review", {})
        previous = review.get("status", "unreviewed")
        previous_states[previous] += 1
        if previous != "reviewed":
            changed += 1
        review["status"] = "reviewed"
    payload["taxonomy_sha256"] = taxonomy["taxonomy_sha256"]
    payload.setdefault("change_log", []).append({
        "action": "bulk_owner_accept_review_status",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "changed_images": changed,
        "previous_review_support": dict(sorted(previous_states.items())),
        "source_annotation_sha256": original_hash,
        "source_taxonomy_sha256": previous_taxonomy_hash,
        "taxonomy_sha256": taxonomy["taxonomy_sha256"],
        "independent_blind_review_performed": False,
        "note": "项目所有者批量确认当前标注；不能视作第二人盲复核。类别 4 和 5 保持独立。",
    })
    report = audit_annotations(payload, taxonomy, args.images, require_complete=True)
    if not report["valid"]:
        raise ValueError(f"最终审计失败，拒绝修改源标注: {report['errors'][:3]}")

    backup.parent.mkdir(parents=True, exist_ok=True)
    with backup.open("xb") as handle:
        handle.write(original_bytes)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, source)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({
        "changed_images": changed,
        "previous_review_support": dict(sorted(previous_states.items())),
        "source_annotation_sha256": original_hash,
        "reviewed_annotation_sha256": report["annotation_sha256"],
        "taxonomy_sha256": taxonomy["taxonomy_sha256"],
        "audit_errors": len(report["errors"]),
        "audit_warnings": len(report["warnings"]),
        "backup": str(backup),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
