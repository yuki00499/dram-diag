import json
from pathlib import Path

from PIL import Image

from dram_diag.detection_data import (annotation_fingerprint, audit_annotations,
                                      build_detection_manifest, canonical_hash,
                                      export_yolo_fold, select_blind_review,
                                      validate_detection_manifest)


def taxonomy():
    value = {
        "schema_version": 3,
        "status": "frozen",
        "object_classes": [
            {"id": 0, "slug": "defect-a", "name_zh": "A", "name_en": "A",
             "definition": "A", "boundary_rule": "tight", "positive_examples": [],
             "negative_examples": [], "confusable_with": []},
            {"id": 1, "slug": "defect-b", "name_zh": "B", "name_en": "B",
             "definition": "B", "boundary_rule": "tight", "positive_examples": [],
             "negative_examples": [], "confusable_with": []},
        ],
        "quality_attributes": [{"id": "blur", "name_zh": "模糊", "name_en": "Blur", "definition": "blur"}],
        "usability_states": ["usable", "review", "unusable"],
    }
    value["taxonomy_sha256"] = canonical_hash(value)
    return value


def frozen_payload(count=30):
    tax = taxonomy()
    images = {}
    for index in range(count):
        objects = [] if index % 3 == 0 else [{
            "instance_id": f"i{index}", "class_id": index % 2,
            "bbox_xyxy": [10, 20, 40 + index % 5, 60],
        }]
        images[f"image_{index}.jpg"] = {
            "width": 480, "height": 320, "objects": objects,
            "quality_attributes": ["blur"] if index % 7 == 0 else [],
            "usability": "usable", "ignore_regions": [],
            "review": {"status": "reviewed", "note": ""},
        }
    payload = {"schema_version": 3, "status": "frozen", "taxonomy_sha256": tax["taxonomy_sha256"],
               "images": images, "change_log": []}
    payload["annotation_sha256"] = annotation_fingerprint(payload)
    return tax, payload


def test_detection_manifest_has_new_isolated_folds():
    tax, payload = frozen_payload()
    protocol = {"protocol_version": "dram-det-v3", "holdout_ratio": .1, "cv_folds": 5, "split_seed": 99}
    manifest = build_detection_manifest(payload, tax, protocol)
    validate_detection_manifest(manifest)
    assert len(manifest["splits"]["test_known"]) == 3
    assert len(manifest["splits"]["development"]) == 27
    assert manifest["split_seed"] != 42


def test_audit_warns_duplicate_class_but_rejects_out_of_bounds(tmp_path):
    tax, payload = frozen_payload(1)
    root = tmp_path / "images"; root.mkdir()
    Image.new("L", (480, 320)).save(root / "image_0.jpg")
    payload["images"]["image_0.jpg"]["objects"] = [
        {"instance_id": "a", "class_id": 0, "bbox_xyxy": [1, 1, 10, 10]},
        {"instance_id": "b", "class_id": 0, "bbox_xyxy": [20, 20, 500, 40]},
    ]
    report = audit_annotations(payload, tax, root, require_complete=True)
    assert not report["valid"]
    assert any("超出" in item["message"] for item in report["errors"])
    assert any("同类别" in item["message"] for item in report["warnings"])


def test_yolo_export_keeps_true_negative_and_excludes_ignore_regions(tmp_path):
    tax, payload = frozen_payload(20)
    # One otherwise valid item is global-only because ambiguous regions must not become background.
    payload["images"]["image_1.jpg"]["ignore_regions"] = [
        {"bbox_xyxy": [100, 100, 150, 150], "reason": "ambiguous", "candidate_class_ids": [0]}]
    payload["annotation_sha256"] = annotation_fingerprint(payload)
    manifest = build_detection_manifest(payload, tax, {
        "protocol_version": "dram-det-v3", "holdout_ratio": .1, "cv_folds": 5, "split_seed": 7})
    image_root = tmp_path / "source"; image_root.mkdir()
    for name in payload["images"]:
        Image.new("L", (480, 320)).save(image_root / name)
    out = tmp_path / "yolo"
    report = export_yolo_fold(manifest, tax, image_root, out, 0)
    assert (out / "data.yaml").exists()
    assert any(item["image_name"] == "image_1.jpg" for item in report["excluded"])
    negative_labels = [path for path in (out / "labels").rglob("*.txt") if not path.read_text()]
    assert negative_labels
    global_labels = json.loads((out / "global_labels.json").read_text(encoding="utf-8"))
    assert "image_1.jpg" in global_labels
    assert (out / "images" / "global" / global_labels["image_1.jpg"]["split"] / "image_1.jpg").exists()
    assert sum(report["global_exported"].values()) == 20


def test_blind_review_is_stratified_and_includes_rare_ambiguous_cases():
    tax, payload = frozen_payload(30)
    payload["images"]["image_0.jpg"]["objects"] = [{
        "instance_id": "rare", "class_id": 1, "bbox_xyxy": [1, 1, 8, 8]}]
    # Make class 1 rare by moving every other object to class 0.
    for name, record in payload["images"].items():
        if name != "image_0.jpg":
            for obj in record["objects"]:
                obj["class_id"] = 0
    payload["images"]["image_2.jpg"]["ignore_regions"] = [{
        "bbox_xyxy": [100, 100, 120, 120], "reason": "uncertain", "candidate_class_ids": [0]}]
    result = select_blind_review(payload, tax, ratio=.2, seed=3)
    assert len(result["images"]) >= 6
    assert "image_0.jpg" in result["images"]
    assert "rare_class" in result["reasons"]["image_0.jpg"]
    assert "image_2.jpg" in result["images"]
    assert result == select_blind_review(payload, tax, ratio=.2, seed=3)


def test_audit_rejects_object_ignore_overlap():
    tax, payload = frozen_payload(1)
    payload["images"]["image_0.jpg"]["objects"] = [{
        "instance_id": "x", "class_id": 0, "bbox_xyxy": [10, 10, 30, 30]}]
    payload["images"]["image_0.jpg"]["ignore_regions"] = [{
        "bbox_xyxy": [20, 20, 40, 40], "reason": "uncertain", "candidate_class_ids": [0]}]
    report = audit_annotations(payload, tax)
    assert any("重叠" in item["message"] for item in report["errors"])
