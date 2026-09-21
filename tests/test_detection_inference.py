import json

from dram_diag.detection_inference import DetectionRuntime


class FakePredictor:
    def predict(self, path):
        return {
            "width": 480, "height": 320,
            "detections": [{"bbox_xyxy": [10, 20, 50, 80],
                            "bbox_xyxy_normalized": [10/480, 20/320, 50/480, 80/320],
                            "class_id": 0, "confidence": .8}],
            "global_defect_probabilities": {0: .1, 1: .9},
            "quality_probabilities": {"blur": .7},
            "usability_probabilities": {"usable": .1, "review": .8, "unusable": .1},
            "hierarchical_outputs_available": True,
        }


def test_detection_runtime_returns_versioned_explainable_payload(tmp_path):
    checkpoint = tmp_path / "model.pt"; checkpoint.write_bytes(b"weights")
    import hashlib
    deployment = {
        "ready": True, "model_version": "test-b3", "protocol_version": "dram-det-v3",
        "variant": "B3", "taxonomy_sha256": "abc", "class_ids": [0, 1],
        "class_names": {"0": "A", "1": "B"}, "quality_attributes": ["blur"],
        "checkpoint": {"path": "model.pt", "sha256": hashlib.sha256(b"weights").hexdigest()},
        "hierarchical_head": True, "p2_head": True,
    }
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(deployment), encoding="utf-8")
    runtime = DetectionRuntime(path, FakePredictor())
    result = runtime.diagnose_path(tmp_path / "image.jpg", "image.jpg")
    assert result["detections"][0]["class_name"] == "A"
    assert result["review_required"]
    assert result["taxonomy_sha256"] == "abc"
    assert runtime.review_queue[0]["image_name"] == "image.jpg"
