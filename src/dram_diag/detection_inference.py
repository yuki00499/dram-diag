"""Inference adapter and stable API payload for dram-det-v3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .detection import ReviewPolicy, post_nms_presence, review_decision


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Predictor(Protocol):
    def predict(self, path: str | Path) -> dict: ...


class UltralyticsPredictor:
    """Production adapter. A full exported hierarchical model may replace it later."""

    def __init__(self, weights: str | Path, class_ids: list[int], imgsz: int = 640,
                 confidence: float = .25, device: str | int | None = None,
                 quality_attributes: list[str] | None = None, hierarchical_head: bool = False):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("dram-det-v3 推理需要 ultralytics==8.4.156") from exc
        if hierarchical_head:
            # Register the custom checkpoint class before torch deserializes it.
            from .detection_training import build_hierarchical_types
            build_hierarchical_types()
        self.model = YOLO(str(weights))
        self.class_ids = [int(value) for value in class_ids]
        self.quality_attributes = list(quality_attributes or [])
        self.hierarchical_head = bool(hierarchical_head)
        self.imgsz = int(imgsz)
        self.confidence = float(confidence)
        self.device = device

    def predict(self, path: str | Path) -> dict:
        results = self.model.predict(
            source=str(path), imgsz=self.imgsz, conf=self.confidence,
            device=self.device, verbose=False)
        result = results[0]
        height, width = result.orig_shape
        detections = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            confidences = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, confidence, class_id in zip(xyxy, confidences, classes):
                x1, y1, x2, y2 = (float(value) for value in box)
                detections.append({
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_xyxy_normalized": [x1 / width, y1 / height, x2 / width, y2 / height],
                    "class_id": int(class_id),
                    "confidence": float(confidence),
                })
        presence = post_nms_presence(detections, self.class_ids)
        global_presence = presence
        quality = {}
        usability = {"usable": 1.0, "review": 0.0, "unusable": 0.0}
        hierarchical_available = False
        raw_model = getattr(self.model, "model", None)
        outputs = getattr(raw_model, "last_global_outputs", None)
        if self.hierarchical_head and outputs:
            defect_values = outputs["global_defect_logits"][0].sigmoid().detach().cpu().tolist()
            quality_values = outputs["quality_logits"][0].sigmoid().detach().cpu().tolist()
            usability_values = outputs["usability_logits"][0].softmax(0).detach().cpu().tolist()
            global_presence = {class_id: float(value) for class_id, value in zip(self.class_ids, defect_values)}
            quality = {key: float(value) for key, value in zip(self.quality_attributes, quality_values)}
            usability = {key: float(value) for key, value in zip(
                ("usable", "review", "unusable"), usability_values)}
            hierarchical_available = True
        # B0/B1 mirror detection presence explicitly; they do not pretend to
        # provide an independent whole-image diagnosis signal.
        return {
            "width": int(width), "height": int(height), "detections": detections,
            "global_defect_probabilities": global_presence,
            "quality_probabilities": quality,
            "usability_probabilities": usability,
            "hierarchical_outputs_available": hierarchical_available,
        }


def validate_v3_deployment(deployment: dict, base_dir: str | Path = ".") -> dict:
    required = ("model_version", "protocol_version", "taxonomy_sha256", "checkpoint",
                "class_ids", "class_names")
    missing = [key for key in required if key not in deployment]
    if missing:
        raise ValueError("v3 deployment 缺少字段: " + ",".join(missing))
    if deployment["protocol_version"] != "dram-det-v3":
        raise ValueError("v3 deployment 协议必须为 dram-det-v3")
    if len(deployment["class_ids"]) != len(deployment["class_names"]):
        raise ValueError("class_ids 与 class_names 长度不一致")
    checkpoint = Path(base_dir) / deployment["checkpoint"]["path"]
    expected = deployment["checkpoint"].get("sha256")
    if checkpoint.exists() and expected and sha256_file(checkpoint) != expected:
        raise ValueError("v3 checkpoint 哈希不匹配")
    return deployment


class DetectionRuntime:
    def __init__(self, deployment_path: str | Path | None = None, predictor: Predictor | None = None):
        self.deployment_path = Path(deployment_path) if deployment_path else None
        self.deployment: dict | None = None
        self.predictor = predictor
        self.review_queue: list[dict] = []
        if self.deployment_path and self.deployment_path.exists():
            self.deployment = validate_v3_deployment(
                json.loads(self.deployment_path.read_text(encoding="utf-8")),
                self.deployment_path.parent)
            if self.predictor is None and self.deployment.get("ready", False):
                checkpoint = self.deployment_path.parent / self.deployment["checkpoint"]["path"]
                self.predictor = UltralyticsPredictor(
                    checkpoint, self.deployment["class_ids"],
                    self.deployment.get("imgsz", 640), self.deployment.get("confidence", .25),
                    self.deployment.get("device"), self.deployment.get("quality_attributes", []),
                    self.deployment.get("hierarchical_head", False))

    @property
    def ready(self) -> bool:
        return self.predictor is not None and self.deployment is not None

    def model_info(self) -> dict:
        deployment = self.deployment or {}
        return {
            "ready": self.ready,
            "task": "hierarchical_detection",
            "model_version": deployment.get("model_version"),
            "protocol_version": deployment.get("protocol_version", "dram-det-v3"),
            "variant": deployment.get("variant"),
            "class_ids": deployment.get("class_ids", []),
            "class_names": deployment.get("class_names", {}),
            "quality_attributes": deployment.get("quality_attributes", []),
            "taxonomy_sha256": deployment.get("taxonomy_sha256"),
            "checkpoint_sha256": (deployment.get("checkpoint") or {}).get("sha256"),
            "hierarchical_head": bool(deployment.get("hierarchical_head", False)),
            "p2_head": bool(deployment.get("p2_head", False)),
        }

    def diagnose_path(self, path: str | Path, image_name: str) -> dict:
        if not self.ready:
            raise RuntimeError("dram-det-v3 模型未就绪")
        prediction = self.predictor.predict(path)
        deployment = self.deployment or {}
        class_names = {str(key): value for key, value in deployment["class_names"].items()}
        detections = []
        for item in prediction.get("detections", []):
            row = dict(item)
            row["class_name"] = class_names.get(str(item["class_id"]), str(item["class_id"]))
            detections.append(row)
        detection_presence = post_nms_presence(detections, deployment["class_ids"])
        global_presence = {
            int(key): float(value) for key, value in
            prediction.get("global_defect_probabilities", detection_presence).items()
        }
        quality = {str(key): float(value) for key, value in prediction.get("quality_probabilities", {}).items()}
        usability = {str(key): float(value) for key, value in prediction.get(
            "usability_probabilities", {"usable": 1, "review": 0, "unusable": 0}).items()}
        raw_policy = deployment.get("review_policy", {})
        policy = ReviewPolicy(**{key: raw_policy[key] for key in raw_policy
                                 if key in ReviewPolicy.__dataclass_fields__})
        review = review_decision(global_presence, detection_presence, quality, usability, policy)
        result = {
            "image_name": image_name,
            "image_size": [prediction.get("width"), prediction.get("height")],
            "detections": detections,
            "global_defect_probabilities": {str(key): value for key, value in global_presence.items()},
            "quality_flags": [key for key, value in quality.items() if value >= .5],
            "quality_probabilities": quality,
            "usability": max(usability, key=usability.get) if usability else "review",
            "usability_probabilities": usability,
            "hierarchical_outputs_available": bool(prediction.get("hierarchical_outputs_available", False)),
            **review,
            "model_version": deployment["model_version"],
            "protocol_version": deployment["protocol_version"],
            "taxonomy_sha256": deployment["taxonomy_sha256"],
            "checkpoint_sha256": deployment["checkpoint"]["sha256"],
        }
        result["review_score"] = max(
            float(result.get("max_disagreement", 0)),
            1.0 - float(usability.get("usable", 0)),
            max(quality.values(), default=0.0),
        )
        if result["review_required"]:
            self.review_queue.append(result)
            self.review_queue = sorted(
                self.review_queue[-1000:], key=lambda row: row["review_score"], reverse=True)
        return result
