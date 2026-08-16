import io
import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .calibration import fused_unknown_score, softmax
from .data import load_labels
from .inference import ModelRunner
from .protocol import validate_manifest, validate_multilabel_manifest
from .retrieval import RetrievalIndex


class BatchRequest(BaseModel):
    image_names: list[str]


class Runtime:
    def __init__(self, deployment_path=None):
        self.data_root = Path(os.getenv("DRAM_DATA_ROOT", "晶圆缺陷分类数据集"))
        self.deployment_path = Path(deployment_path or os.getenv("DRAM_DEPLOYMENT", "artifacts/deployment.json"))
        self.deployment = None
        self.runner = None
        self.manifest = None
        self.prototypes = None
        self.retrieval = None
        self.rows = load_labels(self.data_root)
        self.labels_by_name = {row["IMAGE_NAME"]: int(row["DEFECT_ID"]) for row in self.rows}
        if self.deployment_path.exists():
            self._load()

    def _load(self):
        self.deployment = json.loads(self.deployment_path.read_text(encoding="utf-8"))
        if not self.deployment.get("ready"):
            return
        if self.deployment.get("task") == "multilabel":
            self._load_multilabel()
            return
        self.manifest = validate_manifest(json.loads(Path(self.deployment["manifest"]).read_text(encoding="utf-8")), self.rows)
        self.runner = ModelRunner([item["path"] for item in self.deployment["checkpoints"]], self.manifest, self.data_root)
        _, train_embeddings = self.runner.infer_items(self.manifest["splits"]["train"])
        centers = []
        for defect_id in sorted(self.manifest["classification_classes"]):
            mask = np.asarray([item["defect_id"] == defect_id for item in self.manifest["splits"]["train"]])
            center = train_embeddings[mask].mean(0)
            centers.append(center / np.linalg.norm(center))
        self.prototypes = np.asarray(centers)
        records = json.loads(Path(self.deployment["retrieval_records"]).read_text(encoding="utf-8"))
        embeddings = np.load(self.deployment["retrieval_embeddings"])["embeddings"]
        self.retrieval = RetrievalIndex(embeddings, records)

    def _load_multilabel(self):
        self.manifest = validate_multilabel_manifest(json.loads(Path(self.deployment["manifest"]).read_text(encoding="utf-8")))
        self.runner = ModelRunner([item["path"] for item in self.deployment["checkpoints"]], self.manifest, self.data_root)
        _, train_embeddings = self.runner.infer_items(self.manifest["splits"]["train"])
        self.true_labels = {}
        for split in ("train", "validation", "test_known"):
            for item in self.manifest["splits"][split]:
                self.true_labels[item["image_name"]] = sorted(item.get("labels", []))
        self.labels_by_name = {name: types for name, types in self.true_labels.items()}
        centers = []
        for type_id in self.manifest["types"]:
            mask = np.asarray([type_id in item["labels"] for item in self.manifest["splits"]["train"]])
            if mask.any():
                center = train_embeddings[mask].mean(0)
                centers.append(center / np.linalg.norm(center))
            else:
                centers.append(np.zeros(train_embeddings.shape[1]))
        self.prototypes = np.asarray(centers)
        records = json.loads(Path(self.deployment["retrieval_records"]).read_text(encoding="utf-8"))
        embeddings = np.load(self.deployment["retrieval_embeddings"])["embeddings"]
        self.retrieval = RetrievalIndex(embeddings, records)

    @property
    def ready(self):
        return self.runner is not None

    def diagnose_path(self, path, image_name):
        if not self.ready:
            raise HTTPException(503, "模型未就绪，请先生成并指定通过门槛的部署清单")
        from PIL import Image, ImageStat
        with Image.open(path) as quality_image:
            gray = quality_image.convert("L")
            low_quality = min(gray.size) < 32 or ImageStat.Stat(gray).stddev[0] < 2.0
        logits, embeddings = self.runner.infer_paths([path])
        if self.deployment.get("task") == "multilabel":
            return self.diagnose_multilabel(logits[0], embeddings[0], image_name, low_quality)
        probabilities = softmax(logits, self.deployment["temperature"])[0]
        score = float(fused_unknown_score(probabilities[None], embeddings, self.prototypes, self.deployment["unknown_alpha"])[0])
        order = np.argsort(probabilities)[::-1][:5]
        candidates = [{"defect_id": self.runner.idx_to_class[int(index)], "confidence": float(probabilities[index])} for index in order]
        prototype_similarity = float((embeddings @ self.prototypes.T).max())
        rejection_enabled = bool(self.deployment["unknown_rejection_enabled"])
        rejected = rejection_enabled and score >= self.deployment["unknown_threshold"]
        uncertain = probabilities[order[0]] < .7 or not rejection_enabled
        status = "low_quality" if low_quality else "unknown" if rejected else "known_uncertain" if uncertain else "known_confident"
        return {
            "image_name": image_name, "predicted_class": None if rejected else candidates[0]["defect_id"],
            "confidence": float(probabilities[order[0]]), "prototype_similarity": prototype_similarity,
            "unknown_score": score, "status": status,
            "review_required": status != "known_confident", "top5_candidates": candidates,
            "top5_similar_cases": self.retrieval.search(embeddings[0], 5, image_name),
            "model_version": self.deployment["model_version"],
        }

    def diagnose_multilabel(self, logit_vector, embedding, image_name, low_quality=False):
        import torch
        import numpy as np
        types = self.deployment["types"]
        type_names = self.deployment.get("type_names", {})
        type_names_full = self.deployment.get("type_names_full", {})
        threshold = float(self.deployment.get("threshold", .5))
        probabilities = torch.sigmoid(torch.from_numpy(np.asarray(logit_vector, dtype=np.float32))).numpy()
        order = np.argsort(probabilities)[::-1]
        detected = [types[index] for index in order if probabilities[index] >= threshold]
        candidates = [{
            "type_id": types[index],
            "name": type_names.get(str(types[index]), f"类型 {types[index]}"),
            "confidence": float(probabilities[index]),
        } for index in order]
        true_ids = self.true_labels.get(image_name, [])
        true_labels = [{"type_id": tid, "name": type_names_full.get(str(tid), f"类型 {tid}")} for tid in true_ids]
        max_prob = float(probabilities[order[0]])
        status = "low_quality" if low_quality else "known_confident" if detected else "unknown"
        return {
            "image_name": image_name,
            "predicted_types": detected,
            "true_labels": true_labels,
            "type_probabilities": {str(types[index]): float(probabilities[index]) for index in order},
            "confidence": max_prob,
            "prototype_similarity": float((embedding @ self.prototypes.T).max()),
            "status": status,
            "review_required": status != "known_confident",
            "top5_candidates": candidates,
            "top5_similar_cases": self.retrieval.search(embedding, 5, image_name),
            "model_version": self.deployment["model_version"],
        }


def create_app(deployment_path=None):
    app = FastAPI(title="DRAM 缺陷诊断", version="2.0")
    runtime = Runtime(deployment_path)

    @app.get("/api/model/info")
    def model_info():
        deployment = runtime.deployment or {}
        checkpoint = runtime.runner.checkpoints[0] if runtime.ready else {}
        config = checkpoint.get("config", {})
        if runtime.manifest and runtime.manifest.get("task") == "multilabel":
            num_classes = len(runtime.manifest["types"])
            classification_samples = runtime.manifest["role_stats"]["total_images"]
        elif runtime.manifest:
            num_classes = len(runtime.manifest["classification_classes"])
            classification_samples = runtime.manifest["role_stats"]["classification"]["samples"]
        else:
            num_classes, classification_samples = 7, 1148
        return {"ready": runtime.ready, "model_version": deployment.get("model_version"),
                "protocol_version": deployment.get("protocol_version", "ge20-ml-v1"),
                "mode": deployment.get("mode", "not_deployed"),
                "task": deployment.get("task", "single_label"),
                "unknown_rejection_enabled": deployment.get("unknown_rejection_enabled", False),
                "architecture": config.get("backbone", "resnet18"), "input_size": config.get("image_size", [480, 320]),
                "num_classes": num_classes,
                "classification_samples": classification_samples,
                "total_samples": len(runtime.rows), "device": str(runtime.runner.device) if runtime.ready else "-",
                "best_val_macro_f1": checkpoint.get("best_metrics", {}).get("validation_macro_f1")}

    @app.get("/api/model/history")
    def history():
        if not runtime.ready:
            return []
        path = Path(runtime.deployment["checkpoints"][0]["path"]).parent / "history.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    @app.get("/api/model/class-distribution")
    def distribution():
        if runtime.manifest and runtime.manifest.get("task") == "multilabel":
            return {"task": "multilabel", "types": runtime.manifest["types"],
                    "type_counts": runtime.manifest["role_stats"]["types"],
                    "type_names": runtime.deployment.get("type_names", {}) if runtime.deployment else {},
                    "classification_classes": [], "case_library_classes": [],
                    "split_sizes": runtime.manifest["role_stats"]["split_samples"]}
        counts = runtime.manifest["counts"] if runtime.manifest else {}
        splits = runtime.manifest["role_stats"]["split_samples"] if runtime.manifest else {}
        return {"task": "single_label", "counts": counts,
                "classification_classes": runtime.manifest["classification_classes"] if runtime.manifest else [],
                "case_library_classes": runtime.manifest["case_library_classes"] if runtime.manifest else [], "split_sizes": splits}

    @app.get("/api/dataset/images")
    def dataset_images(offset: int = 0, limit: int = 40, defect_id: int | None = None, q: str = ""):
        if runtime.manifest and runtime.manifest.get("task") == "multilabel":
            full_names = runtime.deployment.get("type_names_full", {})
            items = []
            for name in sorted(runtime.true_labels, key=lambda n: int(n.split("_")[1].split(".")[0])):
                types = [{"type_id": tid, "name": full_names.get(str(tid), f"类型 {tid}")} for tid in runtime.true_labels[name]]
                if defect_id is not None and not any(item["type_id"] == defect_id for item in types):
                    continue
                if q and q.lower() not in name.lower():
                    continue
                items.append({"image_name": name, "types": types})
            return {"total": len(items), "items": items[offset:offset + min(limit, 200)]}
        items = [{"image_name": row["IMAGE_NAME"], "defect_id": int(row["DEFECT_ID"])} for row in runtime.rows]
        if defect_id is not None:
            items = [item for item in items if item["defect_id"] == defect_id]
        if q:
            items = [item for item in items if q.lower() in item["image_name"].lower()]
        return {"total": len(items), "items": items[offset:offset + min(limit, 200)]}

    @app.get("/api/dataset/image/{image_name}")
    def dataset_image(image_name: str):
        if Path(image_name).name != image_name or image_name not in runtime.labels_by_name:
            raise HTTPException(404, "图片不存在")
        return Response(content=(runtime.data_root / "images" / image_name).read_bytes(), media_type="image/jpeg")

    @app.post("/api/diagnose/upload")
    async def diagnose_upload(file: UploadFile = File(...)):
        from PIL import Image
        try:
            data = await file.read()
            image = Image.open(io.BytesIO(data))
            image.verify()
        except Exception as exc:
            raise HTTPException(400, "无效图像") from exc
        suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(data)
            temporary = Path(handle.name)
        try:
            return runtime.diagnose_path(temporary, file.filename or "upload.jpg")
        finally:
            temporary.unlink(missing_ok=True)

    @app.post("/api/diagnose/batch")
    def diagnose_batch(request: BatchRequest):
        if len(request.image_names) > 200:
            raise HTTPException(400, "单次最多处理 200 张")
        results = []
        for name in request.image_names:
            if Path(name).name != name or name not in runtime.labels_by_name:
                results.append({"image_name": name, "error": "图片不存在"})
            else:
                results.append(runtime.diagnose_path(runtime.data_root / "images" / name, name))
        return {"results": results}

    web = Path(__file__).parents[2] / "web"
    app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app


app = create_app()
