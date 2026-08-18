import io
import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .inference import ModelRunner
from .protocol import validate_multilabel_manifest
from .retrieval import RetrievalIndex


class BatchRequest(BaseModel):
    image_names: list[str]


class Runtime:
    def __init__(self, deployment_path=None):
        self.data_root = Path(os.getenv("DRAM_DATA_ROOT", "晶圆缺陷分类数据集"))
        self.deployment_path = Path(deployment_path or os.getenv("DRAM_DEPLOYMENT", "artifacts/deployment-v2.json"))
        self.deployment = None
        self.runner = None
        self.manifest = None
        self.retrieval = None
        self.split_images = {}
        self._evaluation = None
        self.labels_by_name = {}
        if self.deployment_path.exists():
            self._load()

    def _load(self):
        self.deployment = json.loads(self.deployment_path.read_text(encoding="utf-8"))
        if not self.deployment.get("ready"):
            return
        if self.deployment.get("task") != "multilabel":
            raise ValueError("部署清单必须是多标签模型")
        self._load_multilabel()

    def _load_multilabel(self):
        self.manifest = validate_multilabel_manifest(json.loads(Path(self.deployment["manifest"]).read_text(encoding="utf-8")))
        self.runner = ModelRunner([item["path"] for item in self.deployment["checkpoints"]], self.manifest, self.data_root)
        self.true_labels = {}
        available_splits = ("development", "test_known") if "development" in self.manifest["splits"] else ("train", "validation", "test_known")
        for split in available_splits:
            self.split_images[split] = [item["image_name"] for item in self.manifest["splits"][split]]
            for item in self.manifest["splits"][split]:
                self.true_labels[item["image_name"]] = sorted(item.get("labels", []))
        self.labels_by_name = {name: types for name, types in self.true_labels.items()}
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
        probabilities, embeddings = self.runner.infer_multilabel_paths([path])
        return self.diagnose_multilabel(probabilities[0], embeddings[0], image_name, low_quality)

    def diagnose_multilabel(self, probability_vector, embedding, image_name, low_quality=False):
        import numpy as np
        types = self.deployment["types"]
        type_names = self.deployment.get("type_names", {})
        type_names_full = self.deployment.get("type_names_full", {})
        thresholds = self.deployment.get("thresholds", {})
        threshold_by_type = {int(type_id): float(thresholds.get(str(type_id), self.deployment.get("threshold", .5)))
                             for type_id in types}
        probabilities = np.asarray(probability_vector, dtype=np.float32)
        order = np.argsort(probabilities)[::-1]
        detected = [types[index] for index in order if probabilities[index] >= threshold_by_type[types[index]]]
        candidates = [{
            "type_id": types[index],
            "name": type_names.get(str(types[index]), f"类型 {types[index]}"),
            "confidence": float(probabilities[index]),
        } for index in order]
        true_ids = self.true_labels.get(image_name, [])
        true_labels = [{"type_id": tid, "name": type_names_full.get(str(tid), f"类型 {tid}")} for tid in true_ids]
        max_prob = float(probabilities[order[0]])
        decision_margin = float(min(abs(probabilities[index] - threshold_by_type[types[index]]) for index in range(len(types))))
        low_confidence = decision_margin <= float(self.deployment.get("review_margin", .1))
        status = ("low_quality" if low_quality else "no_detection" if not detected else
                  "known_uncertain" if low_confidence else "known_confident")
        return {
            "image_name": image_name,
            "predicted_types": detected,
            "true_labels": true_labels,
            "type_probabilities": {str(types[index]): float(probabilities[index]) for index in order},
            "thresholds": {str(type_id): threshold_by_type[type_id] for type_id in types},
            "confidence": max_prob,
            "decision_margin": decision_margin,
            "low_confidence": low_confidence,
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
        num_classes = len(runtime.manifest["types"]) if runtime.manifest else 7
        classification_samples = runtime.manifest["role_stats"]["total_images"] if runtime.manifest else 0
        return {"ready": runtime.ready, "model_version": deployment.get("model_version"),
                "protocol_version": deployment.get("protocol_version", "dram-ml-v2"),
                "mode": deployment.get("mode", "not_deployed"),
                "task": "multilabel",
                "architecture": config.get("backbone", "resnet18"), "input_size": config.get("image_size", [480, 320]),
                "num_classes": num_classes,
                "classification_samples": classification_samples,
                "total_samples": classification_samples, "device": str(runtime.runner.device) if runtime.ready else "-",
                "best_val_macro_f1": checkpoint.get("best_metrics", {}).get("validation_macro_f1"),
                "threshold": runtime.deployment.get("threshold", .5) if runtime.deployment else None,
                "thresholds": runtime.deployment.get("thresholds", {}) if runtime.deployment else {},
                "review_margin": runtime.deployment.get("review_margin", .1) if runtime.deployment else .1,
                "type_names": runtime.deployment.get("type_names", {}) if runtime.deployment else {},
                "type_names_full": runtime.deployment.get("type_names_full", {}) if runtime.deployment else {}}

    @app.get("/api/model/history")
    def history():
        if not runtime.ready:
            return []
        path = Path(runtime.deployment["checkpoints"][0]["path"]).parent / "history.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    @app.get("/api/model/class-distribution")
    def distribution():
        if runtime.manifest:
            return {"task": "multilabel", "types": runtime.manifest["types"],
                    "type_counts": runtime.manifest["role_stats"]["types"],
                    "type_names": runtime.deployment.get("type_names", {}) if runtime.deployment else {},
                    "split_sizes": runtime.manifest["role_stats"]["split_samples"]}
        return {"task": "multilabel", "types": [], "type_counts": {}, "type_names": {}, "split_sizes": {}}

    @app.get("/api/model/evaluation")
    def evaluation():
        if not runtime.ready:
            return None
        if runtime._evaluation is None:
            oof_path = runtime.deployment.get("oof_evaluation")
            if oof_path and Path(oof_path).exists():
                from .metrics import multilabel_diagnostics
                payload = json.loads(Path(oof_path).read_text(encoding="utf-8"))
                runtime._evaluation = dict(payload["metrics"])
                prediction_path = Path(oof_path).with_name("oof_predictions.json")
                if prediction_path.exists():
                    records = json.loads(prediction_path.read_text(encoding="utf-8"))
                    labels = [int(value) for value in payload.get("labels", runtime.manifest["types"])]
                    truth = np.zeros((len(records), len(labels)), dtype=np.int8)
                    probabilities = np.zeros_like(truth, dtype=np.float32)
                    type_index = {label: index for index, label in enumerate(labels)}
                    for row_index, record in enumerate(records):
                        for label in record.get("labels", []):
                            if int(label) in type_index:
                                truth[row_index, type_index[int(label)]] = 1
                        probabilities[row_index] = np.asarray(record["probabilities"], dtype=np.float32)
                    thresholds = np.asarray([
                        payload.get("thresholds", {}).get(str(label), .5) for label in labels], dtype=float)
                    runtime._evaluation.update(multilabel_diagnostics(truth, probabilities, labels, thresholds))
                runtime._evaluation["source"] = "oof"
                runtime._evaluation["thresholds"] = payload.get("thresholds", runtime._evaluation.get("thresholds", {}))
                for key in ("fold_metrics", "fold_core_macro_f1_mean", "fold_core_macro_f1_std", "core_macro_f1_ci", "threshold_stability"):
                    if key in payload:
                        runtime._evaluation[key] = payload[key]
                return runtime._evaluation
            from .metrics import multilabel_metrics
            split_name = "validation" if "validation" in runtime.manifest["splits"] else "development"
            validation = runtime.manifest["splits"][split_name]
            probabilities, _ = runtime.runner.infer_multilabel_items(validation)
            labels = np.zeros((len(validation), len(runtime.manifest["types"])), dtype=np.float32)
            type_index = {int(tid): index for index, tid in enumerate(runtime.manifest["types"])}
            for index, item in enumerate(validation):
                for label in item.get("labels", []):
                    labels[index, type_index[int(label)]] = 1.0
            threshold_map = runtime.deployment.get("thresholds", {})
            thresholds = np.asarray([
                threshold_map.get(str(label), runtime.deployment.get("threshold", .5))
                for label in runtime.manifest["types"]], dtype=float)
            runtime._evaluation = multilabel_metrics(
                labels, probabilities, runtime.manifest["types"], thresholds,
                runtime.manifest.get("core_types", runtime.manifest["types"]),
                runtime.manifest.get("tracked_pairs", []))
            runtime._evaluation["source"] = split_name
            runtime._evaluation["per_label_auc"] = {
                key: (None if value != value else float(value))
                for key, value in runtime._evaluation["per_label_auc"].items()
            }
        return runtime._evaluation

    @app.get("/api/dataset/images")
    def dataset_images(offset: int = 0, limit: int = 40, defect_id: int | None = None, q: str = "", split: str = "",
                       sample_count: int = 0, sample_seed: int = 42):
        if runtime.manifest:
            full_names = runtime.deployment.get("type_names_full", {})
            names = runtime.true_labels
            if split:
                allowed = set(runtime.split_images.get(split, []))
                names = {name: labels for name, labels in names.items() if name in allowed}
            items = []
            for name in sorted(names, key=lambda n: int(n.split("_")[1].split(".")[0])):
                types = [{"type_id": tid, "name": full_names.get(str(tid), f"类型 {tid}")} for tid in runtime.true_labels[name]]
                if defect_id is not None and not any(item["type_id"] == defect_id for item in types):
                    continue
                if q and q.lower() not in name.lower():
                    continue
                items.append({"image_name": name, "types": types})
            total = len(items)
            if sample_count:
                count = min(max(1, sample_count), 200, total)
                indexes = np.random.default_rng(sample_seed).choice(total, count, replace=False)
                return {"total": total, "items": [items[int(index)] for index in indexes]}
            return {"total": total, "items": items[offset:offset + min(limit, 200)]}
        return {"total": 0, "items": []}

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

    @app.post("/api/diagnose/batch-upload")
    async def diagnose_batch_upload(files: list[UploadFile] = File(...)):
        from PIL import Image
        if len(files) > 200:
            raise HTTPException(400, "单次最多处理 200 张")
        results = []
        for file in files:
            name = file.filename or "upload.jpg"
            try:
                data = await file.read()
                image = Image.open(io.BytesIO(data))
                image.verify()
            except Exception:
                results.append({"image_name": name, "error": "无效图像"})
                continue
            suffix = Path(name).suffix or ".jpg"
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(data)
                temporary = Path(handle.name)
            try:
                results.append(runtime.diagnose_path(temporary, name))
            except Exception as exc:
                results.append({"image_name": name, "error": str(exc)})
            finally:
                temporary.unlink(missing_ok=True)
        return {"results": results}

    web = Path(__file__).parents[2] / "web"

    class NoCacheStaticFiles(StaticFiles):
        async def get_response(self, path: str, scope):
            response = await super().get_response(path, scope)
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            return response

    app.mount("/", NoCacheStaticFiles(directory=web, html=True), name="web")
    return app


app = create_app()
