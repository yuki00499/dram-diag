"""Serve the tracked dram-det-v3 bounding-box annotation application locally."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from dram_diag.detection_data import audit_annotations, blank_annotation_payload, load_taxonomy


class AnnotationPayload(BaseModel):
    payload: dict


def create_app(taxonomy_path="configs/taxonomy_v3.yaml",
               image_root="晶圆缺陷分类数据集/images",
               annotations_path="annotations/dram_det_v3.json",
               review_sample_path="annotations/review_sample_v3.json"):
    app = FastAPI(title="DRAM detection v3 annotation")
    root = Path(__file__).parents[1]
    taxonomy_path = root / taxonomy_path
    image_root = root / image_root
    annotations_path = root / annotations_path
    review_sample_path = root / review_sample_path
    taxonomy = load_taxonomy(taxonomy_path)
    images = sorted(
        (path.name for path in image_root.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda name: int(Path(name).stem.split("_")[-1]))

    @app.get("/api/annotation/config")
    def config():
        review_sample = []
        if review_sample_path.exists():
            review_sample = json.loads(review_sample_path.read_text(encoding="utf-8")).get("images", [])
        return {"taxonomy": taxonomy, "images": images, "total": len(images),
                "review_sample": review_sample}

    @app.get("/api/annotation/state")
    def state():
        if annotations_path.exists():
            return json.loads(annotations_path.read_text(encoding="utf-8"))
        return blank_annotation_payload(taxonomy)

    @app.put("/api/annotation/state")
    def save_state(request: AnnotationPayload):
        report = audit_annotations(request.payload, taxonomy, image_root, require_complete=False)
        if report["errors"]:
            raise HTTPException(422, {"message": "标注未保存", "errors": report["errors"][:20]})
        annotations_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = annotations_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(request.payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(annotations_path)
        return {"saved": True, "path": str(annotations_path.relative_to(root)),
                "warnings": report["warnings"], "stats": report["stats"]}

    @app.get("/api/annotation/image/{image_name}")
    def image(image_name: str):
        if Path(image_name).name != image_name or image_name not in images:
            raise HTTPException(404, "图像不存在")
        return Response((image_root / image_name).read_bytes(), media_type="image/jpeg")

    app.mount("/", StaticFiles(directory=root / "annotation", html=True), name="annotation")
    return app


def main():
    parser = argparse.ArgumentParser(description="启动 dram-det-v3 标注站")
    parser.add_argument("--taxonomy", default="configs/taxonomy_v3.yaml")
    parser.add_argument("--images", default="晶圆缺陷分类数据集/images")
    parser.add_argument("--annotations", default="annotations/dram_det_v3.json")
    parser.add_argument("--review-sample", default="annotations/review_sample_v3.json")
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()
    uvicorn.run(create_app(args.taxonomy, args.images, args.annotations, args.review_sample),
                host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
