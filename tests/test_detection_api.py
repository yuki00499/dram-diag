import io

from fastapi.testclient import TestClient
from PIL import Image

from dram_diag.api import create_app


class FakeV3Runtime:
    ready = True

    def __init__(self):
        self.deployment = {"evaluation": None}
        self.review_queue = [{"image_name": "risk.jpg", "max_disagreement": .8}]

    def model_info(self):
        return {"ready": True, "protocol_version": "dram-det-v3", "variant": "B3"}

    def diagnose_path(self, path, image_name):
        with Image.open(path) as image:
            width, height = image.size
        return {
            "image_name": image_name,
            "image_size": [width, height],
            "detections": [{
                "bbox_xyxy": [1, 2, 10, 12],
                "bbox_xyxy_normalized": [1 / width, 2 / height, 10 / width, 12 / height],
                "class_id": 0,
                "confidence": .9,
            }],
            "review_required": False,
            "review_reasons": [],
        }


def image_bytes():
    stream = io.BytesIO()
    Image.new("L", (40, 20), 127).save(stream, format="PNG")
    return stream.getvalue()


def test_v3_model_info_upload_and_review_queue_contract(tmp_path):
    app = create_app(deployment_path=tmp_path / "missing-v2.json", v3_runtime_override=FakeV3Runtime())
    client = TestClient(app)
    info = client.get("/api/v3/model/info")
    assert info.status_code == 200
    assert info.json()["protocol_version"] == "dram-det-v3"

    response = client.post(
        "/api/v3/diagnose/upload", files={"file": ("sem.png", image_bytes(), "image/png")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["image_size"] == [40, 20]
    assert payload["detections"][0]["bbox_xyxy"] == [1, 2, 10, 12]

    queue = client.get("/api/v3/review-queue").json()
    assert queue["total"] == 1
    assert queue["items"][0]["image_name"] == "risk.jpg"


def test_v3_batch_upload_preserves_per_file_errors(tmp_path):
    app = create_app(deployment_path=tmp_path / "missing-v2.json", v3_runtime_override=FakeV3Runtime())
    client = TestClient(app)
    response = client.post("/api/v3/diagnose/batch-upload", files=[
        ("files", ("ok.png", image_bytes(), "image/png")),
        ("files", ("bad.png", b"not-image", "image/png")),
    ])
    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["image_name"] == "ok.png"
    assert results[1]["image_name"] == "bad.png"
    assert "error" in results[1]
