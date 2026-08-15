from pathlib import Path
import uuid

from dram_diag.data import compute_gray_stats
from dram_diag.training import resolve_split


def test_gray_stats_reports_training_only():
    from PIL import Image
    image = Path("artifacts") / f"test-gray-{uuid.uuid4().hex}.jpg"
    Image.new("L", (20, 20), 128).save(image)
    try:
        assert compute_gray_stats([image], (20, 20))["source"] == "train_only"
    finally:
        image.unlink(missing_ok=True)


def test_cv_resolution_never_adds_holdouts():
    manifest = {"splits": {"train": [{"image_name": "a"}], "validation": [{"image_name": "b"}], "test_known": [{"image_name": "c"}]}, "cv_folds": [{"fold": 0, "validation": [{"image_name": "b"}]}]}
    train, validation = resolve_split(manifest, 0)
    assert {item["image_name"] for item in train + validation} == {"a", "b"}
