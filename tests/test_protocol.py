import csv
import json
from pathlib import Path

from dram_diag.config import load_yaml
from dram_diag.protocol import build_multilabel_manifest, validate_multilabel_manifest

ROOT = Path(__file__).parents[1]


def load_rows():
    with (ROOT / "artifacts/ge20_ml/labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def make_manifest():
    return build_multilabel_manifest(load_rows(), load_yaml(ROOT / "configs/protocols/ge20_ml.yaml"))


def test_ge20_ml_counts_and_splits():
    result = make_manifest()
    assert result["task"] == "multilabel"
    assert result["types"] == [1, 2, 3, 4, 5, 7, 8]
    assert result["role_stats"]["total_images"] == 1148
    split_total = sum(result["role_stats"]["split_samples"].values())
    assert split_total == 1148
    assert result["role_stats"]["split_samples"]["train"] > 800
    for split in ("validation", "test_known"):
        assert result["role_stats"]["split_samples"][split] >= 100
    assert validate_multilabel_manifest(result)


def test_splits_cover_all_images_once():
    result = make_manifest()
    names = [item["image_name"] for split in ("train", "validation", "test_known") for item in result["splits"][split]]
    assert len(names) == len(set(names)) == 1148


def test_saved_manifest_fingerprint_validates():
    saved = json.loads((ROOT / "artifacts/ge20_ml/split_manifest.json").read_text(encoding="utf-8"))
    assert validate_multilabel_manifest(saved, load_rows())


def test_labels_are_valid_types():
    result = make_manifest()
    types = set(result["types"])
    for split in ("train", "validation", "test_known"):
        for item in result["splits"][split]:
            assert set(item["labels"]) <= types
            assert len(item["labels"]) == len(set(item["labels"]))
