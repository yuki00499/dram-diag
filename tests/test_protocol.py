import csv
import json
from pathlib import Path

from dram_diag.config import load_yaml
from dram_diag.protocol import build_multilabel_manifest, validate_multilabel_manifest

ROOT = Path(__file__).parents[1]


def load_rows():
    with (ROOT / "artifacts/dram_ml_v2/labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def make_manifest():
    return build_multilabel_manifest(load_rows(), load_yaml(ROOT / "configs/protocols/dram_ml_v2.yaml"))


def test_dram_ml_v2_counts_and_splits():
    result = make_manifest()
    assert result["task"] == "multilabel"
    assert result["types"] == [1, 2, 3, 4, 5, 7, 8]
    assert result["role_stats"]["total_images"] == 1148
    split_total = sum(result["role_stats"]["split_samples"].values())
    assert split_total == 1148
    assert result["role_stats"]["split_samples"] == {"development": 1033, "test_known": 115}
    assert validate_multilabel_manifest(result)


def test_splits_cover_all_images_once():
    result = make_manifest()
    names = [item["image_name"] for split in ("development", "test_known") for item in result["splits"][split]]
    assert len(names) == len(set(names)) == 1148


def test_saved_manifest_fingerprint_validates():
    saved = json.loads((ROOT / "artifacts/dram_ml_v2/split_manifest.json").read_text(encoding="utf-8"))
    assert validate_multilabel_manifest(saved, load_rows())


def test_labels_are_valid_types():
    result = make_manifest()
    types = set(result["types"])
    for split in ("development", "test_known"):
        for item in result["splits"][split]:
            assert set(item["labels"]) <= types
            assert len(item["labels"]) == len(set(item["labels"]))


def test_dram_ml_v2_iterative_folds_cover_development_and_rare_labels():
    result = make_manifest()
    assert result["protocol_version"] == "dram-ml-v2"
    assert len(result["splits"]["test_known"]) == 115
    assert len(result["splits"]["development"]) == 1033
    development = {item["image_name"] for item in result["splits"]["development"]}
    validation = [item["image_name"] for fold in result["cv_folds"] for item in fold["validation"]]
    assert set(validation) == development
    assert len(validation) == len(set(validation))
    for fold in result["cv_folds"]:
        for rare in (1, 7, 8):
            assert any(rare in item["labels"] for item in fold["validation"])
