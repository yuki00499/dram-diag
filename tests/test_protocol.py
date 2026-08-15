import json
from pathlib import Path

from dram_diag.config import load_yaml
from dram_diag.data import load_labels
from dram_diag.protocol import build_manifest, validate_manifest

ROOT = Path(__file__).parents[1]


def make_manifest():
    return build_manifest(load_labels(ROOT / "晶圆缺陷分类数据集"), load_yaml(ROOT / "configs/protocols/ge20.yaml"))


def test_ge20_exact_counts_and_roles():
    result = make_manifest()
    assert result["role_stats"] == {
        "classification": {"classes": 21, "samples": 523},
        "case_library": {"classes": 78, "samples": 528},
        "unknown": {"calibration": {"classes": 2, "samples": 45}, "test_core": {"classes": 3, "samples": 50}, "test_stress": {"classes": 1, "samples": 4}},
        "split_samples": {"train": 379, "validation": 48, "calibration_known": 48, "test_known": 48, "calibration_unknown": 45, "test_unknown_core": 50, "test_unknown_stress": 4, "case_library": 528},
    }
    assert validate_manifest(result)


def test_cv_only_covers_development_pool_once():
    result = make_manifest()
    development = {item["image_name"] for name in ("train", "validation") for item in result["splits"][name]}
    folded = [item["image_name"] for fold in result["cv_folds"] for item in fold["validation"]]
    assert set(folded) == development
    assert len(folded) == len(set(folded))
    forbidden = {item["image_name"] for name in ("calibration_known", "test_known", "calibration_unknown", "test_unknown_core", "test_unknown_stress") for item in result["splits"][name]}
    assert not development & forbidden


def test_saved_manifest_fingerprint_validates():
    saved = json.loads((ROOT / "artifacts/ge20/split_manifest.json").read_text(encoding="utf-8"))
    assert validate_manifest(saved)
