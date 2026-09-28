"""Safety checks for resumable v3 matrix orchestration (no training started)."""

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_detection_matrix.py"
SPEC = importlib.util.spec_from_file_location("train_detection_matrix", SCRIPT)
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


def test_completed_run_requires_matching_protocol_and_both_checkpoint_hashes(tmp_path):
    run_dir = tmp_path / "run"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"best")
    (weights / "last.pt").write_bytes(b"last")
    task = {
        "variant": "B1", "fold": 2, "name": "run", "run_dir": str(run_dir),
        "manifest_sha256": "manifest", "annotation_sha256": "annotation",
        "taxonomy_sha256": "taxonomy", "config_sha256": "raw-config",
        "resolved_config_sha256": "resolved-config", "pretrained_sha256": "pretrained",
        "data_sha256": "data", "global_labels_sha256": "global", "export_report_sha256": "export",
        "epochs": None,
    }
    provenance = {
        "variant": "B1", "fold": 2, "run_name": "run",
        "manifest_sha256": "manifest", "annotation_sha256": "annotation",
        "taxonomy_sha256": "taxonomy", "experiment_config_sha256": "raw-config",
        "resolved_experiment_config_sha256": "resolved-config", "training_overrides": {},
        "ultralytics_version": "8.4.156",
        "data_yaml_sha256": "data", "global_labels_sha256": "global",
        "export_report_sha256": "export",
        "pretrained_weights": {"sha256": "pretrained"},
        "weights": {
            label: {"sha256": matrix.file_hash(weights / f"{label}.pt")}
            for label in ("best", "last")
        },
    }
    (run_dir / "v3_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    assert matrix.verify_completed(task)

    (weights / "last.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="last.pt"):
        matrix.verify_completed(task)


def test_incomplete_run_is_not_silently_skipped(tmp_path):
    task = {"run_dir": str(tmp_path / "run")}
    assert not matrix.verify_completed(task)
    (tmp_path / "run").mkdir()
    with pytest.raises(RuntimeError, match="不完整训练目录"):
        matrix.verify_completed(task)
