import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from dram_diag.calibration import calibrate_rejection, fit_temperature, fused_unknown_score, softmax
from dram_diag.compat import file_sha256
from dram_diag.inference import ModelRunner
from dram_diag.locking import validate_lock
from dram_diag.metrics import closed_set_metrics, open_set_metrics, stratified_bootstrap
from dram_diag.protocol import validate_manifest
from dram_diag.retrieval import RetrievalIndex, retrieval_metrics


def main():
    parser = argparse.ArgumentParser(description="执行一次性锁定最终评估")
    parser.add_argument("--lock", required=True)
    parser.add_argument("--manifest", default="artifacts/ge20/split_manifest.json")
    parser.add_argument("--data-root", default="晶圆缺陷分类数据集")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError("最终报告已存在；请创建新的实验版本，不允许覆盖")
    manifest = validate_manifest(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    lock_path = Path(args.lock)
    lock = validate_lock(json.loads(lock_path.read_text(encoding="utf-8")), manifest)
    if lock.get("final_test_consumed"):
        raise RuntimeError("该实验锁已经消费过最终测试集")
    runner = ModelRunner([item["path"] for item in lock["checkpoints"]], manifest, args.data_root)
    splits = manifest["splits"]
    cache = {}
    for name in ("train", "calibration_known", "calibration_unknown", "test_known", "test_unknown_core", "test_unknown_stress", "case_library"):
        cache[name] = runner.infer_items(splits[name])

    train_logits, train_embeddings = cache["train"]
    prototypes = []
    for defect_id in sorted(manifest["classification_classes"]):
        mask = np.asarray([item["defect_id"] == defect_id for item in splits["train"]])
        center = train_embeddings[mask].mean(0)
        prototypes.append(center / np.linalg.norm(center))
    prototypes = np.asarray(prototypes)
    calibration_labels = np.asarray([runner.class_to_idx[item["defect_id"]] for item in splits["calibration_known"]])
    temperature = fit_temperature(cache["calibration_known"][0], calibration_labels)
    alpha = float(lock["unknown_score"]["alpha"])
    calibration_known_prob = softmax(cache["calibration_known"][0], temperature)
    calibration_unknown_prob = softmax(cache["calibration_unknown"][0], temperature)
    calibration_known_score = fused_unknown_score(calibration_known_prob, cache["calibration_known"][1], prototypes, alpha)
    calibration_unknown_score = fused_unknown_score(calibration_unknown_prob, cache["calibration_unknown"][1], prototypes, alpha)
    threshold = calibrate_rejection(calibration_known_score, calibration_unknown_score)

    test_labels = np.asarray([runner.class_to_idx[item["defect_id"]] for item in splits["test_known"]])
    test_prob = softmax(cache["test_known"][0], temperature)
    closed = closed_set_metrics(test_labels, test_prob, list(range(len(runner.class_to_idx))))
    closed["macro_f1_ci"] = stratified_bootstrap(test_labels, test_prob,
        lambda truth, values: closed_set_metrics(truth, values, list(range(len(runner.class_to_idx))))["macro_f1"], args.bootstrap_iterations)
    known_scores = fused_unknown_score(test_prob, cache["test_known"][1], prototypes, alpha)
    core_prob = softmax(cache["test_unknown_core"][0], temperature)
    core_scores = fused_unknown_score(core_prob, cache["test_unknown_core"][1], prototypes, alpha)
    core_classes = [item["defect_id"] for item in splits["test_unknown_core"]]
    opened = open_set_metrics(known_scores, core_scores, None, core_classes, threshold)
    open_truth = np.r_[np.zeros(len(known_scores), dtype=int), np.ones(len(core_scores), dtype=int)]
    open_scores = np.r_[known_scores, core_scores]
    opened["auroc_ci"] = stratified_bootstrap(open_truth, open_scores,
        lambda truth, scores: open_set_metrics(scores[truth == 0], scores[truth == 1])["auroc"], args.bootstrap_iterations)
    stress_prob = softmax(cache["test_unknown_stress"][0], temperature)
    stress_scores = fused_unknown_score(stress_prob, cache["test_unknown_stress"][1], prototypes, alpha)
    stress = {"samples": len(stress_scores), "unknown_recall": float(np.mean(stress_scores >= threshold)), "scores": stress_scores.tolist()}

    retrieval_records = splits["train"] + splits["case_library"]
    retrieval_embeddings = np.concatenate([cache["train"][1], cache["case_library"][1]])
    retrieval_index = RetrievalIndex(retrieval_embeddings, retrieval_records)
    retrieval = retrieval_metrics(retrieval_index, retrieval_embeddings, retrieval_records)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".retrieval.npz"), embeddings=retrieval_embeddings)
    output.with_suffix(".retrieval.json").write_text(json.dumps(retrieval_records, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "protocol_version": manifest["protocol_version"], "manifest_fingerprint": manifest["manifest_fingerprint"],
        "lock_sha256": file_sha256(lock_path), "created_at": datetime.now(timezone.utc).isoformat(),
        "temperature": temperature, "unknown_alpha": alpha, "unknown_threshold": threshold,
        "closed_set": closed, "open_set_core": opened, "open_set_stress": stress, "retrieval": retrieval,
        "coverage": {"classification_classes": len(manifest["classification_classes"]), "classification_samples": manifest["role_stats"]["classification"]["samples"], "all_samples": sum(manifest["counts"].values())},
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lock["final_test_consumed"] = True
    lock["final_report"] = {"path": str(output), "sha256": file_sha256(output)}
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
