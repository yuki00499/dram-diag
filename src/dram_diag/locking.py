import json
from pathlib import Path

from .compat import file_sha256


def create_experiment_lock(manifest, checkpoints, output, model_form="single", alpha=.5, force=False):
    destination = Path(output)
    if destination.exists() and not force:
        raise FileExistsError(f"锁文件已存在: {destination}")
    if model_form not in {"single", "ensemble"}:
        raise ValueError("model_form 只能是 single 或 ensemble")
    if model_form == "single" and len(checkpoints) != 1:
        raise ValueError("single 形式必须且只能提供一个 checkpoint")
    payload = {
        "lock_version": 1,
        "protocol_version": manifest["protocol_version"],
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "model_form": model_form,
        "checkpoints": [{"path": str(Path(path)), "sha256": file_sha256(path)} for path in checkpoints],
        "unknown_score": {"method": "max_probability_plus_nearest_prototype", "alpha": float(alpha)},
        "gates": {
            "closed": {"macro_f1": .70, "macro_f1_ci_lower": .60, "balanced_accuracy": .70, "top5_accuracy": .85, "ece_max": .10},
            "open": {"auroc": .80, "auroc_ci_lower": .70, "macro_unknown_recall": .70, "known_false_reject_rate_max": .20, "unknown_false_accept_rate_max": .30},
        },
        "final_test_consumed": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def validate_lock(lock, manifest):
    if lock["protocol_version"] != manifest["protocol_version"] or lock["manifest_fingerprint"] != manifest["manifest_fingerprint"]:
        raise ValueError("实验锁与 manifest 不匹配")
    for item in lock["checkpoints"]:
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"checkpoint 哈希不匹配: {item['path']}")
    return lock
