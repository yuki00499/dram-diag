import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="依据双门槛生成部署清单")
    parser.add_argument("--lock", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--manifest", default="artifacts/ge20/split_manifest.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError("部署清单已存在，不允许覆盖")
    lock = json.loads(Path(args.lock).read_text(encoding="utf-8"))
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if report["manifest_fingerprint"] != lock["manifest_fingerprint"] or report["manifest_fingerprint"] != manifest["manifest_fingerprint"]:
        raise ValueError("锁、报告与 manifest 不一致")
    closed, opened = report["closed_set"], report["open_set_core"]
    closed_ok = (closed["macro_f1"] >= .70 and closed["macro_f1_ci"]["lower"] >= .60 and
                 closed["balanced_accuracy"] >= .70 and closed["top5_accuracy"] >= .85 and closed["ece"] <= .10)
    open_ok = (opened["auroc"] >= .80 and opened["auroc_ci"]["lower"] >= .70 and
               opened.get("macro_unknown_recall", 0) >= .70 and opened["known_false_reject_rate"] <= .20 and
               opened["unknown_false_accept_rate"] <= .30)
    mode = "classification_with_unknown_rejection" if closed_ok and open_ok else "classification_review_only" if closed_ok else "research_only"
    deployment = {
        "model_version": destination.stem, "mode": mode, "ready": mode != "research_only",
        "unknown_rejection_enabled": mode == "classification_with_unknown_rejection",
        "protocol_version": manifest["protocol_version"], "manifest": str(Path(args.manifest)),
        "manifest_fingerprint": manifest["manifest_fingerprint"], "checkpoints": lock["checkpoints"],
        "temperature": report["temperature"], "unknown_alpha": report["unknown_alpha"],
        "unknown_threshold": report["unknown_threshold"], "closed_gate_passed": closed_ok, "open_gate_passed": open_ok,
        "report": str(Path(args.report)), "retrieval_embeddings": str(Path(args.report).with_suffix(".retrieval.npz")),
        "retrieval_records": str(Path(args.report).with_suffix(".retrieval.json")),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(deployment, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(deployment, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
