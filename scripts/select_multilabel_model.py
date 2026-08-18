import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from dram_diag.competition import lock_payload
from dram_diag.protocol import validate_multilabel_manifest


def load_evaluation(path):
    return json.loads((Path(path) / "oof_evaluation.json").read_text(encoding="utf-8"))


def mean_pair_f1(evaluation):
    pairs = evaluation["metrics"].get("pair_metrics", {})
    return sum(value["f1"] for value in pairs.values()) / max(1, len(pairs))


def main():
    parser = argparse.ArgumentParser(description="比较三种多标签模型并冻结最终候选")
    parser.add_argument("--bce", default="runs/dram_ml_v2_bce")
    parser.add_argument("--weighted", default="runs/dram_ml_v2_weighted")
    parser.add_argument("--graph", default="runs/dram_ml_v2_graph")
    parser.add_argument("--manifest", default="artifacts/dram_ml_v2/split_manifest.json")
    parser.add_argument("--out", default="artifacts/dram_ml_v2/model_selection.json")
    parser.add_argument("--min-delta", type=float, default=.005)
    args = parser.parse_args()
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"模型选择清单已存在: {destination}")
    manifest = validate_multilabel_manifest(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    candidates = {name: load_evaluation(path) for name, path in
                  (("bce", args.bce), ("weighted_bce", args.weighted), ("graph_bce", args.graph))}
    versions = {value["protocol_version"] for value in candidates.values()}
    fingerprints = {value["manifest_fingerprint"] for value in candidates.values()}
    if versions != {manifest["protocol_version"]} or fingerprints != {manifest["manifest_fingerprint"]}:
        raise ValueError("候选实验协议或 manifest 指纹不一致")
    weighted, graph = candidates["weighted_bce"], candidates["graph_bce"]
    weighted_core = weighted["metrics"]["core_macro_f1"]
    graph_core = graph["metrics"]["core_macro_f1"]
    weighted_pair, graph_pair = mean_pair_f1(weighted), mean_pair_f1(graph)
    graph_retained = graph_core >= weighted_core and (
        graph_core - weighted_core >= args.min_delta or graph_pair - weighted_pair >= args.min_delta)
    selected_name = "graph_bce" if graph_retained else "weighted_bce"
    selected_dir = {"graph_bce": args.graph, "weighted_bce": args.weighted}[selected_name]
    thresholds = candidates[selected_name]["thresholds"]
    result = {
        "selection_rule": {"minimum_clear_improvement": args.min_delta,
                           "graph_core_not_below_weighted": True,
                           "graph_must_improve_core_or_pair": True},
        "candidates": {name: {"core_macro_f1": value["metrics"]["core_macro_f1"],
                              "all_label_macro_f1": value["metrics"]["macro_f1"],
                              "mean_pair_f1": mean_pair_f1(value),
                              "fold_std": value["fold_core_macro_f1_std"]}
                       for name, value in candidates.items()},
        "graph_retained": graph_retained,
        "selected": selected_name,
        "lock": lock_payload(selected_dir, manifest, thresholds, f"dram-ml-v2-{selected_name}-ensemble"),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
