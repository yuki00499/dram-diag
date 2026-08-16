import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from dram_diag.config import load_yaml
from dram_diag.data import load_labels
from dram_diag.protocol import validate_manifest, validate_multilabel_manifest
from dram_diag.training import run_experiment


def load_multilabel_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description="训练 DRAM 缺陷分类模型")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--out-dir")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    manifest_path = Path(args.manifest or config["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") == "multilabel":
        labels_csv = Path(config.get("labels_csv", "artifacts/ge20_ml/labels.csv"))
        validate_multilabel_manifest(manifest, load_multilabel_rows(labels_csv))
        config["multilabel"] = True
    else:
        validate_manifest(manifest, load_labels(config["data_root"]))
    seeds = args.seeds or [int(value) for value in config.get("seeds", [42])]
    output_dir = args.out_dir or config["out_dir"]
    summary = run_experiment(config, manifest, output_dir, seeds, args.folds)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
