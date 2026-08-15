import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from dram_diag.locking import create_experiment_lock


def main():
    parser = argparse.ArgumentParser(description="冻结最终评估决策")
    parser.add_argument("--manifest", default="artifacts/ge20/split_manifest.json")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--model-form", choices=["single", "ensemble"], default="single")
    parser.add_argument("--alpha", type=float, default=.5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    result = create_experiment_lock(manifest, args.checkpoints, args.out, args.model_form, args.alpha)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
