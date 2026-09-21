"""Select the blinded 20% v3 review set and all mandatory risk cases."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dram_diag.detection_data import load_taxonomy, select_blind_review


def main():
    parser = argparse.ArgumentParser(description="生成 dram-det-v3 盲复核样本清单")
    parser.add_argument("--annotations", default="annotations/dram_det_v3.json")
    parser.add_argument("--taxonomy", default="configs/taxonomy_v3.yaml")
    parser.add_argument("--ratio", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--out", default="annotations/review_sample_v3.json")
    args = parser.parse_args()
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"复核清单已存在，拒绝覆盖: {destination}")
    payload = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    taxonomy = load_taxonomy(args.taxonomy)
    selection = select_blind_review(payload, taxonomy, args.ratio, args.seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
