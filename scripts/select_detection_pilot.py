"""Select a reproducible 100-image pilot without turning v2 labels into v3 truth."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description="选取 v3 类别定义试标图像")
    parser.add_argument("--legacy-labels", default="label_v2.json")
    parser.add_argument("--candidates", default="项目方案/标注复核候选清单.csv")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--out", default="annotations/pilot_v3.json")
    args = parser.parse_args()
    legacy = json.loads(Path(args.legacy_labels).read_text(encoding="utf-8"))["labels"]
    priority = []
    candidate_path = Path(args.candidates)
    if candidate_path.exists():
        with candidate_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                name = row.get("image_name")
                if name in legacy and name not in priority:
                    priority.append(name)
    groups = defaultdict(list)
    for name, labels in legacy.items():
        groups[tuple(sorted(int(value) for value in labels))].append(name)
    rng = random.Random(args.seed)
    for values in groups.values():
        rng.shuffle(values)
    selected = priority[: min(len(priority), args.size // 2)]
    seen = set(selected)
    ordered_groups = sorted(groups, key=lambda key: (len(groups[key]), key))
    while len(selected) < args.size:
        progressed = False
        for key in ordered_groups:
            while groups[key] and groups[key][-1] in seen:
                groups[key].pop()
            if groups[key] and len(selected) < args.size:
                name = groups[key].pop()
                selected.append(name)
                seen.add(name)
                progressed = True
        if not progressed:
            break
    payload = {
        "purpose": "taxonomy_pilot_only",
        "warning": "legacy labels are sampling hints, not v3 ground truth",
        "seed": args.seed,
        "requested_size": args.size,
        "images": selected,
    }
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"试标清单已存在，拒绝覆盖: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": len(selected), "out": str(destination)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
