import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from dram_diag.audit import audit_dataset


def main():
    parser = argparse.ArgumentParser(description="审计数据并生成冻结协议")
    parser.add_argument("--data-root", default="晶圆缺陷分类数据集")
    parser.add_argument("--protocol", default="configs/protocols/ge20.yaml")
    parser.add_argument("--out", default="artifacts/ge20")
    args = parser.parse_args()
    report, _ = audit_dataset(args.data_root, args.protocol, args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
