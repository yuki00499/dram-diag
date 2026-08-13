import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from dram_diag.audit import audit
from dram_diag.config import load_config

cfg=load_config()
parser=argparse.ArgumentParser(description="HPOD classifier training entrypoint")
parser.add_argument("--data-root",default=cfg.get("data_root","晶圆缺陷分类数据集"))
parser.add_argument("--epochs",type=int,default=cfg.get("epochs",20))
parser.add_argument("--batch-size",type=int,default=cfg.get("batch_size",32))
parser.add_argument("--device",default="cuda")
parser.add_argument("--dry-run",action="store_true")
args=parser.parse_args(); report=audit(args.data_root)
if args.dry_run: print("数据协议已生成；训练 dry-run 完成", report["rows"]); raise SystemExit
try:
    import torch
except ImportError as exc: raise SystemExit("请安装 requirements.txt 后再训练") from exc
print("训练入口已准备，当前版本尚未实现 epoch 优化循环。")
