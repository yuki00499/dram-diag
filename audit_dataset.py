import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent/"src"))
from dram_diag.audit import audit

parser=argparse.ArgumentParser(); parser.add_argument("--data-root",default="晶圆缺陷分类数据集"); parser.add_argument("--out",default="artifacts"); parser.add_argument("--seed",type=int,default=42)
args=parser.parse_args()
print(json.dumps(audit(args.data_root,args.out,args.seed),ensure_ascii=False,indent=2))
