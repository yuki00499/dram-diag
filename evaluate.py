import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/"src"))
from dram_diag.audit import audit
p=argparse.ArgumentParser(); p.add_argument("--data-root",default="晶圆缺陷分类数据集"); args=p.parse_args()
print(audit(args.data_root))
