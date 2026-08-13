import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/"src"))
from dram_diag.audit import audit
from dram_diag.config import load_config
p=argparse.ArgumentParser(); p.add_argument("--data-root",default=load_config().get("data_root","晶圆缺陷分类数据集")); args=p.parse_args()
print(audit(args.data_root))
