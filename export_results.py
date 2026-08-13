import argparse, csv, json
from pathlib import Path

parser=argparse.ArgumentParser(); parser.add_argument("input"); parser.add_argument("--output",default="results.csv"); args=parser.parse_args()
data=json.loads(Path(args.input).read_text(encoding="utf-8"))
rows=data if isinstance(data,list) else data.get("results",[data])
if isinstance(rows, dict): rows=[rows]
fields=sorted({k for r in rows for k in r})
with open(args.output,"w",encoding="utf-8-sig",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f"已导出 {len(rows)} 条结果到 {args.output}")
