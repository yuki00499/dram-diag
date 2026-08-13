from pathlib import Path
from collections import Counter
import hashlib, json
from .data import load_labels
from .split import build_split

def audit(data_root, out_dir="artifacts", seed=42):
    root=Path(data_root); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=load_labels(root); files={p.name:p for p in (root/"images").glob("*.jpg")}
    missing=[r["IMAGE_NAME"] for r in rows if r["IMAGE_NAME"] not in files]; invalid=[]; dims=Counter(); hashes={}
    from PIL import Image
    for name,p in files.items():
        try:
            with Image.open(p) as im: dims[f"{im.width}x{im.height}/{im.mode}"]+=1; im.verify()
            hashes.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(),[]).append(name)
        except Exception as e: invalid.append({"image_name":name,"error":str(e)})
    manifest=build_split(rows,seed)
    ids=sorted(manifest["counts"]); mapping=out/"defect_mapping.template.csv"
    mapping.write_text("DEFECT_ID,defect_name,defect_family,mapping_status,reviewer,reviewed_at,notes\n"+"\n".join(f"{i},,family_{i},pending,,," for i in ids),encoding="utf-8")
    report={"rows":len(rows),"image_files":len(files),"missing":missing,"invalid":invalid,"dimensions":dict(dims),"duplicate_groups":[v for v in hashes.values() if len(v)>1],"class_counts":manifest["counts"],"unknown_protocol":{"calibration":manifest["calibration_unknown"],"test":manifest["test_unknown"]}}
    (out/"audit_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"split_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return report
