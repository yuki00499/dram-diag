from collections import defaultdict
import json, random

def build_split(rows, seed=42):
    by=defaultdict(list)
    for r in rows: by[int(r["DEFECT_ID"])].append(r["IMAGE_NAME"])
    counts={k:len(v) for k,v in by.items()}
    high=sorted(k for k,n in counts.items() if n>20); mid=sorted(k for k,n in counts.items() if 5<=n<=20); tail=sorted(k for k,n in counts.items() if n<5)
    if len(high)<2 or len(mid)<2 or len(tail)<1: raise ValueError("高频/中频/长尾类别不足，无法按协议留出 5 类")
    unknown=[high[0],high[1],mid[0],mid[1],tail[0]]; rng=random.Random(seed); rng.shuffle(unknown)
    calib_unknown, test_unknown=unknown[:2],unknown[2:]
    split={k:[] for k in ("train","validation","calibration_known","test_known","calibration_unknown","test_unknown")}
    for cls, imgs0 in by.items():
        imgs=list(imgs0); rng.shuffle(imgs)
        if cls in calib_unknown: split["calibration_unknown"] += [{"image_name":x,"defect_id":cls} for x in imgs]; continue
        if cls in test_unknown: split["test_unknown"] += [{"image_name":x,"defect_id":cls} for x in imgs]; continue
        n=len(imgs)
        if n<5:
            split["train"] += [{"image_name":x,"defect_id":cls,"review_required":True} for x in imgs]; continue
        # Four non-empty partitions are only possible with at least five samples.
        a=max(2, int(n*.7)); b=a+max(1, int(n*.1)); c=b+max(1, int(n*.1))
        while c >= n: a -= 1; b -= 1; c -= 1
        for key,part in (("train",imgs[:a]),("validation",imgs[a:b]),("calibration_known",imgs[b:c]),("test_known",imgs[c:])):
            split[key] += [{"image_name":x,"defect_id":cls} for x in part]
    return {"seed":seed,"counts":counts,"high_frequency":high,"mid_frequency":mid,"long_tail":tail,"calibration_unknown":calib_unknown,"test_unknown":test_unknown,"splits":split}
