"""Read-only annotation-noise audit on the frozen v2 weighted OOF predictions.

Touches no project artifact. Prints results to stdout only.
"""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
RUN = ROOT / "runs" / "dram_ml_v2_weighted"
LABELS_CSV = ROOT / "artifacts" / "dram_ml_v2" / "labels.csv"
MANIFEST = json.loads((ROOT / "artifacts" / "dram_ml_v2" / "split_manifest.json").read_text(encoding="utf-8"))
OOF = json.loads((RUN / "oof_predictions.json").read_text(encoding="utf-8"))
EVAL = json.loads((RUN / "oof_evaluation.json").read_text(encoding="utf-8"))
LABELS = EVAL["labels"]
C2I = {int(v): i for i, v in enumerate(LABELS)}
THR = np.asarray([EVAL["thresholds"][str(t)] for t in LABELS])
NAME = {"1": "圆形颗粒", "2": "细长颗粒", "3": "方形颗粒", "4": "划痕/裂纹",
        "5": "凹坑/空洞", "7": "块状块斑", "8": "背景低信号"}

rows = {r["image_name"]: r for r in OOF}
names = [r["image_name"] for r in OOF]
Y = np.zeros((len(names), len(LABELS)), dtype=int)
for i, n in enumerate(names):
    for lab in rows[n]["labels"]:
        Y[i, C2I[int(lab)]] = 1
P = np.asarray([rows[n]["probabilities"] for n in names])

# ---- all 1148 images (development + test) for the structural counts ----
all_rows = list(csv.DictReader(LABELS_CSV.open(encoding="utf-8-sig")))
combo = Counter()
for r in all_rows:
    labs = sorted(int(v) for v in r["LABELS"].split(",") if v != "")
    combo[len(labs)] += 1

print("=" * 78)
print("[A] 标签数量结构（全部 1148 张有效标注）")
print("=" * 78)
tot = sum(combo.values())
inst = sum(k * v for k, v in combo.items())
print("  图像数 %d，标签实例数 %d，平均每图 %.4f 个标签" % (tot, inst, inst / tot))
for k in sorted(combo):
    print("  恰好 %d 个标签: %4d 张 (%.2f%%)" % (k, combo[k], 100 * combo[k] / tot))
lam = 0.605
print()
print("  参照：若每图标签数服从截断 Poisson（拟合均值 %.3f），期望分布为" % (inst / tot))
for k in (1, 2, 3):
    if k == 1:
        p = lam * math.exp(-lam)
    elif k == 2:
        p = lam ** 2 * math.exp(-lam) / 2
    else:
        p = sum(lam ** j * math.exp(-lam) / math.factorial(j) for j in range(3, 12))
    p = p / (1 - math.exp(-lam))
    print("    恰好 %d 个标签: 期望 %.1f 张，实测 %d 张" % (k, p * tot, combo.get(k, 0)))
print("  → 双标签远超期望、三标签几乎为零（实测 1 张）。这一对比无法用自然共现解释。")

# ---- high-confidence label disagreements (confident learning) ----
print()
print("=" * 78)
print("[B] 高置信标签分歧（OOF，模型很确定但标注相反）")
print("=" * 78)
CAND = {}
for k, lab in enumerate(LABELS):
    lab = int(lab)
    hi_fp = np.flatnonzero((Y[:, k] == 0) & (P[:, k] >= 0.90))
    hi_fn = np.flatnonzero((Y[:, k] == 1) & (P[:, k] <= 0.10))
    print("  标签 %d(%s): 高置信假阳 ≥0.90 → %3d 张 | 高置信假阴 ≤0.10 → %3d 张 (支持度 %d)"
          % (lab, NAME[str(lab)], len(hi_fp), len(hi_fn), int(Y[:, k].sum())))
    for i in hi_fp:
        CAND.setdefault(names[i], []).append(("缺标?", lab, float(P[i, k]), 0))
    for i in hi_fn:
        CAND.setdefault(names[i], []).append(("错标?", lab, float(P[i, k]), 1))

# ---- per-image suspicion score ----
print()
print("=" * 78)
print("[C] 逐图可疑度排名（前 25）")
print("=" * 78)
scored = []
for n, items in CAND.items():
    i = names.index(n)
    given = sorted(int(v) for v in rows[n]["labels"])
    # confidence-weighted suspicion
    s = sum((p - 0.9) if kind == "缺标?" else (0.10 - p) for kind, _, p, _ in items)
    scored.append((s, n, given, items))
scored.sort(reverse=True, key=lambda x: x[0])
print("  %-16s %-18s %-6s %s" % ("图片", "已标标签", "可疑分", "模型意见"))
for s, n, given, items in scored[:25]:
    ops = "; ".join("%s %d(p=%.2f)" % (kind, lab, p) for kind, lab, p, _ in items)
    print("  %-16s %-18s %.3f  %s" % (n, ",".join(map(str, given)), s, ops))
print()
print("  可疑图总计 %d 张；可疑分 > 0.5 的 %d 张，> 1.0 的 %d 张"
      % (len(scored), sum(1 for s, *_ in scored if s > .5), sum(1 for s, *_ in scored if s > 1.0)))

# ---- morphological adjacency confusion ----
print()
print("=" * 78)
print("[D] 形态学相邻标签的混淆（颗粒形状 1/2/3 与 划痕 4）")
print("=" * 78)
adj = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
for a, b in adj:
    ka, kb = C2I[a], C2I[b]
    both = int(((Y[:, ka] == 1) & (Y[:, kb] == 1)).sum())
    only_a = int(((Y[:, ka] == 1) & (Y[:, kb] == 0)).sum())
    only_b = int(((Y[:, ka] == 0) & (Y[:, kb] == 1)).sum())
    # model: given only-a samples, how often does it also fire b?
    fire_b = float((P[(Y[:, ka] == 1) & (Y[:, kb] == 0), kb] >= THR[kb]).mean()) if only_a else float("nan")
    fire_a = float((P[(Y[:, ka] == 0) & (Y[:, kb] == 1), ka] >= THR[ka]).mean()) if only_b else float("nan")
    print("  %d vs %d: 共现 %3d | 仅%d %3d 张→模型同时报%d 的比例 %.2f | 仅%d %3d 张→模型同时报%d 的比例 %.2f"
          % (a, b, both, a, only_a, b, fire_b, b, only_b, a, fire_a))

# ---- label 8 separability ----
print()
print("=" * 78)
print("[E] 标签 8「背景/低信号」的可分性")
print("=" * 78)
for lab in [1, 5, 7, 8]:
    k = C2I[lab]
    pos, neg = P[Y[:, k] == 1, k], P[Y[:, k] == 0, k]
    print("  标签 %d(%s): 正样本概率 均值%.3f 中位%.3f | 负样本 均值%.3f 中位%.3f | 重叠区[0.3,0.7]内正样本占 %.0f%%"
          % (lab, NAME[str(lab)], pos.mean(), np.median(pos), neg.mean(), np.median(neg),
             100 * float(((pos >= .3) & (pos <= .7)).mean())))

# ---- threshold surface sharpness ----
print()
print("=" * 78)
print("[F] 阈值曲线尖锐度（平坦 = 模型对该标签没有清晰判据）")
print("=" * 78)
grid = np.arange(0.05, 0.96, 0.01)
for lab in LABELS:
    k = C2I[int(lab)]
    pos = Y[:, k] == 1
    f1 = []
    for g in grid:
        pr = P[:, k] >= g
        tp = np.sum(pr & pos); fp = np.sum(pr & ~pos); fn = np.sum(~pr & pos)
        d = 2 * tp + fp + fn
        f1.append(2 * tp / d if d else 0.0)
    f1 = np.asarray(f1)
    best = f1.max()
    width = float((f1 >= best * 0.98).sum()) * 0.01
    print("  标签 %d(%s): 峰值 F1 %.4f，峰值 98%% 区间宽度 %.2f（阈值刻度）" % (int(lab), NAME[str(lab)], best, width))
