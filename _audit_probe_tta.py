"""Read-only audit probe v2: E0 TTA + resolution + invariance on frozen v2 weighted checkpoints.

Touches no project artifact. Prints measurements to stdout only.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))
from dram_diag.model import DefectClassifier  # noqa: E402

ROOT = Path(__file__).parent
RUN = ROOT / "runs" / "dram_ml_v2_weighted"
MANIFEST = json.loads((ROOT / "artifacts" / "dram_ml_v2" / "split_manifest.json").read_text(encoding="utf-8"))
OOF_EVAL = json.loads((RUN / "oof_evaluation.json").read_text(encoding="utf-8"))
IMAGE_ROOT = ROOT / "晶圆缺陷分类数据集" / "images"
LABELS = OOF_EVAL["labels"]
C2I = {int(v): i for i, v in enumerate(LABELS)}
CORE = [2, 3, 4, 5]
CORE_COLS = [C2I[t] for t in CORE]
BASE_THR = np.asarray([OOF_EVAL["thresholds"][str(t)] for t in LABELS])
GRID = np.linspace(0.05, 0.95, 91)
BASE_CORE = 0.9231337079460955
GATE = BASE_CORE + 0.005


def letterbox(image, size):
    tw, th = size
    image = image.convert("L")
    scale = min(tw / image.width, th / image.height)
    w, h = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    resized = image.resize((w, h), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (tw, th), 0)
    canvas.paste(resized, ((tw - w) // 2, (th - h) // 2))
    return canvas


def view_image(image, view):
    if view == "identity":
        return image
    if view == "hflip":
        return ImageOps.mirror(image)
    if view == "rot+2":
        return image.rotate(2.0, resample=Image.Resampling.BILINEAR, fillcolor=0)
    if view == "rot-2":
        return image.rotate(-2.0, resample=Image.Resampling.BILINEAR, fillcolor=0)
    if view == "rot+5":
        return image.rotate(5.0, resample=Image.Resampling.BILINEAR, fillcolor=0)
    if view == "zoom1.15":
        w, h = image.size
        cw, ch = round(w / 1.15), round(h / 1.15)
        left, top = (w - cw) // 2, (h - ch) // 2
        return image.crop((left, top, left + cw, top + ch)).resize((w, h), Image.Resampling.BILINEAR)
    if view == "up1.5x":
        return letterbox(image, (720, 480))
    if view == "up2x":
        return letterbox(image, (960, 640))
    raise ValueError(view)


def to_array(image, stats):
    gray = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.repeat(gray[None, ...], 3, axis=0)
    return (arr - float(stats["mean"])) / float(stats["std"])


def load_view(path, size, view, stats):
    with Image.open(path) as source:
        base = letterbox(source, size)
    return to_array(view_image(base, view), stats)


VIEWS = ["identity", "hflip", "rot+2", "rot-2", "zoom1.15", "rot+5", "up1.5x", "up2x"]


def collect():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = sorted(int(v) for v in MANIFEST["types"])
    records, latency = {}, {}
    for fold in range(5):
        ckpt = torch.load(RUN / f"fold-{fold}" / "seed-42" / "best.pt", map_location="cpu", weights_only=False)
        cfg, stats = ckpt["config"], ckpt["data_stats"]
        size = tuple(cfg.get("image_size", [480, 320]))
        model = DefectClassifier(len(classes), cfg.get("backbone", "resnet18"), pretrained=False,
                                 dropout=float(cfg.get("dropout", 0)), label_graph=ckpt.get("label_graph"),
                                 graph_alpha_init=float(cfg.get("graph_alpha_init", .1)))
        model.load_state_dict({k: v for k, v in ckpt["model"].items() if not k.startswith("projection.")},
                              strict=False)
        model = model.to(device).eval()
        items = [it for f in MANIFEST["cv_folds"] if int(f["fold"]) == fold for it in f["validation"]]
        for view in VIEWS:
            t0 = time.perf_counter()
            arrays = [load_view(IMAGE_ROOT / it["image_name"], size, view, stats) for it in items]
            probs, logits = [], []
            with torch.no_grad():
                for s in range(0, len(arrays), 8):
                    batch = torch.from_numpy(np.stack(arrays[s:s + 8])).to(device)
                    out = model(batch)["logits"]
                    logits.append(out.float().cpu().numpy())
                    probs.append(torch.sigmoid(out).float().cpu().numpy())
            probs, logits = np.concatenate(probs), np.concatenate(logits)
            latency.setdefault(view, []).append((time.perf_counter() - t0) / len(items) * 1000)
            for it, p, lg in zip(items, probs, logits):
                records.setdefault(it["image_name"], {"fold": fold, "labels": it["labels"], "views": {}})
                records[it["image_name"]]["views"][view] = (p, lg)
        print(f"  fold {fold} done", flush=True)
    names = list(records)
    y = np.zeros((len(names), len(LABELS)), dtype=int)
    for r, n in enumerate(names):
        for lab in records[n]["labels"]:
            y[r, C2I[int(lab)]] = 1
    folds = np.asarray([records[n]["fold"] for n in names], dtype=int)
    P = {v: np.asarray([records[n]["views"][v][0] for n in names]) for v in VIEWS}
    L = {v: np.asarray([records[n]["views"][v][1] for n in names]) for v in VIEWS}
    return names, y, folds, P, L, latency


def show(names, y, folds, P, L, latency):
    ys_core = y[:, CORE_COLS].astype(bool)

    def core_f1(pred, idx=None):
        yy = ys_core if idx is None else ys_core[idx]
        tp = np.sum(yy & pred, 0); fp = np.sum(~yy & pred, 0); fn = np.sum(yy & ~pred, 0)
        d = 2 * tp + fp + fn
        return float(np.mean(np.divide(2 * tp, d, out=np.zeros(len(CORE)), where=d > 0)))

    def core_at(prob, thr=None, idx=None):
        thr = BASE_THR if thr is None else thr
        pr = prob[:, CORE_COLS] >= np.asarray(thr)[CORE_COLS][None, :]
        return core_f1(pr if idx is None else pr[idx], idx)

    def per_label(prob, thr=None):
        thr = BASE_THR if thr is None else thr
        out = {}
        for col, lab in enumerate(LABELS):
            pr = prob[:, col] >= thr[col]; pos = y[:, col] == 1
            tp = np.sum(pr & pos); fp = np.sum(pr & ~pos); fn = np.sum(~pr & pos)
            d = 2 * tp + fp + fn
            out[int(lab)] = (tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0,
                             2 * tp / d if d else 0.0)
        return out

    def fit_thresholds(prob, idx):
        thr = []
        for col in range(len(LABELS)):
            t = y[idx, col] == 1; s = prob[idx, col]
            pred = s[:, None] >= GRID[None, :]; pos = t[:, None]
            tp = np.sum(pred & pos, 0); fp = np.sum(pred & ~pos, 0); fn = np.sum(~pred & pos, 0)
            d = 2 * tp + fp + fn
            val = np.divide(2 * tp, d, out=np.zeros(len(GRID)), where=d > 0)
            b = val.max()
            thr.append(min([float(g) for g, x in zip(GRID, val) if x == b], key=lambda z: (abs(z - .5), z)))
        return np.asarray(thr)

    def paired(a, b, iters=4000):
        groups = {}
        for i, row in enumerate(y):
            groups.setdefault(tuple(row.tolist()), []).append(i)
        groups = list(groups.values())
        rng = np.random.default_rng(42)
        out = []
        for _ in range(iters):
            idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
            out.append(core_at(a, idx=idx) - core_at(b, idx=idx))
        return np.asarray(out)

    combos = {
        "plan E0: 5 views (id+flip+rot2+zoom)": ["identity", "hflip", "rot+2", "rot-2", "zoom1.15"],
        "rot only (id+rot2)": ["identity", "rot+2", "rot-2"],
        "rot only, wider (id+rot5)": ["identity", "rot+5"],
        "id+rot2+hflip": ["identity", "rot+2", "rot-2", "hflip"],
        "id+rot2+zoom": ["identity", "rot+2", "rot-2", "zoom1.15"],
        "id+rot2+up2x": ["identity", "rot+2", "rot-2", "up2x"],
    }
    avg = {k: np.mean([P[v] for v in vs], 0) for k, vs in combos.items()}
    avg_logit = {k: 1 / (1 + np.exp(-np.mean([L[v] for v in vs], 0))) for k, vs in combos.items()}

    print("\n[A] single-view core Macro-F1 @ FIXED baseline thresholds")
    for v in VIEWS:
        print("   %-10s %.6f  (%+.6f)" % (v, core_at(P[v]), core_at(P[v]) - BASE_CORE))

    print("\n[B] multi-view combinations @ FIXED baseline thresholds")
    print("   %-40s %-11s %-11s" % ("combination", "mean(sigmoid)", "mean(logit)"))
    for k in combos:
        print("   %-40s %.6f    %.6f" % (k, core_at(avg[k]), core_at(avg_logit[k])))

    print("\n[C] with thresholds re-fitted on each TTA variant (the plan's 'deployment candidate')")
    ref = fit_thresholds(P["identity"], np.arange(len(names)))
    print("   %-40s %-10s %s" % ("variant", "core F1", "fitted thresholds"))
    print("   %-40s %.6f   %s" % ("identity (reference)", core_at(P["identity"], ref), np.round(ref, 3).tolist()))
    for k in combos:
        t = fit_thresholds(avg[k], np.arange(len(names)))
        print("   %-40s %.6f   %s" % (k, core_at(avg[k], t), np.round(t, 3).tolist()))

    print("\n[D] per-fold core Macro-F1 @ fixed thresholds (identity -> plan E0 TTA)")
    for f in range(5):
        idx = np.flatnonzero(folds == f)
        a, b = core_at(P["identity"], idx=idx), core_at(avg["plan E0: 5 views (id+flip+rot2+zoom)"], idx=idx)
        print("   fold %d: identity %.4f  TTA5 %.4f  diff %+.4f" % (f, a, b, b - a))

    print("\n[E] precision/recall decomposition @ fixed thresholds, plan E0 TTA")
    base, tta = per_label(P["identity"]), per_label(avg["plan E0: 5 views (id+flip+rot2+zoom)"])
    for t in CORE:
        p0, r0, f0 = base[t]; p1, r1, f1 = tta[t]
        print("   label %d: P %+.4f   R %+.4f   F1 %+.4f" % (t, p1 - p0, r1 - r0, f1 - f0))

    print("\n[F] paired bootstrap vs identity @ fixed thresholds (4000 iter)")
    for k in ["plan E0: 5 views (id+flip+rot2+zoom)", "rot only (id+rot2)", "id+rot2+up2x"]:
        d = paired(avg[k], P["identity"])
        print("   %-40s mean %+.6f  CI [%+.6f, %+.6f]  SE %.6f  P(diff>=+0.005)=%.3f"
              % (k, d.mean(), np.quantile(d, .025), np.quantile(d, .975), d.std(), float(np.mean(d >= .005))))

    print("\n[G] transformation invariance of the frozen model (mean |dP| vs identity, all 7 labels)")
    for v in ["hflip", "rot+2", "rot-2", "rot+5", "zoom1.15", "up1.5x", "up2x"]:
        dp = np.abs(P[v] - P["identity"])
        flip = np.mean((P[v] >= BASE_THR[None, :]) != (P["identity"] >= BASE_THR[None, :]))
        print("   %-10s mean|dP|=%.4f  max|dP|=%.4f  decision-flip rate=%.4f" % (v, dp.mean(), dp.max(), flip))

    print("\n[H] low-confidence rate (min |prob-threshold| <= 0.1)")
    for k, p in [("identity", P["identity"]),
                 ("plan E0 TTA5", avg["plan E0: 5 views (id+flip+rot2+zoom)"]),
                 ("rot only", avg["rot only (id+rot2)"])]:
        flag = np.any(np.abs(p - BASE_THR[None, :]) <= 0.1, axis=1)
        print("   %-16s %.4f  (%d/%d)" % (k, flag.mean(), flag.sum(), len(flag)))

    print("\n[I] inference latency, ms/image @ batch 8")
    for v in VIEWS:
        print("   %-10s %.1f ms" % (v, float(np.mean(latency[v]))))

    print("\n[J] gate check: target %.7f" % GATE)
    for k in combos:
        v1 = core_at(avg[k]); t = fit_thresholds(avg[k], np.arange(len(names))); v2 = core_at(avg[k], t)
        print("   %-40s fixed-thr %.6f %s | refit-thr %.6f %s"
              % (k, v1, "PASS" if v1 >= GATE else "fail", v2, "PASS" if v2 >= GATE else "fail"))


if __name__ == "__main__":
    show(*collect())
