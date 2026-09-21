"""Read-only audit probe: (a) the ensemble-on-development trap, (b) reproduction of the frozen
test report, (c) distribution shift between the calibration population and the deployment population.

Touches no project artifact. Prints measurements to stdout only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))
from dram_diag.model import DefectClassifier  # noqa: E402

ROOT = Path(__file__).parent
RUN = ROOT / "runs" / "dram_ml_v2_weighted"
MANIFEST = json.loads((ROOT / "artifacts" / "dram_ml_v2" / "split_manifest.json").read_text(encoding="utf-8"))
OOF_EVAL = json.loads((RUN / "oof_evaluation.json").read_text(encoding="utf-8"))
TEST_REPORT = json.loads((ROOT / "artifacts" / "dram_ml_v2" / "test_report.json").read_text(encoding="utf-8"))
DEPLOY = json.loads((ROOT / "artifacts" / "deployment-v2.json").read_text(encoding="utf-8"))
IMAGE_ROOT = ROOT / "晶圆缺陷分类数据集" / "images"
LABELS = OOF_EVAL["labels"]
C2I = {int(v): i for i, v in enumerate(LABELS)}
CORE = [2, 3, 4, 5]
CORE_COLS = [C2I[t] for t in CORE]
DEPLOY_THR = np.asarray([DEPLOY["thresholds"][str(t)] for t in LABELS])


def letterbox(image, size=(480, 320)):
    tw, th = size
    image = image.convert("L")
    scale = min(tw / image.width, th / image.height)
    w, h = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    canvas = Image.new("L", (tw, th), 0)
    canvas.paste(image.resize((w, h), Image.Resampling.BILINEAR), ((tw - w) // 2, (th - h) // 2))
    return canvas


def load_models(device):
    models = []
    first_stats = None
    for fold in range(5):
        ckpt = torch.load(RUN / f"fold-{fold}" / "seed-42" / "best.pt", map_location="cpu", weights_only=False)
        if first_stats is None:
            first_stats = ckpt["data_stats"]
        cfg = ckpt["config"]
        m = DefectClassifier(len(LABELS), cfg.get("backbone", "resnet18"), pretrained=False,
                             dropout=float(cfg.get("dropout", 0)), label_graph=ckpt.get("label_graph"),
                             graph_alpha_init=float(cfg.get("graph_alpha_init", .1)))
        m.load_state_dict({k: v for k, v in ckpt["model"].items() if not k.startswith("projection.")}, strict=False)
        models.append(m.to(device).eval())
    # Deployment inference intentionally normalizes every ensemble member with
    # the first checkpoint's statistics; reproduce that exact frozen behavior.
    return models, first_stats


def ensure_baseline(models, stats, items, device):
    arrays = []
    for it in items:
        with Image.open(IMAGE_ROOT / it["image_name"]) as src:
            g = np.asarray(letterbox(src), dtype=np.float32) / 255.0
        arrays.append((np.repeat(g[None, ...], 3, axis=0) - float(stats["mean"])) / float(stats["std"]))
    per_fold = []
    with torch.no_grad():
        for s in range(0, len(arrays), 8):
            batch = torch.from_numpy(np.stack(arrays[s:s + 8])).to(device)
            per_fold.append(torch.stack([torch.sigmoid(m(batch)["logits"]) for m in models]).float().cpu().numpy())
    per_fold = np.concatenate(per_fold, axis=1)  # (5, n, 7)
    return per_fold, per_fold.mean(0)


def matrix(items):
    y = np.zeros((len(items), len(LABELS)), dtype=int)
    for r, it in enumerate(items):
        for lab in it["labels"]:
            y[r, C2I[int(lab)]] = 1
    return y


def core(prob, y, thr):
    ys = y[:, CORE_COLS].astype(bool)
    pr = prob[:, CORE_COLS] >= np.asarray(thr)[CORE_COLS][None, :]
    tp = np.sum(ys & pr, 0); fp = np.sum(~ys & pr, 0); fn = np.sum(ys & ~pr, 0)
    d = 2 * tp + fp + fn
    return float(np.mean(np.divide(2 * tp, d, out=np.zeros(len(CORE)), where=d > 0)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading 5 fold models ...", flush=True)
    models, stats = load_models(device)
    dev_items = MANIFEST["splits"]["development"]
    test_items = MANIFEST["splits"]["test_known"]

    dev_pf, dev_ens = ensure_baseline(models, stats, dev_items, device)
    y_dev = matrix(dev_items)
    # fold assignment for development images
    fold_of = {}
    for f in MANIFEST["cv_folds"]:
        for it in f["validation"]:
            fold_of[it["image_name"]] = int(f["fold"])
    dev_folds = np.asarray([fold_of[it["image_name"]] for it in dev_items])
    dev_oof = np.asarray([dev_pf[dev_folds[i], i] for i in range(len(dev_items))])

    print("\n" + "=" * 78)
    print("[A] the ensemble-on-development trap (plan line 94 warns about this)")
    print("=" * 78)
    print("   single-fold OOF probabilities @ deployed thresholds  = %.6f   <- the honest baseline"
          % core(dev_oof, y_dev, DEPLOY_THR))
    print("   5-fold ensemble on the SAME development images       = %.6f   <- 4/5 models memorised each image"
          % core(dev_ens, y_dev, DEPLOY_THR))
    print("   -> fake 'improvement' of %+.6f if this shortcut is used" % (core(dev_ens, y_dev, DEPLOY_THR)
                                                                         - core(dev_oof, y_dev, DEPLOY_THR)))
    print("   identity-view reproduction check vs frozen oof_evaluation.json:")
    frozen_thr = np.asarray([OOF_EVAL["thresholds"][str(t)] for t in LABELS])
    print("     recomputed single-fold OOF @ frozen thresholds     = %.6f" % core(dev_oof, y_dev, frozen_thr))
    print("     frozen oof_evaluation.json core_macro_f1           = %.7f" % OOF_EVAL["metrics"]["core_macro_f1"])

    print("\n" + "=" * 78)
    print("[B] reproduction of the frozen test report (read-only verification)")
    print("=" * 78)
    test_pf, test_ens = ensure_baseline(models, stats, test_items, device)
    y_test = matrix(test_items)
    print("   recomputed 5-fold ensemble on 115 test images @ deployed thresholds = %.7f"
          % core(test_ens, y_test, DEPLOY_THR))
    print("   frozen test_report.json core_macro_f1                              = %.7f"
          % TEST_REPORT["metrics"]["core_macro_f1"])
    single_mean = test_pf.mean(0)
    print("   (independent check: mean of per-fold probabilities = %.7f)" % core(single_mean, y_test, DEPLOY_THR))

    print("\n" + "=" * 78)
    print("[C] distribution shift: calibration population (single-fold OOF, n=1033)")
    print("    vs deployment population (5-fold ensemble, n=115 test)")
    print("=" * 78)
    print("   %-6s %-22s %-22s %-12s" % ("label", "OOF single-fold", "ensemble(test)", "mean shift"))
    for t in LABELS:
        c = C2I[t]
        print("   %-6d %.4f / %.4f        %.4f / %.4f        %+.4f"
              % (t, dev_oof[:, c].mean(), dev_oof[:, c].std(),
                 test_ens[:, c].mean(), test_ens[:, c].std(),
                 test_ens[:, c].mean() - dev_oof[:, c].mean()))

    print("\n[C2] same-model compression: per-fold model vs 5-fold mean, on test images")
    for t in LABELS:
        c = C2I[t]
        print("   label %-2d per-fold std %.4f -> ensemble std %.4f  (ratio %.3f)"
              % (t, test_pf[:, :, c].reshape(-1).std(), test_ens[:, c].std(),
                 test_ens[:, c].std() / max(1e-9, test_pf[:, :, c].reshape(-1).std())))

    print("\n" + "=" * 78)
    print("[D] how far is the deployed threshold from the ensemble's own optimum on test?")
    print("=" * 78)
    grid = np.linspace(0.05, 0.95, 91)
    ys = y_test[:, CORE_COLS].astype(bool)
    for t in CORE:
        col = C2I[t]
        pos = y_test[:, col] == 1
        best, bg = -1, None
        for g in grid:
            pr = test_ens[:, col] >= g
            tp = np.sum(pr & pos); fp = np.sum(pr & ~pos); fn = np.sum(~pr & pos)
            d = 2 * tp + fp + fn
            v = 2 * tp / d if d else 0.0
            if v > best:
                best, bg = v, g
        pr = test_ens[:, col] >= DEPLOY_THR[col]
        tp = np.sum(pr & pos); fp = np.sum(pr & ~pos); fn = np.sum(~pr & pos)
        d = 2 * tp + fp + fn
        cur = 2 * tp / d if d else 0.0
        print("   label %d: deployed thr %.2f -> F1 %.4f | test-optimal thr %.2f -> F1 %.4f | support %d"
              % (t, DEPLOY_THR[col], cur, bg, best, int(pos.sum())))


if __name__ == "__main__":
    main()
