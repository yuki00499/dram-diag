import json
import math
from pathlib import Path
import random

import numpy as np

from .data import compute_gray_stats, image_array
from .model import DefectClassifier


def seed_everything(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_split(manifest, fold=None):
    if fold is None:
        raise ValueError("dram-ml-v2 训练必须显式指定 fold")
    folds = {int(item["fold"]): item for item in manifest["cv_folds"]}
    if fold not in folds:
        raise ValueError(f"不存在 fold {fold}")
    selected = folds[fold]
    return list(selected["train"]), list(selected["validation"])


def build_label_graph(items, classes, min_support=5):
    class_to_idx = {value: index for index, value in enumerate(classes)}
    matrix = np.zeros((len(items), len(classes)), dtype=np.float32)
    for row, item in enumerate(items):
        for label in item.get("labels", []):
            matrix[row, class_to_idx[int(label)]] = 1
    cooccurrence = matrix.T @ matrix
    support = np.diag(cooccurrence)
    graph = np.zeros_like(cooccurrence)
    for left in range(len(classes)):
        for right in range(len(classes)):
            if left != right and cooccurrence[left, right] >= min_support:
                graph[left, right] = cooccurrence[left, right] / max(1., math.sqrt(support[left] * support[right]))
    row_sum = graph.sum(axis=1, keepdims=True)
    return np.divide(graph, row_sum, out=np.zeros_like(graph), where=row_sum > 0)


def train_one(config, manifest, seed, output_dir, fold=None):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    import yaml

    seed_everything(seed, torch)
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    classes = sorted(int(value) for value in manifest["types"])
    class_to_idx = {value: index for index, value in enumerate(classes)}
    train_items, validation_items = resolve_split(manifest, fold)
    image_root = Path(config["data_root"]) / "images"
    image_size = tuple(config.get("image_size", [480, 320]))
    stats = compute_gray_stats([image_root / item["image_name"] for item in train_items], image_size)

    class RoiDataset(Dataset):
        def __init__(self, items, training=False):
            self.items = items
            self.training = training
            matrix = np.zeros((len(items), len(classes)), dtype=np.float32)
            for i, item in enumerate(items):
                for label in item.get("labels", []):
                    matrix[i, class_to_idx[int(label)]] = 1.0
            self.labels = matrix

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            item = self.items[index]
            array = image_array(
                image_root / item["image_name"],
                size=image_size,
                training=self.training,
                normalization=config.get("normalization", "imagenet"),
                stats=stats,
            )
            return torch.from_numpy(array), torch.from_numpy(self.labels[index])

    train_set = RoiDataset(train_items, training=True)
    validation_set = RoiDataset(validation_items)
    loss_name = config.get("loss", "ce")
    train_loader = DataLoader(train_set, batch_size=int(config.get("batch_size", 8)), shuffle=True,
                              num_workers=int(config.get("num_workers", 0)))
    validation_loader = DataLoader(validation_set, batch_size=int(config.get("batch_size", 8)), shuffle=False,
                                   num_workers=int(config.get("num_workers", 0)))

    label_graph = None
    if loss_name == "graph_bce":
        label_graph = build_label_graph(train_items, classes, int(config.get("graph_min_support", 5)))
    model = DefectClassifier(len(classes), backbone=config.get("backbone", "resnet18"), pretrained=True,
                             dropout=float(config.get("dropout", 0)),
                             label_graph=label_graph,
                             graph_alpha_init=float(config.get("graph_alpha_init", .1))).to(device)
    freeze_epochs = int(config.get("freeze_epochs", 5))
    if freeze_epochs:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    head_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 0 if freeze_epochs else float(config.get("backbone_lr", 1e-4))},
        {"params": head_parameters, "lr": float(config.get("lr", 3e-4))},
    ], weight_decay=float(config.get("weight_decay", 1e-4)))
    pos_weight = None
    if loss_name in {"weighted_bce", "graph_bce"}:
        positives = torch.from_numpy(train_set.labels.sum(axis=0)).float()
        negatives = len(train_set) - positives
        pos_weight = torch.sqrt(negatives / positives.clamp_min(1)).clamp(
            min=1., max=float(config.get("pos_weight_cap", 5.)))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    epochs = int(config.get("epochs", 60))
    patience = int(config.get("patience", 12))
    min_delta = float(config.get("min_delta", 1e-4))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved = {**config, "seed": seed, "fold": fold, "class_count": len(classes),
                "protocol_version": manifest["protocol_version"], "manifest_fingerprint": manifest["manifest_fingerprint"]}
    (output / "config.yaml").write_text(yaml.safe_dump(resolved, allow_unicode=True, sort_keys=True), encoding="utf-8")

    history = []
    best_f1 = -1.0
    best_row = None
    stale = 0
    for epoch in range(1, epochs + 1):
        if freeze_epochs and epoch == freeze_epochs + 1:
            for parameter in model.encoder.parameters():
                parameter.requires_grad = True
        progress = (epoch - 1) / max(1, epochs - 1)
        cosine = .5 * (1 + math.cos(math.pi * progress))
        optimizer.param_groups[0]["lr"] = 0 if epoch <= freeze_epochs else float(config.get("backbone_lr", 1e-4)) * cosine
        optimizer.param_groups[1]["lr"] = float(config.get("lr", 3e-4)) * cosine
        model.train()
        if epoch <= freeze_epochs:
            model.encoder.eval()
        total = correct = 0
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs["logits"], labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = len(labels)
            total += batch_size
            correct += int((torch.sigmoid(outputs["logits"]) >= .5).eq(labels.bool()).all(dim=1).sum().item())
            loss_sum += float(loss.item()) * batch_size

        model.eval()
        y_true, y_pred = [], []
        validation_loss = 0.0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)["logits"]
                validation_loss += float(criterion(logits, labels).item()) * len(labels)
                y_true.append(labels.cpu().numpy())
                y_pred.append(torch.sigmoid(logits).cpu().numpy())
        from .metrics import multilabel_metrics
        ml = multilabel_metrics(np.concatenate(y_true), np.concatenate(y_pred), classes,
                                core_labels=manifest.get("core_types", classes),
                                tracked_pairs=manifest.get("tracked_pairs", []))
        metrics = {
            "accuracy": ml["exact_match"],
            "macro_f1": ml["core_macro_f1"],
            "all_label_macro_f1": ml["macro_f1"],
            "balanced_accuracy": ml["macro_recall"],
        }
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / total,
            "train_accuracy": correct / total,
            "validation_loss": validation_loss / len(validation_set),
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "validation_all_label_macro_f1": metrics.get("all_label_macro_f1", metrics["macro_f1"]),
            "validation_balanced_accuracy": metrics["balanced_accuracy"],
        }
        history.append(row)
        print(json.dumps({"seed": seed, "fold": fold, **row}, ensure_ascii=False))
        if metrics["macro_f1"] > best_f1 + min_delta:
            best_f1, best_row, stale = metrics["macro_f1"], row, 0
            torch.save({
                "model": model.state_dict(),
                "class_to_idx": class_to_idx,
                "config": resolved,
                "normalization": config.get("normalization", "imagenet"),
                "data_stats": stats,
                "protocol_version": manifest["protocol_version"],
                "manifest_fingerprint": manifest["manifest_fingerprint"],
                "label_graph": label_graph.tolist() if label_graph is not None else None,
                "pos_weight": pos_weight.tolist() if pos_weight is not None else None,
                "best_metrics": row,
            }, output / "best.pt")
        else:
            stale += 1
        (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        if patience and stale >= patience:
            break

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    best_probabilities = []
    with torch.no_grad():
        for images, _ in validation_loader:
            best_probabilities.append(torch.sigmoid(model(images.to(device))["logits"]).cpu().numpy())
    prediction_rows = [
        {"image_name": item["image_name"], "fold": fold, "labels": item["labels"],
         "probabilities": [float(value) for value in probability]}
        for item, probability in zip(validation_items, np.concatenate(best_probabilities))
    ]
    (output / "validation_predictions.json").write_text(
        json.dumps(prediction_rows, ensure_ascii=False, indent=1), encoding="utf-8")

    result = {"seed": seed, "fold": fold, "best_epoch": best_row["epoch"], "best_validation_macro_f1": best_f1,
              "best_validation_all_label_macro_f1": best_row.get("validation_all_label_macro_f1"),
              "best_validation_balanced_accuracy": best_row["validation_balanced_accuracy"],
              "best_validation_accuracy": best_row["validation_accuracy"], "epochs_run": len(history),
              "checkpoint": str(output / "best.pt")}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_experiment(config, manifest, output_dir, seeds, folds=None):
    runs = []
    selected_folds = folds if folds is not None else [None]
    for fold in selected_folds:
        for seed in seeds:
            name = f"seed-{seed}" if fold is None else f"fold-{fold}/seed-{seed}"
            runs.append(train_one(config, manifest, seed, Path(output_dir) / name, fold))
    summary = {
        "experiment": config["experiment"],
        "protocol_version": manifest["protocol_version"],
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "runs": runs,
    }
    for key in ("best_validation_macro_f1", "best_validation_balanced_accuracy", "best_validation_accuracy"):
        values = [run[key] for run in runs]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
        summary[f"{key}_min"] = float(np.min(values))
    summary["selected"] = max(runs, key=lambda item: item["best_validation_macro_f1"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if all(run["fold"] is not None for run in runs):
        from .metrics import multilabel_bootstrap_ci, multilabel_metrics, optimize_multilabel_thresholds
        rows = []
        for run in runs:
            rows.extend(json.loads((Path(run["checkpoint"]).parent / "validation_predictions.json").read_text(encoding="utf-8")))
        seen = set()
        unique_rows = []
        for row in rows:
            if row["image_name"] not in seen:
                unique_rows.append(row)
                seen.add(row["image_name"])
        classes = sorted(int(value) for value in manifest["types"])
        class_to_idx = {value: index for index, value in enumerate(classes)}
        truth = np.zeros((len(unique_rows), len(classes)), dtype=int)
        probabilities = np.asarray([row["probabilities"] for row in unique_rows])
        for row_index, row in enumerate(unique_rows):
            for label in row["labels"]:
                truth[row_index, class_to_idx[int(label)]] = 1
        thresholds, stability = optimize_multilabel_thresholds(
            truth, probabilities, classes, manifest.get("rare_types", []),
            int(config.get("threshold_bootstrap_iterations", 500)), int(config.get("threshold_seed", 42)))
        metrics = multilabel_metrics(truth, probabilities, classes, thresholds,
                                     manifest.get("core_types", classes), manifest.get("tracked_pairs", []))
        fold_metrics = []
        for fold in sorted({int(row["fold"]) for row in unique_rows}):
            indexes = [index for index, row in enumerate(unique_rows) if int(row["fold"]) == fold]
            fold_metrics.append({"fold": fold, **multilabel_metrics(
                truth[indexes], probabilities[indexes], classes, thresholds,
                manifest.get("core_types", classes), manifest.get("tracked_pairs", []))})
        fold_core = [row["core_macro_f1"] for row in fold_metrics]
        ci = multilabel_bootstrap_ci(truth, probabilities, classes, thresholds,
                                     manifest.get("core_types", classes),
                                     int(config.get("metric_bootstrap_iterations", 2000)),
                                     int(config.get("metric_bootstrap_seed", 42)))
        oof = {"protocol_version": manifest["protocol_version"],
               "manifest_fingerprint": manifest["manifest_fingerprint"], "labels": classes,
               "image_names": [row["image_name"] for row in unique_rows],
               "thresholds": {str(label): float(value) for label, value in zip(classes, thresholds)},
               "threshold_stability": stability, "metrics": metrics, "fold_metrics": fold_metrics,
               "fold_core_macro_f1_mean": float(np.mean(fold_core)),
               "fold_core_macro_f1_std": float(np.std(fold_core)), "core_macro_f1_ci": ci}
        (output / "oof_predictions.json").write_text(json.dumps(unique_rows, ensure_ascii=False, indent=1), encoding="utf-8")
        (output / "oof_evaluation.json").write_text(json.dumps(oof, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
