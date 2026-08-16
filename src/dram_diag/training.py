from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random

import numpy as np

from .data import compute_gray_stats, image_array, load_labels
from .model import DefectClassifier, batch_prototype_loss, supervised_contrastive_loss
from .protocol import validate_manifest


def seed_everything(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def classification_metrics(y_true, y_pred, labels):
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }


class ClassBalancedBatchSampler:
    def __init__(self, labels, classes_per_batch, samples_per_class, batches, seed):
        self.by_class = defaultdict(list)
        for index, label in enumerate(labels):
            self.by_class[int(label)].append(index)
        self.classes = sorted(self.by_class)
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.batches = batches
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.batches):
            chosen = rng.sample(self.classes, min(self.classes_per_batch, len(self.classes)))
            batch = []
            for label in chosen:
                indexes = self.by_class[label]
                if len(indexes) >= self.samples_per_class:
                    batch.extend(rng.sample(indexes, self.samples_per_class))
                else:
                    batch.extend(rng.choices(indexes, k=self.samples_per_class))
            rng.shuffle(batch)
            yield batch


def resolve_split(manifest, fold=None):
    if fold is None:
        return list(manifest["splits"]["train"]), list(manifest["splits"]["validation"])
    folds = {int(item["fold"]): item["validation"] for item in manifest["cv_folds"]}
    if fold not in folds:
        raise ValueError(f"不存在 fold {fold}")
    validation = list(folds[fold])
    validation_names = {item["image_name"] for item in validation}
    development = manifest["splits"]["train"] + manifest["splits"]["validation"]
    train = [item for item in development if item["image_name"] not in validation_names]
    return train, validation


def train_one(config, manifest, seed, output_dir, fold=None):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    import yaml

    seed_everything(seed, torch)
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    multilabel = bool(config.get("multilabel", False))
    if multilabel:
        classes = sorted(int(value) for value in manifest["types"])
    else:
        classes = sorted(int(value) for value in manifest["classification_classes"])
    class_to_idx = {value: index for index, value in enumerate(classes)}
    train_items, validation_items = resolve_split(manifest, fold)
    image_root = Path(config["data_root"]) / "images"
    image_size = tuple(config.get("image_size", [480, 320]))
    stats = compute_gray_stats([image_root / item["image_name"] for item in train_items], image_size)

    class RoiDataset(Dataset):
        def __init__(self, items, training=False):
            self.items = items
            self.training = training
            if multilabel:
                matrix = np.zeros((len(items), len(classes)), dtype=np.float32)
                for i, item in enumerate(items):
                    for label in item.get("labels", []):
                        matrix[i, class_to_idx[int(label)]] = 1.0
                self.labels = matrix
            else:
                self.labels = [class_to_idx[int(item["defect_id"])] for item in items]

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
            if multilabel:
                return torch.from_numpy(array), torch.from_numpy(self.labels[index])
            return torch.from_numpy(array), self.labels[index]

    train_set = RoiDataset(train_items, training=True)
    validation_set = RoiDataset(validation_items)
    loss_name = config.get("loss", "ce")
    balanced_batches = config.get("balanced_sampling", loss_name in {"ce_proto", "ce_proto_supcon"}) and not multilabel
    if balanced_batches:
        classes_per_batch = int(config.get("classes_per_batch", 7))
        samples_per_class = int(config.get("samples_per_class", 3))
        batches = math.ceil(len(train_set) / (classes_per_batch * samples_per_class))
        sampler = ClassBalancedBatchSampler(train_set.labels, classes_per_batch, samples_per_class, batches, seed)
        train_loader = DataLoader(train_set, batch_sampler=sampler, num_workers=int(config.get("num_workers", 0)))
    else:
        train_loader = DataLoader(train_set, batch_size=int(config.get("batch_size", 8)), shuffle=True,
                                  num_workers=int(config.get("num_workers", 0)))
    validation_loader = DataLoader(validation_set, batch_size=int(config.get("batch_size", 8)), shuffle=False,
                                   num_workers=int(config.get("num_workers", 0)))

    projection_dim = int(config.get("projection_dim", 128)) if loss_name == "ce_proto_supcon" else 0
    model = DefectClassifier(len(classes), backbone=config.get("backbone", "resnet18"), pretrained=True,
                             dropout=float(config.get("dropout", 0)), projection_dim=projection_dim).to(device)
    freeze_epochs = int(config.get("freeze_epochs", 5))
    if freeze_epochs:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
    head_parameters = list(model.classifier.parameters())
    if model.projection is not None:
        head_parameters += list(model.projection.parameters())
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 0 if freeze_epochs else float(config.get("backbone_lr", 1e-4))},
        {"params": head_parameters, "lr": float(config.get("lr", 3e-4))},
    ], weight_decay=float(config.get("weight_decay", 1e-4)))
    counts = Counter(int(item["defect_id"]) for item in train_items) if not multilabel else Counter()
    class_weights = torch.tensor([1 / math.sqrt(counts[key]) for key in classes], dtype=torch.float32) if counts else None
    if class_weights is not None:
        class_weights /= class_weights.mean()
    if multilabel:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if loss_name == "weighted_ce" else None,
            label_smoothing=float(config.get("label_smoothing", 0)),
        )
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
        loss_sum = ce_sum = prototype_sum = supcon_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
                ce = criterion(outputs["logits"], labels)
                prototype = batch_prototype_loss(outputs["embedding"], labels, float(config.get("prototype_margin", .2))) if loss_name in {"ce_proto", "ce_proto_supcon"} else ce * 0
                supcon = supervised_contrastive_loss(outputs["projection"], labels, float(config.get("supcon_temperature", .1))) if loss_name == "ce_proto_supcon" else ce * 0
                loss = ce + float(config.get("lambda_proto", 0)) * prototype + float(config.get("lambda_supcon", 0)) * supcon
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = len(labels)
            total += batch_size
            if multilabel:
                correct += int((torch.sigmoid(outputs["logits"]) >= .5).eq(labels.bool()).all(dim=1).sum().item())
            else:
                correct += int((outputs["logits"].argmax(1) == labels).sum().item())
            loss_sum += float(loss.item()) * batch_size
            ce_sum += float(ce.item()) * batch_size
            prototype_sum += float(prototype.item()) * batch_size
            supcon_sum += float(supcon.item()) * batch_size

        model.eval()
        y_true, y_pred = [], []
        validation_loss = 0.0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)["logits"]
                validation_loss += float(criterion(logits, labels).item()) * len(labels)
                if multilabel:
                    y_true.append(labels.cpu().numpy())
                    y_pred.append(torch.sigmoid(logits).cpu().numpy())
                else:
                    y_true.extend(labels.cpu().tolist())
                    y_pred.extend(logits.argmax(1).cpu().tolist())
        if multilabel:
            from .metrics import multilabel_metrics
            ml = multilabel_metrics(np.concatenate(y_true), np.concatenate(y_pred), classes)
            metrics = {
                "accuracy": ml["exact_match"],
                "macro_f1": ml["macro_f1"],
                "balanced_accuracy": ml["macro_recall"],
                "exact_match": ml["exact_match"],
                "macro_recall": ml["macro_recall"],
            }
        else:
            metrics = classification_metrics(y_true, y_pred, list(range(len(classes))))
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / total,
            "train_ce": ce_sum / total,
            "train_prototype": prototype_sum / total,
            "train_supcon": supcon_sum / total,
            "train_accuracy": correct / total,
            "validation_loss": validation_loss / len(validation_set),
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
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
                "best_metrics": row,
            }, output / "best.pt")
        else:
            stale += 1
        (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        if patience and stale >= patience:
            break

    result = {"seed": seed, "fold": fold, "best_epoch": best_row["epoch"], "best_validation_macro_f1": best_f1,
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
    return summary
