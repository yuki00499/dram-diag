"""Ultralytics integration for joint dram-det-v3 detection and diagnosis training."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch

from .detection import GlobalDiagnosisHead, hierarchical_losses


def _raw_predictions(predictions: Any) -> dict[str, torch.Tensor]:
    """Normalize YOLO26 one-to-many output to ``boxes/scores/feats``."""
    if isinstance(predictions, tuple):
        predictions = predictions[1]
    if isinstance(predictions, dict) and "one2many" in predictions:
        predictions = predictions["one2many"]
    if not isinstance(predictions, dict) or not {"scores", "feats"}.issubset(predictions):
        raise RuntimeError("YOLO26 输出不包含训练层级诊断所需的 scores/feats")
    return predictions


def _feature_channels(model) -> list[int]:
    detector = model.model[-1]
    channels = []
    for branch in detector.one2many["box_head"]:
        first = branch[0]
        convolution = getattr(first, "conv", first)
        channels.append(int(convolution.in_channels))
    return channels


def _letterbox_sem(path: str | Path, size: int, augment: bool = False) -> torch.Tensor:
    """Read a grayscale SEM image and letterbox it without losing whole-image context."""
    with Image.open(path) as source:
        image = source.convert("L")
    if augment:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(.92, 1.08))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(.92, 1.08))
        if random.random() < .1:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(.1, .6)))
        if random.random() < .5:
            image = image.rotate(random.uniform(-5, 5), resample=Image.Resampling.BILINEAR, fillcolor=114)
    scale = min(size / image.width, size / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("L", (size, size), color=114)
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    array = np.asarray(canvas, dtype=np.uint8)
    return torch.from_numpy(np.repeat(array[None, ...], 3, axis=0).copy())


class GlobalImageStream:
    """Balanced cyclic stream that includes unusable and ignored images for global supervision."""

    def __init__(self, global_labels_path: str | Path, class_ids: list[int], quality_ids: list[str],
                 imgsz: int, seed: int):
        self.path = Path(global_labels_path)
        self.root = self.path.parent / "images" / "global" / "train"
        self.records = json.loads(self.path.read_text(encoding="utf-8"))
        self.names = sorted(name for name, record in self.records.items() if record.get("split") == "train")
        if not self.names:
            raise ValueError("global_labels.json 没有 train 图像")
        missing = [name for name in self.names if not (self.root / name).exists()]
        if missing:
            raise FileNotFoundError(f"全局训练图像缺失: {missing[:3]}")
        self.class_ids = list(class_ids)
        self.quality_ids = list(quality_ids)
        self.imgsz = int(imgsz)
        self.random = random.Random(int(seed))
        self.order: list[str] = []
        self.position = 0
        self._reshuffle()

    def _reshuffle(self):
        self.order = list(self.names)
        self.random.shuffle(self.order)
        self.position = 0

    def _next_names(self, count: int) -> list[str]:
        selected = []
        while len(selected) < count:
            if self.position >= len(self.order):
                self._reshuffle()
            take = min(count - len(selected), len(self.order) - self.position)
            selected.extend(self.order[self.position:self.position + take])
            self.position += take
        return selected

    def targets(self, names: list[str]) -> dict[str, torch.Tensor]:
        defect, quality, usability, diagnostic = [], [], [], []
        usability_index = {"usable": 0, "review": 1, "unusable": 2}
        for name in names:
            record = self.records[Path(name).name]
            present = {int(item["class_id"]) for item in record.get("objects", [])}
            qualities = set(record.get("quality_attributes", []))
            defect.append([float(class_id in present) for class_id in self.class_ids])
            quality.append([float(quality_id in qualities) for quality_id in self.quality_ids])
            usability.append(usability_index[record["usability"]])
            diagnostic.append(float(record["usability"] == "usable" and not record.get("ignore_regions")))
        return {
            "defect_presence": torch.tensor(defect, dtype=torch.float32),
            "quality_attributes": torch.tensor(quality, dtype=torch.float32),
            "usability": torch.tensor(usability, dtype=torch.long),
            "diagnostic_mask": torch.tensor(diagnostic, dtype=torch.float32),
        }

    def next_batch(self, count: int) -> dict[str, Any]:
        names = self._next_names(count)
        images = torch.stack([_letterbox_sem(self.root / name, self.imgsz, augment=True) for name in names])
        return {"global_img": images, "global_names": names, **self.targets(names)}


def build_hierarchical_types():
    """Create classes lazily so importing data tooling does not require Ultralytics."""
    try:
        from ultralytics.models.yolo.detect import DetectionTrainer
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import RANK
        from ultralytics.utils.torch_utils import unwrap_model
    except ImportError as exc:
        raise ImportError("层级检测训练需要 ultralytics==8.4.156") from exc

    class HierarchicalDetectionModel(DetectionModel):
        """YOLO26 detector with a shared-neck whole-image diagnosis head."""

        def __init__(self, cfg, nc: int, ch: int, quality_classes: int, hierarchy_config: dict,
                     verbose: bool = True):
            super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
            self.hierarchy_config = dict(hierarchy_config)
            self.hierarchy_epoch = 0
            self.global_head = GlobalDiagnosisHead(
                _feature_channels(self), defect_classes=nc, quality_classes=quality_classes,
                hidden_dim=int(hierarchy_config.get("global_hidden_dim", 256)),
            )
            self.last_global_outputs: dict[str, torch.Tensor] | None = None

        def predict(self, x, *args, **kwargs):
            predictions = super().predict(x, *args, **kwargs)
            if not self.training:
                raw = _raw_predictions(predictions)
                self.last_global_outputs = self.global_head(raw["feats"])
            return predictions

        def loss(self, batch, preds=None):
            if getattr(self, "criterion", None) is None:
                self.criterion = self.init_criterion()
            if preds is None:
                preds = self.predict(batch["img"])
            native_loss, native_items = self.criterion(preds, batch)
            detector_raw = _raw_predictions(preds)
            detector_outputs = self.global_head(detector_raw["feats"])
            detector_targets = {
                key: batch[key] for key in
                ("defect_presence", "quality_attributes", "usability", "diagnostic_mask")
            }
            consistency_config = {
                **self.hierarchy_config,
                "global_defect_weight": 0.0,
                "quality_weight": 0.0,
                "usability_weight": 0.0,
            }
            detector_aux = hierarchical_losses(
                detector_outputs, detector_targets,
                detector_raw["scores"].permute(0, 2, 1).contiguous(),
                self.hierarchy_epoch, consistency_config,
            )

            global_predictions = self.predict(batch["global_img"])
            global_raw = _raw_predictions(global_predictions)
            global_outputs = self.global_head(global_raw["feats"])
            global_targets = {
                "defect_presence": batch["global_defect_presence"],
                "quality_attributes": batch["global_quality_attributes"],
                "usability": batch["global_usability"],
                "diagnostic_mask": batch["global_diagnostic_mask"],
            }
            supervised_config = {**self.hierarchy_config, "consistency_weight": 0.0}
            global_aux = hierarchical_losses(
                global_outputs, global_targets,
                global_raw["scores"].permute(0, 2, 1).contiguous(),
                self.hierarchy_epoch, supervised_config,
            )
            auxiliary = detector_aux["hierarchical_total"] + global_aux["hierarchical_total"]
            batch_size = int(batch["img"].shape[0])
            combined = torch.cat((native_loss.reshape(-1), (auxiliary * batch_size).reshape(1)))
            items = dict(native_items)
            items["hierarchical_loss"] = auxiliary.detach()
            return combined, items

    class HierarchicalDetectionTrainer(DetectionTrainer):
        """Trainer that supplies a detection-safe batch plus an all-image global batch."""

        def __init__(self, hierarchy_config: dict, global_labels_path: str | Path,
                     class_ids: list[int], quality_ids: list[str], *args, **kwargs):
            self.hierarchy_config = dict(hierarchy_config)
            self.global_labels_path = Path(global_labels_path)
            self.v3_class_ids = list(class_ids)
            self.v3_quality_ids = list(quality_ids)
            self.global_stream = None
            super().__init__(*args, **kwargs)

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(HierarchicalDetectionModel(
                cfg, nc=self.data["nc"], ch=self.data["channels"],
                quality_classes=len(self.v3_quality_ids), hierarchy_config=self.hierarchy_config,
                verbose=verbose and RANK == -1,
            ))
            if weights:
                model.load(weights)
            return model

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            if self.global_stream is None:
                self.global_stream = GlobalImageStream(
                    self.global_labels_path, self.v3_class_ids, self.v3_quality_ids,
                    int(self.args.imgsz), int(self.args.seed),
                )
            detector_targets = self.global_stream.targets([Path(value).name for value in batch["im_file"]])
            for key, value in detector_targets.items():
                batch[key] = value.to(self.device, non_blocking=self.device.type not in {"cpu", "mps"})
            global_batch = self.global_stream.next_batch(len(batch["im_file"]))
            batch["global_img"] = global_batch.pop("global_img").to(
                self.device, non_blocking=self.device.type not in {"cpu", "mps"}).float() / 255
            batch["global_names"] = global_batch.pop("global_names")
            for key, value in global_batch.items():
                batch[f"global_{key}"] = value.to(
                    self.device, non_blocking=self.device.type not in {"cpu", "mps"})
            unwrap_model(self.model).hierarchy_epoch = int(self.epoch)
            return batch

    # Stable module paths are required when torch serializes the custom checkpoint.
    HierarchicalDetectionModel.__module__ = __name__
    HierarchicalDetectionTrainer.__module__ = __name__
    HierarchicalDetectionModel.__qualname__ = "HierarchicalDetectionModel"
    HierarchicalDetectionTrainer.__qualname__ = "HierarchicalDetectionTrainer"
    globals()["HierarchicalDetectionModel"] = HierarchicalDetectionModel
    globals()["HierarchicalDetectionTrainer"] = HierarchicalDetectionTrainer
    return HierarchicalDetectionModel, HierarchicalDetectionTrainer


def create_hierarchical_trainer(**kwargs):
    """Return a configured Ultralytics trainer without starting training."""
    _, trainer_type = build_hierarchical_types()
    return trainer_type(**kwargs)
