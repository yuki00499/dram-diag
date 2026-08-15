from pathlib import Path

import numpy as np

from .compat import validate_checkpoint
from .data import image_array
from .model import DefectClassifier


class ModelRunner:
    def __init__(self, checkpoint_paths, manifest, data_root="晶圆缺陷分类数据集", device=None):
        import torch

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.manifest = manifest
        self.data_root = Path(data_root)
        self.models, self.checkpoints = [], []
        for path in checkpoint_paths:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            validate_checkpoint(checkpoint, manifest)
            config = checkpoint["config"]
            model = DefectClassifier(len(checkpoint["class_to_idx"]), config.get("backbone", "resnet18"), pretrained=False,
                                     dropout=float(config.get("dropout", 0)), projection_dim=0)
            state = {key: value for key, value in checkpoint["model"].items() if not key.startswith("projection.")}
            model.load_state_dict(state, strict=False)
            self.models.append(model.to(self.device).eval())
            self.checkpoints.append(checkpoint)
        self.class_to_idx = {int(key): int(value) for key, value in self.checkpoints[0]["class_to_idx"].items()}
        self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}

    def infer_paths(self, paths, batch_size=8):
        torch = self.torch
        config = self.checkpoints[0]["config"]
        arrays = [image_array(path, tuple(config.get("image_size", [480, 320])), False,
                              self.checkpoints[0]["normalization"], self.checkpoints[0]["data_stats"]) for path in paths]
        all_logits, all_embeddings = [], []
        with torch.no_grad():
            for start in range(0, len(arrays), batch_size):
                batch = torch.from_numpy(np.stack(arrays[start:start + batch_size])).to(self.device)
                outputs = [model(batch) for model in self.models]
                all_logits.append(torch.stack([item["logits"] for item in outputs]).mean(0).cpu().numpy())
                embedding = torch.stack([item["embedding"] for item in outputs]).mean(0)
                embedding = torch.nn.functional.normalize(embedding, dim=1)
                all_embeddings.append(embedding.cpu().numpy())
        return np.concatenate(all_logits), np.concatenate(all_embeddings)

    def infer_items(self, items):
        paths = [self.data_root / "images" / item["image_name"] for item in items]
        return self.infer_paths(paths)
