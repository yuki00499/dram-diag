"""Hierarchical detection building blocks shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class GlobalDiagnosisHead(nn.Module):
    """Pool multiple neck levels into image-level defect, quality and usability logits."""

    def __init__(self, feature_channels: Iterable[int], defect_classes: int, quality_classes: int,
                 hidden_dim: int = 256, dropout: float = .1):
        super().__init__()
        channels = [int(value) for value in feature_channels]
        if not channels or defect_classes < 1 or quality_classes < 0:
            raise ValueError("全局诊断头的维度配置无效")
        self.feature_channels = channels
        self.defect_classes = int(defect_classes)
        self.quality_classes = int(quality_classes)
        self.project = nn.Sequential(
            nn.Linear(sum(channels), hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.defect_head = nn.Linear(hidden_dim, defect_classes)
        self.quality_head = nn.Linear(hidden_dim, quality_classes) if quality_classes else None
        self.usability_head = nn.Linear(hidden_dim, 3)

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        if len(features) != len(self.feature_channels):
            raise ValueError("特征层数量与全局诊断头配置不一致")
        pooled = []
        for feature, channels in zip(features, self.feature_channels):
            if feature.ndim != 4 or feature.shape[1] != channels:
                raise ValueError("全局诊断头收到无效特征尺寸")
            pooled.append(F.adaptive_avg_pool2d(feature, 1).flatten(1))
        hidden = self.project(torch.cat(pooled, dim=1))
        quality = self.quality_head(hidden) if self.quality_head is not None else hidden.new_zeros((len(hidden), 0))
        return {
            "global_defect_logits": self.defect_head(hidden),
            "quality_logits": quality,
            "usability_logits": self.usability_head(hidden),
            "global_embedding": F.normalize(hidden, dim=1),
        }


class P2DetectionHead(nn.Module):
    """Small stride-4 auxiliary prediction head used by the B1/B3 variants."""

    def __init__(self, in_channels: int, num_classes: int, reg_max: int = 16):
        super().__init__()
        hidden = max(64, min(256, int(in_channels)))
        self.num_classes = int(num_classes)
        self.reg_max = int(reg_max)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
        )
        self.box = nn.Conv2d(hidden, 4 * reg_max, 1)
        self.classes = nn.Conv2d(hidden, num_classes, 1)

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.stem(feature)
        return {"box_distribution": self.box(hidden), "class_logits": self.classes(hidden)}


def soft_presence_pool(class_logits: torch.Tensor, temperature: float = .25) -> torch.Tensor:
    """Differentiably aggregate pre-NMS candidate logits into image-level logits.

    Input is ``[batch, candidates, classes]``. Normalized log-mean-exp prevents
    the number of candidates from changing the probability scale.
    """
    if class_logits.ndim != 3 or class_logits.shape[1] < 1:
        raise ValueError("class_logits 必须为 [B,N,C] 且 N>0")
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature 必须大于0")
    count = class_logits.shape[1]
    return temperature * (
        torch.logsumexp(class_logits / temperature, dim=1) - math.log(count)
    )


def consistency_weight(epoch: int, warmup_epochs: int, target_weight: float) -> float:
    if target_weight < 0 or warmup_epochs < 0:
        raise ValueError("一致性损失配置不能为负数")
    if warmup_epochs == 0:
        return float(target_weight)
    return float(target_weight) * min(max(int(epoch), 0) / warmup_epochs, 1.0)


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return loss.mean()
    mask = mask.to(dtype=loss.dtype, device=loss.device)
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    weighted = loss * mask
    return weighted.sum() / mask.expand_as(loss).sum().clamp_min(1.0)


def symmetric_bernoulli_kl(left_logits: torch.Tensor, right_logits: torch.Tensor,
                           eps: float = 1e-6) -> torch.Tensor:
    left = left_logits.sigmoid().clamp(eps, 1 - eps)
    right = right_logits.sigmoid().clamp(eps, 1 - eps)

    def kl(p, q):
        return p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()

    return .5 * (kl(left, right) + kl(right, left))


def hierarchical_losses(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor],
                        detection_class_logits: torch.Tensor, epoch: int,
                        config: dict) -> dict[str, torch.Tensor]:
    """Return the auxiliary losses added to the native YOLO detection loss."""
    pooled = soft_presence_pool(
        detection_class_logits, float(config.get("presence_temperature", .25)))
    diagnostic_mask = targets.get("diagnostic_mask")
    defect = _masked_mean(
        F.binary_cross_entropy_with_logits(
            outputs["global_defect_logits"], targets["defect_presence"].float(), reduction="none"),
        diagnostic_mask,
    )
    quality = outputs["global_defect_logits"].new_zeros(())
    if outputs["quality_logits"].shape[1]:
        quality = F.binary_cross_entropy_with_logits(
            outputs["quality_logits"], targets["quality_attributes"].float())
    usability = F.cross_entropy(outputs["usability_logits"], targets["usability"].long())
    consistency = _masked_mean(
        symmetric_bernoulli_kl(outputs["global_defect_logits"], pooled), diagnostic_mask)
    ramp = consistency_weight(
        epoch, int(config.get("consistency_warmup_epochs", 10)),
        float(config.get("consistency_weight", .2)))
    total = (
        float(config.get("global_defect_weight", .5)) * defect
        + float(config.get("quality_weight", .25)) * quality
        + float(config.get("usability_weight", .25)) * usability
        + ramp * consistency
    )
    return {
        "hierarchical_total": total,
        "global_defect_loss": defect,
        "quality_loss": quality,
        "usability_loss": usability,
        "consistency_loss": consistency,
        "consistency_weight": total.new_tensor(ramp),
        "pooled_detection_logits": pooled,
    }


def post_nms_presence(detections: Iterable[dict], class_ids: Iterable[int]) -> dict[int, float]:
    """Noisy-OR aggregation of post-NMS detection confidence per class."""
    remaining = {int(class_id): 1.0 for class_id in class_ids}
    for item in detections:
        class_id = int(item["class_id"])
        if class_id not in remaining:
            continue
        confidence = min(max(float(item["confidence"]), 0.0), 1.0)
        remaining[class_id] *= 1.0 - confidence
    return {class_id: 1.0 - value for class_id, value in remaining.items()}


@dataclass(frozen=True)
class ReviewPolicy:
    global_threshold: float = .5
    detection_threshold: float = .25
    disagreement_threshold: float = .35
    usability_review_threshold: float = .5


def review_decision(global_probabilities: dict[int, float], detection_probabilities: dict[int, float],
                    quality_probabilities: dict[str, float], usability_probabilities: dict[str, float],
                    policy: ReviewPolicy | None = None) -> dict:
    policy = policy or ReviewPolicy()
    reasons: list[dict] = []
    all_classes = sorted(set(global_probabilities) | set(detection_probabilities))
    disagreement = {
        class_id: abs(float(global_probabilities.get(class_id, 0))
                      - float(detection_probabilities.get(class_id, 0)))
        for class_id in all_classes
    }
    for class_id in all_classes:
        global_positive = global_probabilities.get(class_id, 0) >= policy.global_threshold
        detected = detection_probabilities.get(class_id, 0) >= policy.detection_threshold
        if global_positive and not detected:
            reasons.append({"code": "global_without_box", "class_id": class_id})
        elif detected and not global_positive:
            reasons.append({"code": "box_without_global", "class_id": class_id})
        if disagreement[class_id] >= policy.disagreement_threshold:
            reasons.append({"code": "global_local_disagreement", "class_id": class_id,
                            "value": disagreement[class_id]})
    unusable = float(usability_probabilities.get("unusable", 0))
    needs_review = float(usability_probabilities.get("review", 0))
    if unusable >= policy.usability_review_threshold:
        reasons.append({"code": "predicted_unusable", "value": unusable})
    elif needs_review >= policy.usability_review_threshold:
        reasons.append({"code": "quality_review", "value": needs_review})
    active_quality = [key for key, value in quality_probabilities.items() if float(value) >= .5]
    if active_quality:
        reasons.append({"code": "quality_degradation", "attributes": active_quality})
    return {
        "review_required": bool(reasons),
        "review_reasons": reasons,
        "global_local_disagreement": {str(key): float(value) for key, value in disagreement.items()},
        "max_disagreement": max(disagreement.values(), default=0.0),
    }
