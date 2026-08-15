try:
    import torch
    from torch import nn
    from torchvision.models import ConvNeXt_Tiny_Weights, ResNet18_Weights, convnext_tiny, resnet18
except ImportError as exc:
    raise ImportError("模型功能需要安装 torch 和 torchvision") from exc


class DefectClassifier(nn.Module):
    def __init__(self, num_classes, backbone="resnet18", pretrained=True, dropout=0.0, projection_dim=0):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "resnet18":
            base = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            self.encoder = nn.Sequential(*list(base.children())[:-1])
            self.embedding_dim = 512
        elif backbone == "convnext_tiny":
            base = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
            self.encoder = nn.Sequential(base.features, base.avgpool, base.classifier[0])
            self.embedding_dim = 768
        else:
            raise ValueError(f"不支持的 backbone: {backbone}")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embedding_dim, num_classes)
        self.projection = None
        if projection_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, projection_dim),
            )

    def forward(self, inputs):
        features = self.encoder(inputs).flatten(1)
        embedding = nn.functional.normalize(features, dim=1)
        output = {"logits": self.classifier(self.dropout(features)), "embedding": embedding}
        if self.projection is not None:
            output["projection"] = nn.functional.normalize(self.projection(features), dim=1)
        return output


def batch_prototype_loss(embeddings, labels, margin=.2):
    unique = labels.unique()
    centers = []
    compactness = embeddings.sum() * 0
    for label in unique:
        members = embeddings[labels == label]
        center = nn.functional.normalize(members.mean(dim=0), dim=0)
        centers.append(center)
        compactness = compactness + (1 - members @ center).mean()
    compactness = compactness / max(1, len(centers))
    if len(centers) < 2:
        return compactness
    matrix = torch.stack(centers)
    similarity = matrix @ matrix.T
    mask = ~torch.eye(len(matrix), dtype=torch.bool, device=matrix.device)
    separation = torch.relu(similarity[mask] - (1 - margin)).mean()
    return compactness + separation


def supervised_contrastive_loss(features, labels, temperature=.1):
    similarity = features @ features.T / temperature
    logits_mask = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & logits_mask
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(similarity) * logits_mask
    log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    counts = positive_mask.sum(dim=1)
    valid = counts > 0
    if not valid.any():
        return features.sum() * 0
    mean_log_prob = (positive_mask * log_prob).sum(dim=1) / counts.clamp_min(1)
    return -mean_log_prob[valid].mean()
