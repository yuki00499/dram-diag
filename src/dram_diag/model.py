import math

try:
    import torch
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError as exc:
    raise ImportError("模型功能需要安装 torch 和 torchvision") from exc


class DefectClassifier(nn.Module):
    def __init__(self, num_classes, backbone="resnet18", pretrained=True, dropout=0.0,
                 label_graph=None, graph_alpha_init=.1):
        super().__init__()
        self.backbone_name = backbone
        if backbone != "resnet18":
            raise ValueError(f"不支持的 backbone: {backbone}")
        base = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.embedding_dim = 512
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embedding_dim, num_classes)
        if label_graph is not None:
            graph = torch.as_tensor(label_graph, dtype=torch.float32)
            if graph.shape != (num_classes, num_classes):
                raise ValueError("标签图尺寸与输出类别数不一致")
            self.register_buffer("label_graph", graph)
            initial = min(max(float(graph_alpha_init), 1e-4), 1 - 1e-4)
            self.graph_alpha_logit = nn.Parameter(torch.tensor(math.log(initial / (1 - initial))))
        else:
            self.label_graph = None
            self.graph_alpha_logit = None

    def forward(self, inputs):
        features = self.encoder(inputs).flatten(1)
        embedding = nn.functional.normalize(features, dim=1)
        raw_logits = self.classifier(self.dropout(features))
        logits = raw_logits
        if self.label_graph is not None:
            message = torch.tanh(raw_logits) @ self.label_graph
            logits = raw_logits + torch.sigmoid(self.graph_alpha_logit) * message
        return {"logits": logits, "raw_logits": raw_logits, "embedding": embedding}
