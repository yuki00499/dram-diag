try:
    import torch
    from torch import nn
    from torchvision.models import resnet18, ResNet18_Weights
except ImportError as exc:
    raise ImportError("模型功能需要安装 torch torchvision；审计和划分功能无需 torch。") from exc

class HPODModel(nn.Module):
    def __init__(self, num_classes, num_families=0, pretrained=True):
        super().__init__(); weights=ResNet18_Weights.DEFAULT if pretrained else None
        base=resnet18(weights=weights); self.encoder=nn.Sequential(*list(base.children())[:-1]); self.embedding_dim=512
        self.fine_head=nn.Linear(512,num_classes); self.family_head=nn.Linear(512,num_families) if num_families else None
    def forward(self,x):
        z=self.encoder(x).flatten(1); emb=nn.functional.normalize(z,dim=1); out={"logits":self.fine_head(z),"embedding":emb}
        if self.family_head is not None: out["family_logits"]=self.family_head(z)
        return out
