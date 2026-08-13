import argparse, json, sys, random
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from dram_diag.audit import audit
from dram_diag.config import load_config
from dram_diag.data import image_array
from dram_diag.model import HPODModel

def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Train HPOD-Classifier ResNet18")
    parser.add_argument("--data-root", default=cfg.get("data_root", "晶圆缺陷分类数据集"))
    parser.add_argument("--epochs", type=int, default=cfg.get("epochs", 20))
    parser.add_argument("--batch-size", type=int, default=cfg.get("batch_size", 16))
    parser.add_argument("--lr", type=float, default=cfg.get("lr", 3e-4))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default="runs/hpod-resnet18")
    parser.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    args = parser.parse_args()

    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    manifest = audit(args.data_root, "artifacts", args.seed)
    split = json.loads(Path("artifacts/split_manifest.json").read_text(encoding="utf-8"))["splits"]
    known = sorted({x["defect_id"] for x in split["train"]})
    class_to_idx = {c:i for i,c in enumerate(known)}
    root = Path(args.data_root) / "images"

    class RoiDataset(Dataset):
        def __init__(self, items): self.items = [x for x in items if x["defect_id"] in class_to_idx]
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            item=self.items[i]; x=torch.from_numpy(image_array(root / item["image_name"])); y=class_to_idx[item["defect_id"]]
            return x, y

    train_loader=DataLoader(RoiDataset(split["train"]), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader=DataLoader(RoiDataset(split["validation"]), batch_size=args.batch_size, shuffle=False, num_workers=0)
    model=HPODModel(len(known), pretrained=True).to(device)
    optimizer=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion=nn.CrossEntropyLoss(); scaler=torch.amp.GradScaler("cuda", enabled=device.type=="cuda")
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True); best=0.0; history=[]
    for epoch in range(1, args.epochs+1):
        model.train(); total=correct=loss_sum=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type=="cuda"):
                output = model(x); loss=criterion(output["logits"],y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss_sum += loss.item()*len(y); correct += (output["logits"].argmax(1)==y).sum().item(); total += len(y)
        model.eval(); val_correct=val_total=0
        with torch.no_grad():
            for x,y in val_loader:
                logits=model(x.to(device))["logits"]; val_correct += (logits.argmax(1)==y.to(device)).sum().item(); val_total += len(y)
        val_acc=val_correct/max(1,val_total); row={"epoch":epoch,"loss":loss_sum/max(1,total),"train_acc":correct/max(1,total),"val_acc":val_acc}; history.append(row); print(row)
        if val_acc >= best:
            best=val_acc; torch.save({"model":model.state_dict(),"class_to_idx":class_to_idx,"config":vars(args)},out/"best.pt")
    (out/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    print(f"训练完成，最佳验证准确率: {best:.4f}，模型: {out/'best.pt'}")

if __name__ == "__main__": main()
