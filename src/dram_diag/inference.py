from pathlib import Path
import json
from .data import image_array
from .open_set import evidence, classify

class Predictor:
    def __init__(self, model=None, prototypes=None, threshold=.55, device="cpu", model_version="untrained", family_enabled=False):
        self.model=model; self.prototypes=prototypes; self.threshold=threshold; self.device=device; self.model_version=model_version; self.family_enabled=family_enabled
    def predict(self, path):
        if self.model is None: return {"image_name":Path(path).name,"status":"untrained","review_required":True,"model_version":self.model_version}
        import torch
        self.model.eval(); x=torch.from_numpy(image_array(path)).unsqueeze(0).to(self.device)
        with torch.no_grad(): out=self.model(x); prob=torch.softmax(out["logits"],1); conf,idx=prob.max(1)
        emb=out["embedding"][0].detach().cpu().numpy(); cases=self.prototypes.search(emb) if self.prototypes else []
        dist=1-cases[0]["similarity"] if cases else 1.0; score=evidence(float(conf),dist, family_enabled=self.family_enabled)
        return {"image_name":Path(path).name,"defect_class":int(idx),"confidence":float(conf),"unknown_score":1-score,"status":classify(score,self.threshold),"top5_similar_cases":cases,"review_required":score<self.threshold,"model_version":self.model_version}
