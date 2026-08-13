import numpy as np

class PrototypeStore:
    def __init__(self): self.embeddings=np.empty((0,0),np.float32); self.labels=[]; self.names=[]
    def fit(self, embeddings, labels, names=None):
        self.embeddings=np.asarray(embeddings,np.float32); self.embeddings/=np.maximum(np.linalg.norm(self.embeddings,axis=1,keepdims=True),1e-8); self.labels=list(labels); self.names=list(names or range(len(labels))); return self
    def search(self, embedding, k=5):
        if not len(self.embeddings): return []
        q=np.asarray(embedding,np.float32); q/=max(float(np.linalg.norm(q)),1e-8); scores=self.embeddings@q; idx=np.argsort(-scores)[:k]
        return [{"image_name":self.names[i],"defect_id":self.labels[i],"similarity":float(scores[i])} for i in idx]
