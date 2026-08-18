import numpy as np


class RetrievalIndex:
    def __init__(self, embeddings, records):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.records = list(records)

    def search(self, embedding, k=5, exclude_image=None):
        scores = self.embeddings @ np.asarray(embedding, dtype=np.float32)
        order = np.argsort(scores)[::-1]
        output = []
        for index in order:
            record = self.records[int(index)]
            if exclude_image and record.get("image_name") == exclude_image:
                continue
            output.append({**record, "similarity": float(scores[index])})
            if len(output) == k:
                break
        return output
