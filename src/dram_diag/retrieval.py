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


def retrieval_metrics(index, embeddings, records, k=5):
    recalls, aps, ndcgs = [], [], []
    for embedding, query in zip(embeddings, records):
        results = index.search(embedding, k, query.get("image_name"))
        relevant = [int(item["defect_id"] == query["defect_id"]) for item in results]
        total_relevant = max(1, sum(item["defect_id"] == query["defect_id"] and item.get("image_name") != query.get("image_name") for item in index.records))
        recalls.append(sum(relevant) / min(k, total_relevant))
        precisions = [sum(relevant[:i + 1]) / (i + 1) for i in range(len(relevant)) if relevant[i]]
        aps.append(sum(precisions) / min(k, total_relevant))
        dcg = sum(value / np.log2(i + 2) for i, value in enumerate(relevant))
        ideal = sum(1 / np.log2(i + 2) for i in range(min(k, total_relevant)))
        ndcgs.append(dcg / ideal if ideal else 0)
    return {"recall_at_5": float(np.mean(recalls)), "map_at_5": float(np.mean(aps)), "ndcg_at_5": float(np.mean(ndcgs))}
