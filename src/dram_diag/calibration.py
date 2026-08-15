import numpy as np


def softmax(logits, temperature=1.0):
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    values = np.exp(scaled)
    return values / values.sum(axis=1, keepdims=True)


def fit_temperature(logits, labels):
    from scipy.optimize import minimize_scalar

    logits, labels = np.asarray(logits), np.asarray(labels, dtype=int)
    def objective(log_temperature):
        probabilities = softmax(logits, np.exp(log_temperature))
        return -np.log(probabilities[np.arange(len(labels)), labels].clip(1e-12)).mean()
    result = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
    return float(np.exp(result.x))


def fused_unknown_score(probabilities, embeddings, prototypes, alpha=.5):
    confidence_evidence = 1 - np.asarray(probabilities).max(axis=1)
    prototype_similarity = np.asarray(embeddings) @ np.asarray(prototypes).T
    distance_evidence = (1 - prototype_similarity.max(axis=1)) / 2
    return alpha * confidence_evidence + (1 - alpha) * distance_evidence


def calibrate_rejection(known_scores, unknown_scores, max_known_reject=.2):
    known_scores, unknown_scores = np.asarray(known_scores), np.asarray(unknown_scores)
    candidates = np.unique(np.r_[known_scores, unknown_scores])
    feasible = [value for value in candidates if np.mean(known_scores >= value) <= max_known_reject]
    if not feasible:
        return float(np.max(known_scores) + 1e-9)
    return float(max(feasible, key=lambda value: np.mean(unknown_scores >= value)))

