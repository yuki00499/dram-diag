import numpy as np

def evidence(max_prob, prototype_distance, consistency=1.0, augmentation_consistency=1.0, family_enabled=False):
    distance_score=max(0.0,min(1.0,1.0-float(prototype_distance)))
    values=[float(max_prob),distance_score,float(augmentation_consistency)]
    if family_enabled: values.append(float(consistency))
    return float(np.mean(values))

def classify(score, threshold=.55):
    return "known_confident" if score>=threshold else "unknown"
