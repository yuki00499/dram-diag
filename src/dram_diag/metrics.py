import numpy as np

def expected_calibration_error(confidence, correct, bins=10):
    confidence=np.asarray(confidence); correct=np.asarray(correct); ece=0.0
    edges=np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask=(confidence>=lo)&(confidence<hi if hi<1 else confidence<=hi)
        if mask.any(): ece += mask.mean()*abs(confidence[mask].mean()-correct[mask].mean())
    return float(ece)
