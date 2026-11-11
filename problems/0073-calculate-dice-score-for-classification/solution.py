
import numpy as np

def dice_score(y_true, y_pred):
    intersection = np.sum((y_true == 1) & (y_pred == 1))
    total = np.sum(y_true == 1) + np.sum(y_pred == 1)
    if total == 0:
        return 0.0
    res = (2*intersection)/total
    return round(res, 3)
