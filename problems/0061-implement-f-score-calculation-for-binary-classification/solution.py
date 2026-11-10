import numpy as np

def f_score(y_true, y_pred, beta):
    """
    Calculate F-Score for a binary classification task.

    :param y_true: Numpy array of true labels
    :param y_pred: Numpy array of predicted labels
    :param beta: Weight of recall relative to precision (beta > 1 favours recall)
    :return: F-Score rounded to three decimal places
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))

    FP = np.sum((y_true == 0) & (y_pred == 1))

    FN = np.sum((y_true == 1) & (y_pred == 0))

    precesion = TP/(TP+FP) if (TP+FP)!= 0 else 0.0
    recall = TP/(TP+FN) if (TP+FN) != 0 else 0.0

    beta_sq = beta**2
    denominator = beta_sq*precesion + recall
    if denominator == 0:
        return 0.0
    score = (1+beta_sq)*precesion*recall/denominator
    return round(score,3)
