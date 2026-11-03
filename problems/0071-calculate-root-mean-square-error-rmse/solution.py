
import numpy as np

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("Arrays must have the same shape")

    if y_true.size == 0:
        raise ValueError("Arrays must not be empty")

    error = y_true - y_pred
    squared_error = error ** 2
    mean_squared_error = np.mean(squared_error)

    rmse_res = float(np.sqrt(mean_squared_error))
    return round(rmse_res, 3)
