
import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    A = np.asarray(scores)
    return A - np.log(np.sum(np.exp(A)))
