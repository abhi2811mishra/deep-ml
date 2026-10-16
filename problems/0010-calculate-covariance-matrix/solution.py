
import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    covariance = np.cov(vectors, ddof=1)
    return covariance.tolist()
