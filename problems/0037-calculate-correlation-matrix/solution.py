import numpy as np

def calculate_correlation_matrix(X, Y=None):
    if Y is None:
        Y = X

    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)

    numerator = X_centered.T @ Y_centered

    X_norm = np.sqrt(np.sum(X_centered ** 2, axis=0))
    Y_norm = np.sqrt(np.sum(Y_centered ** 2, axis=0))

    denominator = np.outer(X_norm, Y_norm)

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0
    )