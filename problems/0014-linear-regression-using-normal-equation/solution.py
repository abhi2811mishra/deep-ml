import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    x = np.array(X)
    y = np.array(y)

    coefficient = np.linalg.inv(x.T @ x) @ x.T @ y

    theta = []
    for c in coefficient:
        theta.append(round(float(c), 4))

    return theta