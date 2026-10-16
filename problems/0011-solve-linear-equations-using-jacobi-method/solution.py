
import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    size = len(A)
    x = np.zeros(size)

    for _ in range(n):
        new_x = np.zeros(size)

        for i in range(size):
            total = sum(A[i, j] * x[j] for j in range(size) if j != i)
            new_x[i] = (b[i] - total) / A[i, i]

        x = new_x

    return np.round(x, 4).tolist()
