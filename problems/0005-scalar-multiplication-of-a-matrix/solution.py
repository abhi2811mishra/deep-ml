
import numpy as np

def scalar_multiply(matrix: list[list[int | float]], scalar: int | float) -> list[list[int | float]]:
    a = np.array(matrix)
    result = scalar * a
    ans = result.tolist()
    return ans
