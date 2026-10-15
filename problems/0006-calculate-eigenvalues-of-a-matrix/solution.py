
import numpy as np

def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    a = np.array(matrix)
    eigenvalue = np.linalg.eigvals(a)
    eigenvalues = eigenvalue.tolist()
    return eigenvalues
