import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    a = np.array(matrix)

    if mode == 'row':
        me = np.mean(a, axis=1)
    elif mode == 'column':
        me = np.mean(a, axis=0)
    else:
        return []

    means = me.tolist()
    return means