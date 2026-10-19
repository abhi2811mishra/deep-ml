
import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(
    n_samples: int,
    k: int = 5,
    shuffle: bool = True
) -> List[Tuple[List[int], List[int]]]:

    indices = np.arange(n_samples)  # Fix 1: arange, not arrange

    if shuffle:
        np.random.shuffle(indices)

    folds = np.array_split(indices, k)
    result = []

    for i in range(k):
        test_indices = folds[i].tolist()

        train_indices = np.concatenate(
            [folds[j] for j in range(k) if j != i]
        ).tolist()

        result.append((train_indices, test_indices))  # Fix 2

    return result  # Fix 3: return the result
