import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.

    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return

    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Standardize the data
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1.0
    standardized_data = (data - mean) / std

   
    cov_matrix = np.cov(standardized_data, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, indices]

    components = eigenvectors[:, :k].copy()

    for i in range(components.shape[1]):
        nonzero = np.where(np.abs(components[:, i]) > 1e-10)[0]

        if len(nonzero) > 0 and components[nonzero[0], i] < 0:
            components[:, i] *= -1

    return np.round(components, 4)