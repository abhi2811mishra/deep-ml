
import numpy as np

def train_neuron(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    learning_rate: float,
    epochs: int
) -> tuple[list[float], float, list[float]]:

    weights = initial_weights.copy()
    bias = initial_bias
    mse_values = []

    n = len(features)

    for _ in range(epochs):
       
        z = np.dot(features, weights) + bias
        predictions = 1 / (1 + np.exp(-z))

        mse = np.mean((predictions - labels) ** 2)
        mse_values.append(float(mse))

        delta = (
            2 * (predictions - labels)
            * predictions * (1 - predictions)
        )

        weight_gradients = np.dot(features.T, delta) / n
        bias_gradient = np.mean(delta)

        weights -= learning_rate * weight_gradients
        bias -= learning_rate * bias_gradient

    updated_weights = [round(float(w), 4) for w in weights]
    updated_bias = round(float(bias), 4)
    mse_values = [round(m, 4) for m in mse_values]

    return updated_weights, updated_bias, mse_values
