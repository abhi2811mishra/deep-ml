
import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    probabilities = []

    for vector in features:
        z = bias

        for x, w in zip(vector, weights):
            z += x * w

        probability = 1 / (1 + math.exp(-z))
        probabilities.append(probability)

    mse = sum((p - label) ** 2 for p, label in zip(probabilities, labels)) / len(labels)

    probabilities = [round(p, 4) for p in probabilities]
    mse = round(mse, 4)

    return probabilities, mse
