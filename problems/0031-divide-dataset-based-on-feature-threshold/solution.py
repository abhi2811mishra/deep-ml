
import numpy as np

def divide_on_feature(X, feature_i, threshold):
    left = []
    right = []

    for row in X:
        if row[feature_i] >= threshold:
            left.append(row)
        else:
            right.append(row)

    return [np.array(left), np.array(right)]
