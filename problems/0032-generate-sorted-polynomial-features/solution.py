import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def polynomial_features(X, degree):
    poly = PolynomialFeatures(degree=degree)
    result = poly.fit_transform(X)
    result = np.sort(result, axis=1)
    return result