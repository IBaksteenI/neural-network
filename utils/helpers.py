import numpy as np

# math behind sigmoid activation function
# sigmoid(x) = 1 / (1 + e^(-x))
# sigmoid_derivative(x) = sigmoid(x) * (1 - sigmoid(x))
# what is e? e is Euler's number, approximately equal to 2.71828
# e is the base of the natural logarithm
# e is an irrational number, meaning it cannot be expressed as a simple fraction
# e is also a transcendental number, meaning it is not a root of any non-zero polynomial equation with rational coefficients


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)