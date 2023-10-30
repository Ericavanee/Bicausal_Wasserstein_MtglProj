import numpy as np

# This package provides a list of continuous bounded functions and two additional unbounded continuous polynomial functions

def inverse(x):
    return 1/x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def gaussian(x, mean=0, stddev=1):
    return np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def relu(x):
    return np.maximum(0, x)

def sinusoidal(x):
    return np.sin(x)

def piecewise(x):
    if x < 0:
        return -1
    elif x > 1:
        return 1
    else:
        return x

def arctan(x):
    return np.arctan(x)

# additional unbounded polynomial functions
def identity(x):
    return x

def square(x):
    return x**2