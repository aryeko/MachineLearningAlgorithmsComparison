import numpy as np

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def sigmoid_kernel(x, y):
    return np.tanh(np.dot(x,y) + 1)

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))