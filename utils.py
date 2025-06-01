
import numpy as np


def sigmoid(x, safety_threshold=200):
    exp_safety_threshold = np.exp(safety_threshold)
    exp_minus_x = np.where(x<-safety_threshold, exp_safety_threshold, np.exp(-x))
    return 1 / (1+exp_minus_x)

def tanh(x, safety_threshold=200):
    exp_safety_threshold = np.exp(safety_threshold)
    exp_x = np.where(x>safety_threshold, exp_safety_threshold, np.exp(x))
    exp_minus_x = np.where(x<-safety_threshold, exp_safety_threshold, np.exp(-x))
    return (exp_x-exp_minus_x) / (exp_x+exp_minus_x)

def softmax(X, safety_threshold=200, axis=-1):
    exp_safety_threshold = np.exp(safety_threshold)
    exp_X = np.where(X>safety_threshold, exp_safety_threshold, np.exp(X))
    return exp_X / np.sum(exp_X, axis=axis, keepdims=True)

def dummy_encode(X, F=None):
    if len(X.shape) == 1:
        F = np.max(X) if F is None else F
        return np.array([np.where(np.arange(F) == x, 1, 0) for x in X])
    return np.array([dummy_encode(x, F) for x in X])

def one_hot_encode(X, F=None):
    if len(X.shape) == 1:
        F = np.max(X) if F is None else F
        return np.array([np.where(np.arange(1, F) == x, 1, 0) for x in X])
    return np.array([one_hot_encode(x, F) for x in X])
