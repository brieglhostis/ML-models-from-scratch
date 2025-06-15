
import numpy as np


def mse(Y, Y_pred, axis=0):
    """
    Compute the mean square error: MSE = mean((Y-Y_pred)**2)
    Arguments:
     - Y      (np.ndarray)         - target actuals
     - Y_pred (np.ndarray)         - predicted target
     - axis (Any[int, tuple(int)]) - axis along which to average out
    """
    return np.mean(np.square(Y-Y_pred), axis=axis)

def r_square(Y, Y_pred, axis=0):
    """
    Compute the R²: R² = 1 - sum((Y-Y_pred)**2) / sum((Y-Y_mean)**2)
    Arguments:
     - Y      (np.ndarray)         - target actuals
     - Y_pred (np.ndarray)         - predicted target
     - axis (Any[int, tuple(int)]) - axis along which to sum
    """
    return 1 - np.sum(np.square(Y-Y_pred), axis=axis) / np.sum(np.square(Y-np.mean(Y, axis=axis)), axis=axis)

def log_loss(Y, Y_pred, axis=0):
    """
    Compute the log loss: log loss = - mean(Y * log(Y_pred) + (1-Y) * log(1-Y_pred))
    Arguments:
     - Y      (np.ndarray)         - binary target actuals
     - Y_pred (np.ndarray)         - predicted target probability
     - axis (Any[int, tuple(int)]) - axis along which to average out
    """
    return - np.mean(Y * np.log(Y_pred) + (1 - Y) * np.log(1 - Y_pred), axis=axis)

def sigmoid(x, safety_threshold=200):
    """
    Return the sigmoid of the input: s(x) = 1 / (1 + exp(-x))
    Arguments:
     - x (np.ndarray)                     - input of the function
     - safety_threshold (Any[float, int]) - input threshold for the exponential function
    """
    exp_safety_threshold = np.exp(safety_threshold)
    exp_minus_x = np.where(x<-safety_threshold, exp_safety_threshold, np.exp(-x))
    return 1 / (1+exp_minus_x)

def tanh(x, safety_threshold=200):
    """
    Return the hyperbolic tangent of the input: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Arguments:
     - x (np.ndarray)                     - input of the function
     - safety_threshold (Any[float, int]) - input threshold for the exponential function
    """
    exp_safety_threshold = np.exp(safety_threshold)
    exp_x = np.where(x>safety_threshold, exp_safety_threshold, np.exp(x))
    exp_minus_x = np.where(x<-safety_threshold, exp_safety_threshold, np.exp(-x))
    return (exp_x-exp_minus_x) / (exp_x+exp_minus_x)

def softmax(X, axis=-1, safety_threshold=200):
    """
    Return the softmax of the input: smax(x) = exp(x) / sum(exp(x))
    Arguments:
     - x (np.ndarray)                     - input of the function
     - axis (Any[int, tuple(int)])        - axis along which to sum for normalization
     - safety_threshold (Any[float, int]) - input threshold for the exponential function
    """
    exp_safety_threshold = np.exp(safety_threshold)
    exp_X = np.where(X>safety_threshold, exp_safety_threshold, np.exp(X))
    return exp_X / np.sum(exp_X, axis=axis, keepdims=True)

def dummy_encode(X, F=None):
    """
    Encode numerical input into F binary categories
    Arguments:
     - X (np.ndarray) - input to encode
     - F (int)        - number of categories in input
    """
    if len(X.shape) == 1:
        F = np.max(X) if F is None else F
        return np.array([np.where(np.arange(F) == x, 1, 0) for x in X])
    return np.array([dummy_encode(x, F) for x in X])

def one_hot_encode(X, F=None):
    """
    Encode numerical input into F-1 binary categories
    (the first category is ignored)
    Arguments:
     - X (np.ndarray) - input to encode
     - F (int)        - number of categories in input
    """
    if len(X.shape) == 1:
        F = np.max(X) if F is None else F
        return np.array([np.where(np.arange(1, F) == x, 1, 0) for x in X])
    return np.array([one_hot_encode(x, F) for x in X])
