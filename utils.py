
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

def accuracy(Y, Y_pred, threshold=0.5, axis=0):
    """
    Compute the accuracy for a binary classification: accuracy = (TP + TN) / (TP + TN + FP + FN)
    Arguments:
     - Y      (np.ndarray)         - binary target actuals
     - Y_pred (np.ndarray)         - predicted target probability
     - threshold (float)           - classification threshold to apply to Y_pred
     - axis (Any[int, tuple(int)]) - axis along which to sum
    """
    return np.mean(Y * np.where(Y_pred > threshold, 1, 0) + (1-Y) * np.where(Y_pred > threshold, 0, 1), axis=axis)

def precision(Y, Y_pred, threshold=0.5, axis=0):
    """
    Compute the precision for a binary classification: precision = TP / (TP + FP)
    Arguments:
     - Y      (np.ndarray)         - binary target actuals
     - Y_pred (np.ndarray)         - predicted target probability
     - threshold (float)           - classification threshold to apply to Y_pred
     - axis (Any[int, tuple(int)]) - axis along which to sum
    """
    return np.sum(Y * np.where(Y_pred > threshold, 1, 0), axis=axis) / np.sum(np.where(Y_pred > threshold, 1, 0), axis=axis)

def recall(Y, Y_pred, threshold=0.5, axis=0):
    """
    Compute the recall for a binary classification: recall = TP / (TP + FN)
    Arguments:
     - Y      (np.ndarray)         - binary target actuals
     - Y_pred (np.ndarray)         - predicted target probability
     - threshold (float)           - classification threshold to apply to Y_pred
     - axis (Any[int, tuple(int)]) - axis along which to sum
    """
    return np.sum(Y * np.where(Y_pred > threshold, 1, 0), axis=axis) / np.sum(Y, axis=axis)

def f1_score(Y, Y_pred, threshold=0.5, axis=0):
    """
    Compute the F1 score for a binary classification: F1 = 2 * precision * recall / (precision + recall)
    Arguments:
     - Y      (np.ndarray)         - binary target actuals
     - Y_pred (np.ndarray)         - predicted target probability
     - threshold (float)           - classification threshold to apply to Y_pred
     - axis (Any[int, tuple(int)]) - axis along which to sum
    """
    return 2 / (1/precision(Y, Y_pred, threshold=threshold, axis=axis) + 1/recall(Y, Y_pred, threshold=threshold, axis=axis))

def roc(Y, Y_pred, n=100, axis=0):
    """
    Compute ROC curve coordinates for a binary classification with values of threshold distributed between 0 and 1
    Arguments:
     - Y      (np.ndarray)         - binary target actuals
     - Y_pred (np.ndarray)         - predicted target probability
     - n (int)                     - number of threshold values
     - axis (Any[int, tuple(int)]) - axis along which to sum
    """
    thresholds = np.linspace(1.0, 0.0, n)
    fpr = np.array([np.sum((1-Y) * np.where(Y_pred > t, 1, 0), axis=axis) / np.sum(1-Y, axis=axis) for t in thresholds])
    tpr = np.array([np.sum(Y * np.where(Y_pred > t, 1, 0), axis=axis) / np.sum(Y, axis=axis) for t in thresholds])
    return tpr, fpr

def auc(Y, Y_pred, n=100, axis=0):
    """
    Compute area under curve (AUC) for a binary classification
    Arguments:
     - Y      (np.ndarray)         - binary target actuals
     - Y_pred (np.ndarray)         - predicted target probability
     - n (int)                     - number of threshold values
     - axis (Any[int, tuple(int)]) - axis along which to sum
    """
    tpr, fpr = roc(Y, Y_pred, n=n, axis=axis)
    return np.sum(0.5 * (tpr[1:]+tpr[:-1]) * (fpr[1:]-fpr[:-1]), axis=0)

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
