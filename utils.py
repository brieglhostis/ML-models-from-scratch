
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

def random_argmin(A, axis=0):
    """
    Compute the argmin for a 1-D or 2-D array
    Arguments:
     - A (np.ndarray) - input array
     - axis (int)     - axis along with to find min in case of 2-D array
    """
    if len(A.shape) > 2:
        raise ValueError(f"Random argmin is not supported to arrays with more than 2 axes")
    if len(A.shape) == 1:
        p = np.where(A == np.min(A), 1, 0)
        return np.random.choice(np.arange(len(A)), p=p/np.sum(p))
    elif axis == 0:
        return np.array([random_argmin(a) for a in A.T]).T
    else:
        return np.array([random_argmin(a) for a in A])

def random_argmax(A, axis=0):
    """
    Compute the argmax for a 1-D or 2-D array
    Arguments:
     - A (np.ndarray) - input array
     - axis (int)     - axis along with to find max in case of 2-D array
    """
    if len(A.shape) > 2:
        raise ValueError(f"Random argmax is not supported to arrays with more than 2 axes")
    if len(A.shape) == 1:
        p = np.where(A == np.max(A), 1, 0)
        return np.random.choice(np.arange(len(A)), p=p/np.sum(p))
    elif axis == 0:
        return np.array([random_argmax(a) for a in A.T]).T
    else:
        return np.array([random_argmax(a) for a in A])

def simplex_linear_program(c, A, b, maximize=False):
    """
    Solve linear problem with the simplex solution to find X that minimizes c X such that A X <= b and X >= 0
    https://en.wikipedia.org/wiki/Simplex_algorithm
    """
    
    def solve_simplex(tableau, c, A, b, index=0):
        if all(c <= 0) or index == A.shape[1]:
            # Compute optimal input
            x = np.zeros((c.shape[0], 1))
            n_positive = np.where(c == 0)
            x[n_positive] = A[:,n_positive].T @ b
            return x
        M, N = A.shape
        # Select pivot column
        n_min = np.random.choice(np.where(c == np.min(c))[0])
        A_min = A[:,n_min]
        # Select pivot row
        ratios = np.where(A_min > 0, b / A_min, np.inf)
        m_min = np.random.choice(np.where(ratios == np.min(ratios))[0])
        # Update pivot row
        tableau[1+m_min] /= tableau[1+m_min, 1+n_min]
        # Update simplex tableau
        for i in range(M+1):
            if i != 1+m_min:
                tableau[i] -= tableau[1+m_min] * tableau[i, 1+n_min]
        return solve_simplex(tableau, -tableau[0, 1:N+1], tableau[1:M+1, 1:N+1], tableau[1:M+1, -1], index=index+1)
    
    if maximize:
        c = -c
    # Add slack variable expansions
    c_expanded = np.r_[c, np.zeros((A.shape[0], 1))]
    A_expanded = np.c_[A, np.eye(A.shape[0])]
    # Compute initial simplex tableau
    tableau = np.c_[
        np.r_[np.ones((1, 1)), np.zeros((A.shape[0], 1))]
        , np.r_[-c_expanded.T, A_expanded]
        , np.r_[np.zeros((1, 1)), b]
    ]
    return solve_simplex(tableau, c_expanded[:,0], A_expanded, b[:,0])[:c.shape[0]]
    
def frank_wolfe_quadratic_program(x_0, Q, c, A, b, A_eq, b_eq, max_steps=100, maximize=False, verbose=True):
    """
    Solve quadratic problem with the Frank Wolfe programm to find 
    X that minimizes 0.5 XT Q X + c X such that A X <= b and A_eq X = b_eq
    # https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm
    """
    if maximize:
        Q = - Q
        c = - c
    x = x_0.copy()
    for k in range(max_steps):
        # Compute gradient
        grad = Q @ x + c
        # Solve linear problem with gradient
        s = simplex_linear_program(grad, np.r_[A, A_eq, -A_eq], np.r_[b, b_eq, -b_eq])
        # Find best alpha to update inputs
        alphas = np.linspace(0, 1, 1000)
        f = [(0.5 * (x + alpha * (s - x)).T @ Q @ (x + alpha * (s - x)) + c.T @ (x + alpha * (s - x)))[0,0] for alpha in alphas]
        alpha = alphas[np.argmin(f)]
        if alpha == 0.0:
            return x
        # Update inputs
        x = x + alpha * (s - x)
        if verbose:
            print(f"Step {k+1}/{max_steps} - F_min = {(0.5 * x.T @ Q @ x + c.T @ x)[0,0]:.2f}")
    return x
