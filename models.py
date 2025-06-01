
import numpy as np

from utils import mse


class LinearRegression:
    """
    Linear regression model:
    Y = X W + b
    """
    
    def __init__(self, add_bias=True, l1=0.0, l2=0.0):
        """
        Arguments:
         - add_bias (bool) - set to True to add a bias to the equation
         - l1 (float)      - L1 or "Lasso" regularization parameter 
         - l2 (float)      - L2 or "Ridge" regularization parameter 
        """
        self.add_bias = add_bias
        self.l1 = l1
        self.l2 = l2
        self.W = None
        
    def loss(self, X, Y):
        """
        Compute the prediction loss, including regularization
        Arguments:
         - X (np.ndarray) - input features (NxF)
         - Y (np.ndarray) - target actuals (NxD)
        """
        Y_pred = self.predict(X)
        mse_loss = mse(Y, Y_pred)
        l1_loss = self.l1 * np.mean(np.abs(self.W), axis=0)
        l2_loss = self.l2 * np.mean(np.square(self.W), axis=0)
        return mse_loss + l1_loss + l2_loss
    
    def predict(self, X):
        """
        Compute predictions for a set of input samples 
        Arguments:
         - X (np.ndarray) - input features (NxF)
         - Y (np.ndarray) - target actuals (NxD)
        """
        if self.W is None:
            raise ValueError('Cannot predict before model is fitted')
        if self.add_bias and X.shape[-1] == self.W.shape[0]-1:
            X = np.concatenate((X, np.ones(tuple(list(X.shape[:-1])+[1]))), axis=-1)
        return X@self.W
    
    def fit(self, X, Y):
        """
        Fit regression model by computing the weights that minimize the loss
        Arguments:
         - X (np.ndarray) - input features (NxF)
         - Y (np.ndarray) - target actuals (NxD)
        """
        if self.add_bias:
            X = np.concatenate((X, np.ones(tuple(list(X.shape[:-1])+[1]))), axis=-1)
        XtX_inv = np.linalg.pinv(X.T@X + self.l2*np.eye(X.shape[1]))
        self.W = XtX_inv@X.T@Y
        if self.l1 > 0.0:
            self.W = np.sign(self.W) * np.where(np.abs(self.W) - self.l1 > 0, np.abs(self.W) - self.l1, 0)
        return self.loss(X, Y)
