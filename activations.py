
import numpy as np


class LinearActivation:
    def forward(self, X):
        return X
    def backward(self, X):
        return np.ones(X.shape)
    
class ReLuActivation:
    def forward(self, X):
        return np.where(X>0, X, 0)
    def backward(self, X):
        return np.where(X>0, 1, 0)
    
class TanHActivation:
    def __init__(self, safety_threshold=200):
        self.safety_threshold = safety_threshold
        self.exp_safety_threshold = np.exp(self.safety_threshold)
    def forward(self, X):
        exp_X = np.where(X>self.safety_threshold, self.exp_safety_threshold, np.exp(X))
        exp_minus_X = np.where(X<-self.safety_threshold, self.exp_safety_threshold, np.exp(-X))
        return (exp_X-exp_minus_X) / (exp_X+exp_minus_X)
    def backward(self, X):
        return 1 - np.square(self.forward(X))
