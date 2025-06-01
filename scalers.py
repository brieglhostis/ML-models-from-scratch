
import numpy as np


class NormalScaler:
    """
    Normal scaler class to transform X into:
    X_scaled = (X - X_mean) / (X_var + epsilon)**0.5
    """
    
    def __init__(self, axis=0, epsilon=1e-6):
        self.axis = axis
        self.epsilon = epsilon
        self.mean = None
        self.var = None
        self.std = None
        
    def fit(self, x):
        self.mean = np.mean(x, axis=self.axis, keepdims=True)
        self.var = np.var(x, axis=self.axis, keepdims=True)
        self.std = np.sqrt(self.var + self.epsilon)
        
    def transform(self, x):
        return (x - self.mean) / self.std
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
