
import numpy as np


class LayerNormalization:
    
    def __init__(self, epsilon=1e-3):
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None
    
    def forward(self, X):
        if self.gamma is None or self.beta is None:
            self.gamma = 1.0 * np.ones([1]*(len(X.shape)-1)+[X.shape[-1]])
            self.beta = 0.0 * self.gamma
        mu = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        return self.gamma * (X - mu) / np.sqrt(var + self.epsilon) + self.beta
    
    def backward(self, X, Y_error, l2=0.0):
        mu = np.mean(X, axis=-1, keepdims=True)
        var = np.var(X, axis=-1, keepdims=True)
        safe_sigma = np.sqrt(var + self.epsilon)
        X_norm = (X - mu) / safe_sigma
        # Parameter gradients
        gamma_gradient = np.sum(Y_error * X_norm, axis=(0,1), keepdims=True) + 2*l2 * self.gamma / self.gamma.shape[-1]
        beta_gradient = np.sum(Y_error, axis=(0,1), keepdims=True) + 2*l2 * self.beta / self.beta.shape[-1]
        # Propagated error
        F = X.shape[-1]
        X_error = (
            F * Y_error 
            - np.sum(Y_error, axis=-1, keepdims=True) 
            - np.sum(Y_error * X_norm, axis=-1, keepdims=True) * (X_norm - np.sum(X_norm, axis=-1, keepdims=True) / F)
        ) * self.gamma / safe_sigma / F
        return {'gamma_gradient': gamma_gradient, 'beta_gradient': beta_gradient}, X_error
    
    def update(self, gamma_gradient, beta_gradient):
        self.gamma += gamma_gradient
        self.beta += beta_gradient


class BatchNormalization:
    
    def __init__(self, epsilon=1e-3, alpha=0.0):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = None
        self.beta = None
        self.mu = None
        self.sigma = None
        
    def forward(self, X, inference=True):
        assert len(X.shape) in [2, 3]
        agg_axis = 0 if len(X.shape) == 2 else (0,1)
        if self.gamma is None or self.beta is None:
            self.gamma = 1.0 * np.ones((1, X.shape[-1]) if len(X.shape) == 2 else (1, 1, X.shape[-1]))
            self.beta = 0.0 * self.gamma
        if self.alpha == 0.0 or self.mu is None or self.sigma is None:
            self.mu = np.mean(X, axis=agg_axis, keepdims=True)
            self.var = np.var(X, axis=agg_axis, keepdims=True)
            self.sigma = np.sqrt(self.var)
        elif not inference:
            self.mu = self.alpha * self.mu + (1-self.alpha) * np.mean(X, axis=agg_axis, keepdims=True)
            self.sigma = self.alpha * self.sigma + (1-self.alpha) * np.std(X, axis=agg_axis, keepdims=True)
            self.var = np.square(self.sigma)
        return self.gamma * (X-self.mu) / np.sqrt(self.var + self.epsilon) + self.beta
    
    def backward(self, X, Y_error, l2=0.0):
        assert len(X.shape) in [2, 3]
        agg_axis = 0 if len(X.shape) == 2 else (0,1)
        if self.alpha == 0.0 or self.mu is None or self.sigma is None:
            self.mu = np.mean(X, axis=agg_axis, keepdims=True)
            self.var = np.var(X, axis=agg_axis, keepdims=True)
            self.sigma = np.sqrt(self.var)
        safe_sigma = np.sqrt(self.var + self.epsilon)
        X_norm = (X-self.mu) / safe_sigma
        # Parameter gradients
        gamma_gradient = np.sum(Y_error * X_norm, axis=agg_axis, keepdims=True) + 2*l2 * self.gamma / self.gamma.shape[-1]
        beta_gradient = np.sum(Y_error, axis=agg_axis, keepdims=True) + 2*l2 * self.beta / self.beta.shape[-1]
        # Propagated error
        N = X.shape[0] if len(X.shape) == 2 else X.shape[0] * X.shape[1]
        X_error = (
            N * Y_error 
            - (1-self.alpha) * np.sum(Y_error, axis=agg_axis, keepdims=True) 
            - (1-self.alpha) * np.sum(Y_error * X_norm, axis=agg_axis, keepdims=True) * (X_norm - np.sum(X_norm, axis=agg_axis, keepdims=True) / N)
        ) * self.gamma / safe_sigma / N
        return {'gamma_gradient': gamma_gradient, 'beta_gradient': beta_gradient}, X_error
    
    def update(self, gamma_gradient, beta_gradient):
        self.gamma += gamma_gradient
        self.beta += beta_gradient 
