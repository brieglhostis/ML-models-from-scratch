
import numpy as np


class LayerNormalization:
    
    def __init__(self, epsilon=1e-3):
        self.F = None
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None
    
    def forward(self, X):
        if self.gamma is None or self.beta is None:
            self.F = X.shape[-1]
            self.gamma = 1.0 * np.ones([1]*(len(X.shape)-1)+[self.F])
            self.beta = 0.0 * self.gamma
        mu = X.mean(axis=-1, keepdims=True)
        var = X.var(axis=-1, keepdims=True)
        return self.gamma * (X - mu) / np.sqrt(var + self.epsilon) + self.beta
    
    def backward(self, X, Y_error, l2=0.0):
        mu = X.mean(axis=-1, keepdims=True)
        var = X.var(axis=-1, keepdims=True)
        safe_sigma = np.sqrt(var + self.epsilon)
        X_norm = (X - mu) / safe_sigma
        gamma_gradient = np.sum(Y_error * X_norm, axis=(0,1), keepdims=True) + 2*l2 * self.gamma / self.F
        beta_gradient = np.sum(Y_error, axis=(0,1), keepdims=True) + 2*l2 * self.beta / self.F
        error = self.F * Y_error * self.gamma - Y_error @ np.transpose(self.gamma, axes=(0,2,1)) - X_norm * np.sum(X_norm * Y_error, axis=-1, keepdims=True)
        error = error / safe_sigma / self.F
        return {'gamma_gradient': gamma_gradient, 'beta_gradient': beta_gradient}, error
    
    def update(self, gamma_gradient, beta_gradient):
        self.gamma += gamma_gradient
        self.beta += beta_gradient


class BatchNormalization:
    
    def __init__(self, F=None, epsilon=1e-6, alpha=0.0):
        self.F = F
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = None
        self.beta = None
        self.mu = None
        self.sigma = None
        
    def forward(self, X, inference=True):
        if self.F is None:
            self.F = X.shape[-1]
            self.gamma = 1.0 * np.ones((1, self.F))
            self.beta = 0.0 * np.ones((1, self.F))
        if self.alpha == 0.0 or self.sigma is None:
            self.mu = X.mean(axis=0, keepdims=True)
            self.sigma = X.std(axis=0, keepdims=True)
        elif not inference:
            self.mu = self.alpha * self.mu + (1-self.alpha) *  X.mean(axis=0, keepdims=True)
            self.sigma = self.alpha * self.sigma + (1-self.alpha) *  X.std(axis=0, keepdims=True)
        return self.gamma * (X-self.mu) / np.sqrt(np.square(self.sigma) + self.epsilon) + self.beta
    
    def backward(self, X, Y_error):
        if self.alpha == 0.0 or self.sigma is None:
            self.mu = X.mean(axis=0, keepdims=True)
            self.sigma = X.std(axis=0, keepdims=True)
        safe_sigma = np.sqrt(np.square(self.sigma) + self.epsilon)
        X_norm = (X-self.mu) / safe_sigma
        gamma_gradient = (Y_error * X_norm).sum(axis=0, keepdims=True)
        beta_gradient = Y_error.sum(axis=0, keepdims=True)
        error = Y_error * self.gamma / safe_sigma
        return (gamma_gradient, beta_gradient), error
    
    def update(self, gamma_gradient, beta_gradient):
        self.gamma += gamma_gradient
        self.beta += beta_gradient


class RecurrentBatchNormalization:
    
    def __init__(self, T=None, F=None, epsilon=1e-6, alpha=0.0):
        self.T = T
        self.F = F
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = None
        self.beta = None
        self.mu = None
        self.sigma = None
        
    def forward(self, X, inference=True):
        if self.T is None or self.F is None:
            self.T = X.shape[-2]
            self.F = X.shape[-1]
            self.gamma = 1.0 * np.ones((1, self.T, self.F))
            self.beta = 0.0 * np.ones((1, self.T, self.F))
        if self.alpha == 0.0 or self.sigma is None:
            self.mu = X.mean(axis=0, keepdims=True)
            self.sigma = X.std(axis=0, keepdims=True)
        elif not inference:
            self.mu = self.alpha * self.mu + (1-self.alpha) *  X.mean(axis=0, keepdims=True)
            self.sigma = self.alpha * self.sigma + (1-self.alpha) *  X.std(axis=0, keepdims=True)
        return self.gamma * (X-self.mu) / np.sqrt(np.square(self.sigma) + self.epsilon) + self.beta
    
    def backward(self, X, Y_error):
        if self.alpha == 0.0 or self.sigma is None:
            self.mu = X.mean(axis=0, keepdims=True)
            self.sigma = X.std(axis=0, keepdims=True)
        safe_sigma = np.sqrt(np.square(self.sigma) + self.epsilon)
        X_norm = (X-self.mu) / safe_sigma
        gamma_gradient = (Y_error * X_norm).sum(axis=0, keepdims=True)
        beta_gradient = Y_error.sum(axis=0, keepdims=True)
        error = Y_error * self.gamma / safe_sigma
        return (gamma_gradient, beta_gradient), error
    
    def update(self, gamma_gradient, beta_gradient):
        self.gamma += gamma_gradient
        self.beta += beta_gradient
 