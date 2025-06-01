
import numpy as np


class BatchNormalization:
    
    def __init__(self, F=None, alpha=0.0):
        self.F = F
        self.alpha = alpha
        self.gamma = 1.0
        self.beta = 0.0
        self.mu = None
        self.sigma = None
        
    def forward(self, X, inference=True):
        if self.F is None:
            self.F = X.shape[-1]
            self.gamma = self.gamma * np.ones((1, self.F))
            self.beta = self.beta * np.ones((1, self.F))
        if self.alpha == 0.0 or self.sigma is None:
            self.mu = X.mean(axis=0, keepdims=True)
            self.sigma = X.std(axis=0, keepdims=True)
        elif not inference:
            self.mu = self.alpha * self.mu + (1-self.alpha) *  X.mean(axis=0, keepdims=True)
            self.sigma = self.alpha * self.sigma + (1-self.alpha) *  X.std(axis=0, keepdims=True)
        return self.gamma * (X-self.mu) / np.sqrt(np.square(self.sigma) + 1e-6) + self.beta
    
    def backward(self, X, X_norm_error):
        if self.alpha == 0.0 or self.sigma is None:
            self.mu = X.mean(axis=0, keepdims=True)
            self.sigma = X.std(axis=0, keepdims=True)
        safe_sigma = np.sqrt(np.square(self.sigma) + 1e-6)
        X_norm = (X-self.mu) / safe_sigma
        gamma_gradient = (X_norm_error * X_norm).sum(axis=0, keepdims=True)
        beta_gradient = X_norm_error.sum(axis=0, keepdims=True)
        error = X_norm_error * self.gamma / safe_sigma
        return (gamma_gradient, beta_gradient), error
    
    def update(self, gamma_gradient, beta_gradient):
        self.gamma += gamma_gradient
        self.beta += beta_gradient


class RecurrentBatchNormalization:
    
    def __init__(self, T=None, F=None, alpha=0.0):
        self.T = T
        self.F = F
        self.alpha = alpha
        self.gamma = 1.0
        self.beta = 0.0
        self.mu = None
        self.sigma = None
        
    def forward(self, X, inference=True):
        if self.T is None or self.F is None:
            self.T = X.shape[-2]
            self.F = X.shape[-1]
            self.gamma = self.gamma * np.ones((1, self.T, self.F))
            self.beta = self.beta * np.ones((1, self.T, self.F))
        if self.alpha == 0.0 or self.sigma is None:
            self.mu = X.mean(axis=0, keepdims=True)
            self.sigma = X.std(axis=0, keepdims=True)
        elif not inference:
            self.mu = self.alpha * self.mu + (1-self.alpha) *  X.mean(axis=0, keepdims=True)
            self.sigma = self.alpha * self.sigma + (1-self.alpha) *  X.std(axis=0, keepdims=True)
        return self.gamma * (X-self.mu) / np.sqrt(np.square(self.sigma) + 1e-6) + self.beta
    
    def backward(self, X, X_norm_error):
        if self.alpha == 0.0 or self.sigma is None:
            self.mu = X.mean(axis=0, keepdims=True)
            self.sigma = X.std(axis=0, keepdims=True)
        safe_sigma = np.sqrt(np.square(self.sigma) + 1e-6)
        X_norm = (X-self.mu) / safe_sigma
        gamma_gradient = (X_norm_error * X_norm).sum(axis=0, keepdims=True)
        beta_gradient = X_norm_error.sum(axis=0, keepdims=True)
        error = X_norm_error * self.gamma / safe_sigma
        return (gamma_gradient, beta_gradient), error
    
    def update(self, gamma_gradient, beta_gradient):
        self.gamma += gamma_gradient
        self.beta += beta_gradient
 