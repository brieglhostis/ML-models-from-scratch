
import numpy as np


class GradientDescentOptimizer:
    """
    Regular gradient descent optimizer:
    W[t] = W[t-1] - lr * G[t]
    """

    def update(self, gradients, learning_rate=0.1):
        """
        Compute updated gradients
        Arguments:
         - gradients (dict[str->Any[list, np.ndarray]]) - dictionnary of gradients to update
         - learning_rate (float)                        - gradient descent learning rate
        """
        updated_gradients = {}
        for gradient_name in gradients:
            if isinstance(gradients[gradient_name], list):
                # For lists, iterate update through the list
                updated_gradients[gradient_name] = [
                    self.update(gradient, learning_rate=learning_rate) 
                    for gradient in gradients[gradient_name]]
            else:
                updated_gradients[gradient_name] = learning_rate * gradients[gradient_name]
        return updated_gradients


class AdaGradOptimizer:
    """
    AdaGrad gradient descent optimizer:
    G²[t] = G²[t-1] + G[t]²
    W[t] = W[t-1] - lr * G[t] / sqrt(G²[t] + eps)
    """
    
    def __init__(self, epsilon=1e-6):
        """
        Initialize object with epsilon and an empty dictionary for G²
        Arguments:
         - epsilon (float) - safety parameter in sqrt function
        """
        self.epsilon = epsilon
        self.gradient_norms = {}
        
    def update_sub_gradients(self, gradients, gradient_norms, learning_rate=0.1):
        """
        Compute updated gradients from gradients and gradient norms
        Arguments:
         - gradients (dict[str->Any[list, np.ndarray]])      - dictionnary of gradients to update
         - gradient_norms (dict[str->Any[list, np.ndarray]]) - dictionnary of past gradients norms
         - learning_rate (float)                             - gradient descent learning rate
        """
        updated_gradients = {}
        updated_gradient_norms = {}
        for gradient_name in gradients:
            if isinstance(gradients[gradient_name], list):
                # For lists, iterate update through the list
                updated_gradients[gradient_name] = [None] * len(gradients[gradient_name])
                updated_gradient_norms[gradient_name] = [None] * len(gradients[gradient_name])
                sub_gradient_norms = gradient_norms[gradient_name] if gradient_name in gradient_norms else [{}] * len(gradients[gradient_name])
                assert len(gradients[gradient_name]) == len(sub_gradient_norms)
                for i in range(len(gradients[gradient_name])):
                    updated_gradients[gradient_name][i], updated_gradient_norms[gradient_name][i] = self.update_sub_gradients(
                        gradients[gradient_name][i], sub_gradient_norms[i], learning_rate=learning_rate)
            else:
                updated_gradient_norms[gradient_name] = (
                    gradient_norms[gradient_name] 
                    if gradient_name in gradient_norms 
                    else 0.0) + np.square(gradients[gradient_name]) 
                updated_gradients[gradient_name] = learning_rate * gradients[gradient_name] / np.sqrt(updated_gradient_norms[gradient_name] + self.epsilon)
        return updated_gradients, updated_gradient_norms
    
    def update(self, gradients, learning_rate=0.1):
        """
        Compute updated gradients and update gradient norms
        Arguments:
         - gradients (dict[str->Any[list, np.ndarray]]) - dictionnary of gradients to update
         - learning_rate (float)                        - gradient descent learning rate
        """
        updated_gradients, self.gradient_norms = self.update_sub_gradients(
            gradients, self.gradient_norms, learning_rate=learning_rate)
        return updated_gradients


class RMSPropOptimizer:
    """
    RMSProp gradient descent optimizer:
    G²[t] = Beta * G²[t-1] + (1-Beta) * G[t]²
    W[t] = W[t-1] - lr * G[t] / sqrt(G²[t] + eps)
    """
    
    def __init__(self, gradient_decay=0.9, epsilon=1e-6):
        """
        Initialize object with beta, epsilon and an empty dictionary for G²
        Arguments:
         - gradient_decay (float) - exponential decay parameter for gradient norms
         - epsilon (float)        - safety parameter in sqrt function
        """
        self.gradient_decay = gradient_decay
        self.epsilon = epsilon
        self.gradient_norms = {}
        
    def update_sub_gradients(self, gradients, gradient_norms, learning_rate=0.1):
        """
        Compute updated gradients from gradients and gradient norms
        Arguments:
         - gradients (dict[str->Any[list, np.ndarray]])      - dictionnary of gradients to update
         - gradient_norms (dict[str->Any[list, np.ndarray]]) - dictionnary of past gradients norms
         - learning_rate (float)                             - gradient descent learning rate
        """
        updated_gradients = {}
        updated_gradient_norms = {}
        for gradient_name in gradients:
            if isinstance(gradients[gradient_name], list):
                # For lists, iterate update through the list
                updated_gradients[gradient_name] = [None] * len(gradients[gradient_name])
                updated_gradient_norms[gradient_name] = [None] * len(gradients[gradient_name])
                sub_gradient_norms = gradient_norms[gradient_name] if gradient_name in gradient_norms else [{}] * len(gradients[gradient_name])
                assert len(gradients[gradient_name]) == len(sub_gradient_norms)
                for i in range(len(gradients[gradient_name])):
                    updated_gradients[gradient_name][i], updated_gradient_norms[gradient_name][i] = self.update_sub_gradients(
                        gradients[gradient_name][i], sub_gradient_norms[i], learning_rate=learning_rate)
            else:
                updated_gradient_norms[gradient_name] = self.gradient_decay * (
                    gradient_norms[gradient_name] 
                    if gradient_name in gradient_norms 
                    else 0.0) + (1-self.gradient_decay) * np.square(gradients[gradient_name]) 
                updated_gradients[gradient_name] = learning_rate * gradients[gradient_name] / np.sqrt(updated_gradient_norms[gradient_name] + self.epsilon)
        return updated_gradients, updated_gradient_norms
    
    def update(self, gradients, learning_rate=0.1):
        """
        Compute updated gradients and update gradient norms
        Arguments:
         - gradients (dict[str->Any[list, np.ndarray]]) - dictionnary of gradients to update
         - learning_rate (float)                        - gradient descent learning rate
        """
        updated_gradients, self.gradient_norms = self.update_sub_gradients(
            gradients, self.gradient_norms, learning_rate=learning_rate)
        return updated_gradients


class AdamOptimizer:
    """
    Adam gradient descent optimizer:
    MA[t] = beta1 * MA[t-1] + (1-beta1) * G[t]
    G²[t] = beta2 * G²[t-1] + (1-beta2) * G[t]²
    MA_unbiased[t] = MA[t] / (1-beta1**(t+1))
    G²_unbiased[t] = G²[t] / (1-beta2**(t+1))
    W[t] = W[t-1] - lr * MA_unbiased[t] / sqrt(G²_unbiased[t] + eps)
    """
    
    def __init__(self, gradient_decay=0.9, gradient_norm_decay=0.999, epsilon=1e-6):
        """
        Initialize object with beta1, beta2, epsilon and an empty dictionary for MA and G²
        Arguments:
         - gradient_decay (float)      - exponential decay parameter for gradient moving average
         - gradient_norm_decay (float) - exponential decay parameter for gradient norms
         - epsilon (float)             - safety parameter in sqrt function
        """
        self.gradient_decay = gradient_decay
        self.gradient_norm_decay = gradient_norm_decay
        self.epsilon = epsilon
        self.gradient_moving_averages = {}
        self.gradient_norms = {}
        self.t = 0
        
    def update_sub_gradients(self, gradients, gradient_moving_averages, gradient_norms, learning_rate=0.1):
        """
        Compute updated gradients from gradients, gradient moving averages, and gradient norms
        Arguments:
         - gradients (dict[str->Any[list, np.ndarray]])                - dictionnary of gradients to update
         - gradient_moving_averages (dict[str->Any[list, np.ndarray]]) - dictionnary of past gradients moving averages
         - gradient_norms (dict[str->Any[list, np.ndarray]])           - dictionnary of past gradients norms
         - learning_rate (float)                                       - gradient descent learning rate
        """
        updated_gradients = {}
        updated_gradient_moving_averages = {}
        updated_gradient_norms = {}
        for gradient_name in gradients:
            if isinstance(gradients[gradient_name], list):
                # For lists, iterate update through the list
                updated_gradients[gradient_name] = [None] * len(gradients[gradient_name])
                updated_gradient_moving_averages[gradient_name] = [None] * len(gradients[gradient_name])
                updated_gradient_norms[gradient_name] = [None] * len(gradients[gradient_name])
                sub_gradient_moving_averages = gradient_moving_averages[gradient_name] if gradient_name in gradient_moving_averages else [{}] * len(gradients[gradient_name])
                sub_gradient_norms = gradient_norms[gradient_name] if gradient_name in gradient_norms else [{}] * len(gradients[gradient_name])
                assert len(gradients[gradient_name]) == len(sub_gradient_moving_averages)
                assert len(gradients[gradient_name]) == len(sub_gradient_norms)
                for i in range(len(gradients[gradient_name])):
                    updated_gradients[gradient_name][i], updated_gradient_moving_averages[gradient_name][i], updated_gradient_norms[gradient_name][i] = self.update_sub_gradients(
                        gradients[gradient_name][i], sub_gradient_moving_averages[i], sub_gradient_norms[i], learning_rate=learning_rate)
            else:
                updated_gradient_moving_averages[gradient_name] = self.gradient_decay * (
                    gradient_moving_averages[gradient_name] 
                    if gradient_name in gradient_moving_averages 
                    else 0.0) + (1-self.gradient_decay) * gradients[gradient_name] 
                updated_gradient_norms[gradient_name] = self.gradient_norm_decay * (
                    gradient_norms[gradient_name] 
                    if gradient_name in gradient_norms 
                    else 0.0) + (1-self.gradient_norm_decay) * np.square(gradients[gradient_name]) 
                unbiased_gradient_moving_average = updated_gradient_moving_averages[gradient_name] / (1 - self.gradient_decay**(self.t+1))
                unbiased_gradient_norm = updated_gradient_norms[gradient_name] / (1 - self.gradient_norm_decay**(self.t+1))
                updated_gradients[gradient_name] = learning_rate * unbiased_gradient_moving_average / np.sqrt(unbiased_gradient_norm + self.epsilon)
        return updated_gradients, updated_gradient_moving_averages, updated_gradient_norms
    
    def update(self, gradients, learning_rate=0.1):
        """
        Compute updated gradients and update gradient moving averages and norms
        Arguments:
         - gradients (dict[str->Any[list, np.ndarray]]) - dictionnary of gradients to update
         - learning_rate (float)                        - gradient descent learning rate
        """
        updated_gradients, self.gradient_moving_averages, self.gradient_norms = self.update_sub_gradients(
            gradients, self.gradient_moving_averages, self.gradient_norms, learning_rate=learning_rate)
        self.t += 1
        return updated_gradients
