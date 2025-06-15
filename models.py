
import numpy as np

from utils import mse, r_square
from optimizers import GradientDescentOptimizer, AdaGradOptimizer, RMSPropOptimizer, AdamOptimizer
from activations import LinearActivation, ReLuActivation, TanHActivation
from layers import DenseLayer, RecurrentLayer, LSTMLayer
from normalization import BatchNormalization, RecurrentBatchNormalization


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


class RegressionNeuralNetwork:
    """
    Regression Neural Network model:
    Y = activation_f(X W + b)
    Generalised to N layers
    """
    
    def __init__(
        self, F, D, layer_sizes=[10], add_bias=True, activations=[ReLuActivation, LinearActivation]
        , l2=0.0, dropout_p=0.0, batch_normalization=False, batch_normalization_alpha=0.0):
        """
        Initialize the neural network object
        Arguments:
         - F (int)                           - input dimension size
         - D (int)                           - output dimension size
         - layer_sizes (list[int])           - hidden layers dimension sizes
         - add_bias (Any[bool, list[bool]])  - set to True to add bias in all layers, or use list to specify layer by layer
         - activations (list[Activation])    - list of layer activation functions
         - l2 (float)                        - L2 regularization parameter
         - dropout (Any[float, list[float]]) - dropout rate in all layers, or use list to specify layer by layer
         - batch_normalization (bool)        - set to True to add batch normalization to all layers except the output layer
         - batch_normalization_alpha (float) - batch normalization moving average exponential decay parameter
        """
        
        self.F = F
        self.D = D
        self.l2 = l2
        
        # Initialize layers
        self.layer_sizes = layer_sizes
        self.n_layers = len(self.layer_sizes)+1
        add_bias = add_bias if isinstance(add_bias, list) else [add_bias] * self.n_layers
        dropout_p = dropout_p if isinstance(dropout_p, list) else [dropout_p] * self.n_layers
        assert len(add_bias) == self.n_layers
        assert len(dropout_p) == self.n_layers
        self.layers = [
            DenseLayer(f, d, b, dropout_p=p) for f, d, b, p in zip([F]+self.layer_sizes, self.layer_sizes+[D], add_bias, dropout_p)
        ]
        self.activations = [
            a() for a in activations
        ]
        self.batch_normalization_layers = [
            BatchNormalization(alpha=batch_normalization_alpha) for i in range(self.n_layers-1)
        ] + [None] if batch_normalization else [None] * self.n_layers

        self.history = []
        
    def loss(self, X, Y):
        """
        Compute the prediction MSE
        Arguments:
         - X (np.ndarray) - input features (NxF)
         - Y (np.ndarray) - target actuals (NxD)
        """
        Y_pred = self.predict(X)
        return mse(Y, Y_pred)
    
    def predict(self, X):
        """
        Compute predictions for a set of input samples 
        Arguments:
         - X (np.ndarray) - input features (NxF)
        """
        return self.forward(X, inference=True)[-1]
    
    def forward(self, X, inference=True):
        """
        Forward pass all layers 
        Arguments:
         - X (np.ndarray)   - input features (NxF)
         - inference (bool) - set to True to avoid returning intermediate layer results
        """
        layer_input = X.copy()
        layer_inputs = [] if inference else [layer_input]
        for layer, activation, batch_normalization in zip(self.layers, self.activations, self.batch_normalization_layers):
            # Layer forward pass
            activation_input = layer.forward(layer_input, inference=inference) 
            # Activation forward pass
            batch_normalization_input = activation.forward(activation_input) 
            # Batch norm forward pass
            layer_input = batch_normalization.forward(batch_normalization_input, inference=inference) if batch_normalization is not None else batch_normalization_input
            if not inference:
                layer_inputs += [activation_input, batch_normalization_input, layer_input]
        if inference:
            layer_inputs.append(layer_input)
        return layer_inputs
    
    def fit(
        self, X, Y, X_val=None, Y_val=None, learning_rate=0.1, epochs=100, batch_size=1000
        , optimizer='Adam', gradient_decay=0.9, gradient_norm_decay=0.999, verbose=True):
        """
        Fit neural network using gradient descent to minimize loss
        Arguments:
         - X (np.ndarray)              - input training features (NxF)
         - Y (np.ndarray)              - target training actuals (NxD)
         - X_val (np.ndarray)          - input validation actuals (N'xF)
         - Y_val (np.ndarray)          - target validation actuals (N'xD)
         - learning_rate (float)       - learning rate in gradient descent
         - epochs (int)                - number of optimization steps
         - batch_size (int)            - size of batches in batch gradient descent
         - optimizer (str)             - name of the optimizer to use among 'Regular', 'AdaGrad', 'RMSProp', or 'Adam'
         - gradient_decay (float)      - exponential gradient decay parameter for RMSProp and Adam
         - gradient_norm_decay (float) - exponential gradient norm decay parameter for Adam
         - verbose (float)             - set to True to print intermediary evaluation metrics
        """
        
        # Optimizers initialization
        if optimizer == 'Regular':
            optimizer = GradientDescentOptimizer()
            batch_norm_optimizer = GradientDescentOptimizer()
        elif optimizer == 'AdaGrad':
            optimizer = AdaGradOptimizer()
            batch_norm_optimizer = AdaGradOptimizer()
        elif optimizer == 'RMSProp':
            optimizer = RMSPropOptimizer(gradient_decay=gradient_decay)
            batch_norm_optimizer = RMSPropOptimizer(gradient_decay=gradient_decay)
        elif optimizer == 'Adam':
            optimizer = AdamOptimizer(gradient_decay=gradient_decay, gradient_norm_decay=gradient_norm_decay)
            batch_norm_optimizer = AdamOptimizer(gradient_decay=gradient_decay, gradient_norm_decay=gradient_norm_decay)
        else:
            raise ValueError(f"Optimizer must be one of 'Regular', 'AdaGrad', 'RMSProp', or 'Adam'")
        
        # Batch creation
        if batch_size >= X.shape[0]:
            X_batches = [X]
            Y_batches = [Y]
        else:
            X_batches = [X[i*batch_size:min(X.shape[0],(i+1)*batch_size)] for i in range(X.shape[0] // batch_size)]
            Y_batches = [Y[i*batch_size:min(X.shape[0],(i+1)*batch_size)] for i in range(X.shape[0] // batch_size)]
            
        self.history = []
        for e in range(epochs):
            # Fitting
            for X_batch, Y_batch in zip(X_batches, Y_batches):
                # Forward pass
                layer_inputs = self.forward(X_batch, inference=False)
                
                # Back propagation
                Y_pred = layer_inputs.pop(-1)
                layer_error = (Y_pred - Y_batch) / Y_batch.shape[0]
                layer_gradients = []
                batch_normalization_gradients = []
                activation_inputs = layer_inputs[1::3]
                batch_normalization_inputs = layer_inputs[2::3]
                layer_inputs = layer_inputs[0::3]
                for layer, activation, batch_normalization, layer_input, activation_input, batch_normalization_input in zip(
                    self.layers[::-1], self.activations[::-1], self.batch_normalization_layers[::-1], layer_inputs[::-1], activation_inputs[::-1], batch_normalization_inputs[::-1]):
                    if batch_normalization is None:
                        batch_normalization_gradient = {}
                    else:
                        batch_normalization_gradient, layer_error = batch_normalization.backward(batch_normalization_input, layer_error, l2=self.l2)
                    layer_error = activation.backward(activation_input) * layer_error
                    layer_gradient, layer_error = layer.backward(layer_input, layer_error, l2=self.l2)
                    layer_gradients = [layer_gradient] + layer_gradients
                    batch_normalization_gradients = [batch_normalization_gradient] + batch_normalization_gradients
                
                # Layer updates
                layer_gradients = optimizer.update({'gradients': layer_gradients}, learning_rate=learning_rate)
                for layer, layer_gradient in zip(self.layers, layer_gradients['gradients']):
                    layer.update(**layer_gradient)
                
                # Batch normalization updates
                batch_normalization_gradients = batch_norm_optimizer.update({'gradients': batch_normalization_gradients}, learning_rate=learning_rate)
                for batch_norm, batch_norm_gradient in zip(self.batch_normalization_layers, batch_normalization_gradients['gradients']):
                    if batch_norm is not None:
                        batch_norm.update(**batch_norm_gradient)
                        
            # Evaluation
            Y_pred = self.predict(X)
            metrics = {
                'train_mse': mse(Y, Y_pred)
                , 'train_r_square': r_square(Y, Y_pred)
            }
            if X_val is not None and Y_val is not None:
                Y_pred_val = self.predict(X_val)
                metrics['val_mse'] = mse(Y_val, Y_pred_val)
                metrics['val_r_square'] = r_square(Y_val, Y_pred_val)
            self.history.append(metrics)
            
            # Verbose
            if verbose and (e+1) % 10 == 0:
                print(f"Epoch {e+1}/{epochs} - train loss = {metrics['train_mse'].mean():.4f}" + (f", validation loss = {metrics['val_mse'].mean():.4f}" if 'val_mse' in metrics else ""))
            
        return self.history


class RegressionRecurrentNeuralNetwork:
    """
    Recurrent regression Neural Network model:
    Y = activation_f(X W + b)
    Generalised to N layers and with X and Y being sequential arrays
    """
    
    LAYER_TYPES = ['Dense', 'Recurrent', 'LSTM', 'BidirectionalRecurrent', 'BidirectionalLSTM']
    
    def __init__(
        self, F, D, layer_sizes=[10], layer_types=['Recurrent', 'Dense'], add_bias=[True, True], activations=[ReLuActivation, LinearActivation]
        , l2=0.0, dropout_p=0.0, batch_normalization=False, batch_normalization_alpha=0.0):
        """
        Initialize the neural network object
        Arguments:
         - F (int)                                     - input dimension size
         - D (int)                                     - output dimension size
         - layer_sizes (list[int])                     - hidden layers dimension sizes
         - layer_types (list[str])                     - hidden layers types among 'Dense', 'Recurrent', 'LSTM', 'BidirectionalRecurrent', or 'BidirectionalLSTM'
         - add_bias (Any[bool, list[bool]])            - set to True to add bias in all layers, or use list to specify layer by layer
         - activations (list[Activation])              - list of layer activation functions
         - l2 (float)                                  - L2 regularization parameter
         - dropout (Any[float, list[float]])           - dropout rate in all layers, or use list to specify layer by layer
         - batch_normalization (Any[bool, list[bool]]) - set to True to add batch normalization to all layers except the output layer, or use list to specify layer by layer
         - batch_normalization_alpha (float)           - batch normalization moving average exponential decay parameter
        """
        
        assert all([t in self.LAYER_TYPES for t in layer_types])
        
        self.F = F
        self.D = D
        self.l2 = l2
        
        # Initialize layers
        self.layer_sizes = layer_sizes
        self.n_layers = len(self.layer_sizes)+1
        add_bias = add_bias if isinstance(add_bias, list) else [add_bias] * self.n_layers
        dropout_p = dropout_p if isinstance(dropout_p, list) else [dropout_p] * self.n_layers
        batch_normalization = batch_normalization if isinstance(batch_normalization, list) else [batch_normalization] * (self.n_layers-1)
        assert len(add_bias) == self.n_layers
        assert len(dropout_p) == self.n_layers
        assert len(batch_normalization) == self.n_layers-1
        self.layers = [
            RecurrentLayer(f, d, b, dropout_p=p) if t == 'Recurrent' else 
            LSTMLayer(f, d, b, dropout_p=p) if t == 'LSTM' else 
            BidirectionalRecurrentLayer(f, d, b, dropout_p=p) if t == 'BidirectionalRecurrent' else 
            BidirectionalLSTMLayer(f, d, b, dropout_p=p) if t == 'BidirectionalLSTM' else 
            DenseLayer(f, d, b, dropout_p=p)
            for f, d, t, b, p in zip([F]+self.layer_sizes, self.layer_sizes+[D], layer_types, add_bias, dropout_p)
        ]
        self.activations = [
            a() for a in activations
        ]
        self.batch_normalization_layers = [
            BatchNormalization(alpha=batch_normalization_alpha) if b else None
            for b in batch_normalization
        ] + [None]
        
        self.history = []
        
    def loss(self, X, Y):
        """
        Compute the prediction MSE
        Arguments:
         - X (np.ndarray) - input features (NxTxF)
         - Y (np.ndarray) - target actuals (NxTxD)
        """
        Y_pred = self.predict(X)
        return mse(Y, Y_pred, axis=(0,1))
    
    def predict(self, X):
        """
        Compute predictions for a set of input samples 
        Arguments:
         - X (np.ndarray) - input features (NxTxF)
        """
        return self.forward(X, inference=True)[-1]
    
    def forward(self, X, inference=True):
        """
        Forward pass all layers 
        Arguments:
         - X (np.ndarray)   - input features (NxTxF)
         - inference (bool) - set to True to avoid returning intermediate layer results
        """
        layer_input = X.copy()
        layer_inputs = [] if inference else [layer_input]
        for layer, activation, batch_normalization in zip(self.layers, self.activations, self.batch_normalization_layers):
            # Layer forward pass
            activation_input = layer.forward(layer_input, inference=inference)
            # Activation forward pass
            batch_normalization_input = activation.forward(activation_input)
            # Batch norm forward pass
            layer_input = batch_normalization.forward(batch_normalization_input, inference=inference) if batch_normalization is not None else batch_normalization_input
            if not inference:
                layer_inputs += [activation_input, batch_normalization_input, layer_input]
        if inference:
            layer_inputs.append(layer_input)
        return layer_inputs
    
    def fit(
        self, X, Y, X_val=None, Y_val=None, learning_rate=0.1, epochs=100, batch_size=100
        , optimizer='Adam', gradient_decay=0.9, gradient_norm_decay=0.999, verbose=True):
        """
        Fit neural network using gradient descent to minimize loss
        Arguments:
         - X (np.ndarray)              - input training features (NxTxF)
         - Y (np.ndarray)              - target training actuals (NxTxD)
         - X_val (np.ndarray)          - input validation actuals (N'xTxF)
         - Y_val (np.ndarray)          - target validation actuals (N'xTxD)
         - learning_rate (float)       - learning rate in gradient descent
         - epochs (int)                - number of optimization steps
         - batch_size (int)            - size of batches in batch gradient descent
         - optimizer (str)             - name of the optimizer to use among 'Regular', 'AdaGrad', 'RMSProp', or 'Adam'
         - gradient_decay (float)      - exponential gradient decay parameter for RMSProp and Adam
         - gradient_norm_decay (float) - exponential gradient norm decay parameter for Adam
         - verbose (float)             - set to True to print intermediary evaluation metrics
        """
        
        # Optimizers initialization
        if optimizer == 'Regular':
            optimizer = GradientDescentOptimizer()
            batch_norm_optimizer = GradientDescentOptimizer()
        elif optimizer == 'AdaGrad':
            optimizer = AdaGradOptimizer()
            batch_norm_optimizer = AdaGradOptimizer()
        elif optimizer == 'RMSProp':
            optimizer = RMSPropOptimizer(gradient_decay=gradient_decay)
            batch_norm_optimizer = RMSPropOptimizer(gradient_decay=gradient_decay)
        elif optimizer == 'Adam':
            optimizer = AdamOptimizer(gradient_decay=gradient_decay, gradient_norm_decay=gradient_norm_decay)
            batch_norm_optimizer = AdamOptimizer(gradient_decay=gradient_decay, gradient_norm_decay=gradient_norm_decay)
        else:
            raise ValueError(f"Optimizer must be one of 'Regular', 'AdaGrad', 'RMSProp', or 'Adam'")
        
        # Batch creation
        if batch_size >= X.shape[0]:
            X_batches = [X]
            Y_batches = [Y]
        else:
            X_batches = [X[i*batch_size:min(X.shape[0],(i+1)*batch_size)] for i in range(X.shape[0] // batch_size)]
            Y_batches = [Y[i*batch_size:min(X.shape[0],(i+1)*batch_size)] for i in range(X.shape[0] // batch_size)]
        
        self.history = []
        for e in range(epochs):
            # Fitting
            for X_batch, Y_batch in zip(X_batches, Y_batches):
                # Forward pass
                layer_inputs = self.forward(X_batch, inference=False)

                # Back propagation
                Y_pred = layer_inputs.pop(-1)
                layer_error = (Y_pred - Y_batch) / Y_batch.shape[0] / Y_batch.shape[1]
                layer_gradients = []
                batch_normalization_gradients = []
                activation_inputs = layer_inputs[1::3]
                batch_normalization_inputs = layer_inputs[2::3]
                layer_inputs = layer_inputs[0::3]
                for layer, activation, batch_normalization, layer_input, activation_input, batch_normalization_input in zip(self.layers[::-1], self.activations[::-1], self.batch_normalization_layers[::-1], layer_inputs[::-1], activation_inputs[::-1], batch_normalization_inputs[::-1]):
                    if batch_normalization is None:
                        batch_normalization_gradient = {}
                    else:
                        batch_normalization_gradient, layer_error = batch_normalization.backward(batch_normalization_input, layer_error, l2=self.l2)
                    layer_error = activation.backward(activation_input) * layer_error
                    layer_gradient, layer_error = layer.backward(layer_input, layer_error, l2=self.l2)
                    layer_gradients = [layer_gradient] + layer_gradients
                    batch_normalization_gradients = [batch_normalization_gradient] + batch_normalization_gradients
                
                # Layer updates
                layer_gradients = optimizer.update({'gradients': layer_gradients}, learning_rate=learning_rate)
                for layer, layer_gradient in zip(self.layers, layer_gradients['gradients']):
                    layer.update(**layer_gradient)
                
                # Batch normalization updates
                batch_normalization_gradients = batch_norm_optimizer.update({'gradients': batch_normalization_gradients}, learning_rate=learning_rate)
                for batch_norm, batch_norm_gradient in zip(self.batch_normalization_layers, batch_normalization_gradients['gradients']):
                    if batch_norm is not None:
                        batch_norm.update(**batch_norm_gradient)
                     
            # Evaluation
            Y_pred = self.predict(X)
            metrics = {
                'train_mse': mse(Y, Y_pred, axis=(0,1))
            }
            if X_val is not None and Y_val is not None:
                Y_pred_val = self.predict(X_val)
                metrics['val_mse'] = mse(Y_val, Y_pred_val, axis=(0,1))
            self.history.append(metrics)
            
            # Verbose
            if verbose and (e+1) % 10 == 0:
                print(f"Epoch {e+1}/{epochs} - train loss = {metrics['train_mse'].mean():.4f}" + (f", validation loss = {metrics['val_mse'].mean():.4f}" if 'val_mse' in metrics else ""))
        
        return self.history
