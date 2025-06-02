
import numpy as np

from utils import mse, r_square
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


class RegressionNeuralNetwork:
    
    GRADIENT_DESCENT_METHODS = ["Regular", "RMSProp", "AdaGrad"]
    
    def __init__(
        self, F, D, layer_sizes=[10], add_bias=[True, True], activations=[ReLuActivation, LinearActivation]
        , l2=0.0, dropout_p=0.0, batch_normalization=False, batch_normalization_alpha=0.0):
        
        self.F = F
        self.D = D
        
        self.l2 = l2
        self.dropout_p = dropout_p if isinstance(dropout_p, list) else [dropout_p] * len(layer_sizes)
        self.batch_normalization = batch_normalization
        
        self.layer_sizes = layer_sizes
        self.layers = [
            DenseLayer(f, d, b, dropout_p=p) for f, d, b, p in zip([F]+self.layer_sizes, self.layer_sizes+[D], add_bias, [0.0]+self.dropout_p)
        ]
        self.activations = [
            a() for a in activations
        ]
        self.batch_normalization_layers = [
            BatchNormalization(alpha=batch_normalization_alpha) for i in range(len(self.layers)-1)
        ] + [None] if self.batch_normalization else [None] * len(self.layers)
        self.history = []
        
    def loss(self, X, Y):
        Y_pred = self.predict(X)
        return mse(Y, Y_pred)
    
    def predict(self, X):
        return self.forward(X, inference=True)[-1]
    
    def forward(self, X, inference=True):
        layer_input = X.copy()
        layer_inputs = [] if inference else [layer_input]
        for layer, activation, batch_normalization in zip(self.layers, self.activations, self.batch_normalization_layers):
            activation_input = layer.forward(layer_input, inference=inference)
            batch_normalization_input = activation.forward(activation_input)
            layer_input = batch_normalization.forward(batch_normalization_input, inference=inference) if batch_normalization is not None else batch_normalization_input
            if not inference:
                layer_inputs += [activation_input, batch_normalization_input, layer_input]
        if inference:
            layer_inputs.append(layer_input)
        return layer_inputs
    
    def fit(self, X, Y, X_val=None, Y_val=None, lr=0.1, epochs=100, batch_size=1000, gradient_descent_method="AdaGrad", gradient_decay=0.8, verbose=True):
                
        assert gradient_descent_method in self.GRADIENT_DESCENT_METHODS
        
        # Batch creation
        if batch_size >= X.shape[0]:
            X_batches = [X]
            Y_batches = [Y]
        else:
            X_batches = [X[i*batch_size:min(X.shape[0],(i+1)*batch_size)] for i in range(X.shape[0] // batch_size)]
            Y_batches = [Y[i*batch_size:min(X.shape[0],(i+1)*batch_size)] for i in range(X.shape[0] // batch_size)]
            
        self.history = []
        past_layer_gradient_norms = [{} for layer in self.layers]
        past_batch_normalization_gradient_norms = [{} for layer in self.layers]
        for e in range(epochs):
            # Fitting
            for X_batch, Y_batch in zip(X_batches, Y_batches):
                # Forward pass
                layer_inputs = self.forward(X_batch, inference=False)
                
                # Back propagation
                Y_pred = layer_inputs.pop(-1)
                layer_error = (Y_batch - Y_pred) / Y.shape[0]
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
                if gradient_descent_method == 'AdaGrad':
                    past_layer_gradient_norms = [
                        {k: (pgn[k] if k in pgn else 0.0)+np.square(g[k]) for k in g} 
                        for pgn, g in zip(past_layer_gradient_norms, layer_gradients)]
                elif gradient_descent_method == 'RMSProp':
                    past_layer_gradient_norms = [
                        {k: gradient_decay*(pgn[k] if k in pgn else 0.0)+(1-gradient_decay)*np.square(g[k]) for k in g} 
                        for pgn, g in zip(past_layer_gradient_norms, layer_gradients)]
                for layer, layer_gradient, past_layer_gradient_norm in zip(self.layers, layer_gradients, past_layer_gradient_norms):
                    if gradient_descent_method in ('AdaGrad', 'RMSProp'):
                        kwargs = {k: lr * layer_gradient[k] / np.sqrt(past_layer_gradient_norm[k]+1e-6) for k in layer_gradient}
                        layer.update(**kwargs)
                    else:
                        kwargs = {k: lr * layer_gradient[k] for k in layer_gradient}
                        layer.update(**kwargs)
                
                # Batch normalization updates
                if gradient_descent_method == 'AdaGrad':
                    past_batch_normalization_gradient_norms = [
                        {k: (pgn[k] if k in pgn else 0.0)+np.square(g[k]) for k in g} 
                        for pgn, g in zip(past_batch_normalization_gradient_norms, batch_normalization_gradients)]
                elif gradient_descent_method == 'RMSProp':
                    past_batch_normalization_gradient_norms = [
                        {k: gradient_decay*(pgn[k] if k in pgn else 0.0)+(1-gradient_decay)*np.square(g[k]) for k in g} 
                        for pgn, g in zip(past_batch_normalization_gradient_norms, batch_normalization_gradients)]
                for layer, layer_gradient, past_layer_gradient_norm in zip(self.batch_normalization_layers, batch_normalization_gradients, past_batch_normalization_gradient_norms):
                    if layer is not None:
                        if gradient_descent_method in ('AdaGrad', 'RMSProp'):
                            kwargs = {k: lr * layer_gradient[k] / np.sqrt(past_layer_gradient_norm[k]+1e-6) for k in layer_gradient}
                            layer.update(**kwargs)
                        else:
                            kwargs = {k: lr * layer_gradient[k] for k in layer_gradient}
                            layer.update(**kwargs)
                        
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
            
            if verbose and (e+1) % 10 == 0:
                print(f"Epoch {e+1}/{epochs} - train loss = {metrics['train_mse'].mean():.4f}" + (f", validation loss = {metrics['val_mse'].mean():.4f}" if 'val_mse' in metrics else ""))
            
        return self.history


class RegressionRecurrentNeuralNetwork:
    
    LAYER_TYPES = ['Dense', 'Recurrent', 'LSTM']
    GRADIENT_DESCENT_METHODS = ["Regular", "RMSProp", "AdaGrad"]
    
    def __init__(
        self, F, D, layer_sizes=[10], add_bias=[True, True], layer_types=['Recurrent', 'Dense'], activations=[ReLuActivation, LinearActivation]
        , l2=0.0, dropout_p=0.0, batch_normalization=False, batch_normalization_alpha=0.0):
        
        assert all([t in self.LAYER_TYPES for t in layer_types])
        
        self.F = F
        self.D = D
        
        self.l2 = l2
        self.dropout_p = dropout_p if isinstance(dropout_p, list) else [dropout_p] * len(layer_sizes)
        self.batch_normalization = batch_normalization
        
        self.layer_sizes = layer_sizes
        self.layers = [
            RecurrentLayer(f, d, b, dropout_p=p) if t == 'Recurrent' else LSTMLayer(f, d, b, dropout_p=p) if t == 'LSTM' else DenseLayer(f, d, b, dropout_p=p)
            for f, d, b, t, p in zip([F]+self.layer_sizes, self.layer_sizes+[D], add_bias, layer_types, [0.0]+self.dropout_p)
        ]
        self.activations = [
            a() for a in activations
        ]
        self.batch_normalization_layers = [
            BatchNormalization(alpha=batch_normalization_alpha) for i in range(len(self.layers)-1)
        ] + [None] if self.batch_normalization else [None] * len(self.layers)
        self.history = []
        
    def loss(self, X, Y):
        Y_pred = self.predict(X)
        return mse(Y, Y_pred, axis=(0,1))
    
    def predict(self, X):
        return self.forward(X, inference=True)[-1]
    
    def forward(self, X, inference=True):
        layer_input = X.copy()
        layer_inputs = [] if inference else [layer_input]
        for layer, activation, batch_normalization in zip(self.layers, self.activations, self.batch_normalization_layers):
            activation_input = layer.forward(layer_input, inference=inference)
            batch_normalization_input = activation.forward(activation_input)
            layer_input = batch_normalization.forward(batch_normalization_input, inference=inference) if batch_normalization is not None else batch_normalization_input
            if not inference:
                layer_inputs += [activation_input, batch_normalization_input, layer_input]
        if inference:
            layer_inputs.append(layer_input)
        return layer_inputs
    
    def fit(self, X, Y, X_val=None, Y_val=None, lr=0.1, epochs=100, batch_size=100, gradient_descent_method="AdaGrad", gradient_decay=0.8, verbose=True):
        
        assert gradient_descent_method in self.GRADIENT_DESCENT_METHODS
        
        # Batch creation
        if batch_size >= X.shape[0]:
            X_batches = [X]
            Y_batches = [Y]
        else:
            X_batches = [X[i*batch_size:min(X.shape[0],(i+1)*batch_size)] for i in range(X.shape[0] // batch_size)]
            Y_batches = [Y[i*batch_size:min(X.shape[0],(i+1)*batch_size)] for i in range(X.shape[0] // batch_size)]
        
        self.history = []
        past_layer_gradient_norms = [{} for layer in self.layers]
        past_batch_normalization_gradient_norms = [{} for layer in self.layers]
        for e in range(epochs):
            # Fitting
            for X_batch, Y_batch in zip(X_batches, Y_batches):
                # Forward pass
                layer_inputs = self.forward(X_batch, inference=False)

                # Back propagation
                Y_pred = layer_inputs.pop(-1)
                layer_error = (Y_batch - Y_pred) / Y_batch.shape[0] / Y_batch.shape[1]
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
                if gradient_descent_method == 'AdaGrad':
                    past_layer_gradient_norms = [
                        {k: (pgn[k] if k in pgn else 0.0)+np.square(g[k]) for k in g} 
                        for pgn, g in zip(past_layer_gradient_norms, layer_gradients)]
                elif gradient_descent_method == 'RMSProp':
                    past_layer_gradient_norms = [
                        {k: gradient_decay*(pgn[k] if k in pgn else 0.0)+(1-gradient_decay)*np.square(g[k]) for k in g} 
                        for pgn, g in zip(past_layer_gradient_norms, layer_gradients)]
                for layer, layer_gradient, past_layer_gradient_norm in zip(self.layers, layer_gradients, past_layer_gradient_norms):
                    if gradient_descent_method in ('AdaGrad', 'RMSProp'):
                        kwargs = {k: lr * layer_gradient[k] / np.sqrt(past_layer_gradient_norm[k]+1e-6) for k in layer_gradient}
                        layer.update(**kwargs)
                    else:
                        kwargs = {k: lr * layer_gradient[k] for k in layer_gradient}
                        layer.update(**kwargs)

                # Batch normalization updates
                if gradient_descent_method == 'AdaGrad':
                    past_batch_normalization_gradient_norms = [
                        {k: (pgn[k] if k in pgn else 0.0)+np.square(g[k]) for k in g} 
                        for pgn, g in zip(past_batch_normalization_gradient_norms, batch_normalization_gradients)]
                elif gradient_descent_method == 'RMSProp':
                    past_batch_normalization_gradient_norms = [
                        {k: gradient_decay*(pgn[k] if k in pgn else 0.0)+(1-gradient_decay)*np.square(g[k]) for k in g} 
                        for pgn, g in zip(past_batch_normalization_gradient_norms, batch_normalization_gradients)]
                for layer, layer_gradient, past_layer_gradient_norm in zip(self.batch_normalization_layers, batch_normalization_gradients, past_batch_normalization_gradient_norms):
                    if layer is not None:
                        if gradient_descent_method in ('AdaGrad', 'RMSProp'):
                            kwargs = {k: lr * layer_gradient[k] / np.sqrt(past_layer_gradient_norm[k]+1e-6) for k in layer_gradient}
                            layer.update(**kwargs)
                        else:
                            kwargs = {k: lr * layer_gradient[k] for k in layer_gradient}
                            layer.update(**kwargs)
                     
            # Evaluation
            Y_pred = self.predict(X)
            metrics = {
                'train_mse': mse(Y, Y_pred, axis=(0,1))
            }
            if X_val is not None and Y_val is not None:
                Y_pred_val = self.predict(X_val)
                metrics['val_mse'] = mse(Y_val, Y_pred_val, axis=(0,1))
            self.history.append(metrics)
            
            if verbose and (e+1) % 10 == 0:
                print(f"Epoch {e+1}/{epochs} - train loss = {metrics['train_mse'].mean():.4f}" + (f", validation loss = {metrics['val_mse'].mean():.4f}" if 'val_mse' in metrics else ""))
        
        return self.history
