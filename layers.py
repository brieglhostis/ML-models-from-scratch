
import numpy as np

from utils import sigmoid, tanh, softmax, dummy_encode, one_hot_encode
from normalization import LayerNormalization


class AddLayer:
    
    def __init__(self):
        return
    
    def forward(self, X):
        return np.sum(X, axis=0)
    
    def backward(self, X, Y_error):
        return np.repeat(np.expand_dims(Y_error, axis=0), len(X), axis=0)


class ConcatLayer:
    
    def __init__(self):
        return
    
    def forward(self, X):
        return np.concatenate(X, axis=-1)
    
    def backward(self, X, Y_error):
        return np.split(Y_error, 2, axis=-1)


class DenseLayer:
    
    def __init__(self, F, D, add_bias=True, dropout_p=0.0):
        self.add_bias = add_bias
        self.dropout_p = dropout_p
        self.F = F+1 if self.add_bias else F
        self.D = D
        self.W = np.random.normal(loc=0, scale=1/self.F, size=(self.F, self.D))
    
    def forward(self, X, inference=True):
        assert len(X.shape) in [2, 3]
        if self.add_bias:
            if len(X.shape) == 2:
                X = np.c_[X, np.ones(X.shape[0])]
            else: 
                X = np.array([np.c_[x, np.ones(X.shape[1])] for x in X])
        if not inference and self.dropout_p > 0.0 and self.dropout_p < 1.0:
            return X @ (self.W * np.where(np.random.random(size=self.W.shape)>self.dropout_p, 1.0, 0.0) / (1-self.dropout_p))
        return X @ self.W
    
    def backward(self, X, Y_error, l2=0.0):
        assert len(X.shape) in [2, 3]
        if self.add_bias:
            if len(X.shape) == 2:
                X = np.c_[X, np.ones(X.shape[0])]
            else: 
                X = np.array([np.c_[x, np.ones(X.shape[1])] for x in X])
        if len(X.shape) == 2:
            W_gradient = X.T @ Y_error
        else:
            W_gradient = np.sum(np.transpose(X, axes= (0,2,1)) @ Y_error, axis=0)
        W_gradient = W_gradient + 2*l2 * self.W / self.F / self.D
        error = Y_error @ (self.W[:-1] if self.add_bias else self.W).T
        return {'W_gradient': W_gradient}, error
    
    def update(self, W_gradient):
        self.W -= W_gradient


class RecurrentLayer:
    
    def __init__(self, F, D, add_bias=True, dropout_p=0.0):
        self.add_bias = add_bias
        self.dropout_p = dropout_p
        self.F = F+1 if self.add_bias else F
        self.D = D
        self.W = np.random.normal(loc=0, scale=1/self.F, size=(self.F, self.D)) # current input weights
        self.V = np.random.normal(loc=0, scale=1/self.D, size=(self.D, self.D)) # previous output weights
        
    def forward(self, X, inference=True):
        if self.add_bias:
            X = np.array([np.c_[x, np.ones(X.shape[1])] for x in X])
        if not inference and self.dropout_p > 0.0 and self.dropout_p < 1.0:
            W = self.W * np.where(np.random.random(size=self.W.shape)>self.dropout_p, 1.0, 0.0) / (1-self.dropout_p)
            V = self.V * np.where(np.random.random(size=self.V.shape)>self.dropout_p, 1.0, 0.0) / (1-self.dropout_p)
        else:
            W, V = self.W, self.V
        H = np.zeros((X.shape[0], X.shape[1], self.D))
        for t in range(X.shape[1]):
            H[:,t,:] = X[:,t,:] @ W + H[:,t-1,:] @ V if t > 0 else X[:,t,:] @ W
        return H
    
    def backward(self, X, Y_error, l2=0.0):
        if self.add_bias:
            X = np.array([np.c_[x, np.ones(X.shape[1])] for x in X])
        
        # Forward pass
        H = np.zeros((X.shape[0], X.shape[1], self.D))
        for t in range(X.shape[1]):
            H[:,t,:] = X[:,t,:] @ self.W + H[:,t-1,:] @ self.V if t > 0 else X[:,t,:] @ self.W
        
        # Back Propagation Through Time (BPTT)
        non_bias_W = self.W[:-1] if self.add_bias else self.W
        W_gradient = np.zeros((self.F, self.D)) # W gradient
        V_gradient = np.zeros((self.D, self.D)) # V gradient
        X_error = np.zeros((X.shape[0], X.shape[1], non_bias_W.shape[0])) # Propagated error
        for t in range(X.shape[1]-1,0,-1):
            W_gradient += X[:,t,:].T @ Y_error[:,t,:]
            X_error[:,t,:] = Y_error[:,t,:] @ non_bias_W.T
            if t > 0:
                V_gradient += H[:,t-1,:].T @ Y_error[:,t,:]
                Y_error[:,t-1,:] = Y_error[:,t,:] @ self.V.T
                
        W_gradient = W_gradient + 2*l2 * self.W / self.F / self.D
        V_gradient = V_gradient + 2*l2 * self.V / self.D / self.D
        return {'W_gradient': W_gradient, 'V_gradient': V_gradient}, X_error
    
    def update(self, W_gradient, V_gradient):
        self.W -= W_gradient
        self.V -= V_gradient

        
class LSTMLayer:
    
    def __init__(self, F, D, add_bias=True, dropout_p=0.0):
        self.add_bias = add_bias
        self.dropout_p = dropout_p
        self.F = F+1 if self.add_bias else F
        self.D = D
        self.W_forget = np.random.normal(loc=0, scale=1/self.F, size=(self.F, self.D))
        self.W_input = np.random.normal(loc=0, scale=1/self.F, size=(self.F, self.D))
        self.W_output = np.random.normal(loc=0, scale=1/self.F, size=(self.F, self.D))
        self.W_concat = np.random.normal(loc=0, scale=1/self.F, size=(self.F, self.D))
        self.U_forget = np.random.normal(loc=0, scale=1/self.D, size=(self.D, self.D))
        self.U_input = np.random.normal(loc=0, scale=1/self.D, size=(self.D, self.D))
        self.U_output = np.random.normal(loc=0, scale=1/self.D, size=(self.D, self.D))
        self.U_concat = np.random.normal(loc=0, scale=1/self.D, size=(self.D, self.D))
        self.c = 0.0
        self.h = 0.0

    def forward(self, X, inference=True):
        if self.add_bias:
            X = np.array([np.c_[x, np.ones(X.shape[1])] for x in X])
        W = {'f': self.W_forget, 'i': self.W_input, 'o': self.W_output, 'c': self.W_concat}
        U = {'f': self.U_forget, 'i': self.U_input, 'o': self.U_output, 'c': self.U_concat}
        if not inference and self.dropout_p > 0.0 and self.dropout_p < 1.0:
            for k in ['f', 'i', 'o', 'c']:
                W[k] = W[k] * np.where(np.random.random(size=W[k].shape)>self.dropout_p, 1.0, 0.0) / (1-self.dropout_p)
                U[k] = U[k] * np.where(np.random.random(size=U[k].shape)>self.dropout_p, 1.0, 0.0) / (1-self.dropout_p)
        tmp = {k: np.zeros((X.shape[0], self.D)) for k in ['f', 'i', 'o', 'c']}
        C_t = np.zeros((X.shape[0], self.D))
        C_t_minus_1 = np.zeros((X.shape[0], self.D))
        H = np.zeros((X.shape[0], X.shape[1], self.D))
        for t in range(X.shape[1]):
            for k in ['f', 'i', 'o', 'c']:
                activation_function = tanh if k == 'c' else sigmoid
                tmp[k] = activation_function(X[:,t,:] @ W[k] + H[:,t-1,:] @ U[k] if t > 0 else X[:,t,:] @ W[k])
            C_t = tmp['i'] * tmp['c'] + tmp['f'] * C_t_minus_1 if t > 0 else tmp['i'] * tmp['c']
            C_t_minus_1 = C_t
            H[:,t,:] = tmp['o'] * tanh(C_t)
        return H
    
    def backward(self, X, Y_error, l2=0.0):
        if self.add_bias:
            X = np.array([np.c_[x, np.ones(X.shape[1])] for x in X])
            
        # Forward pass
        W = {'f': self.W_forget, 'i': self.W_input, 'o': self.W_output, 'c': self.W_concat}
        U = {'f': self.U_forget, 'i': self.U_input, 'o': self.U_output, 'c': self.U_concat}
        tmp = {k: np.zeros((X.shape[0], self.D)) for k in ['f', 'i', 'o', 'c']}
        C = np.zeros((X.shape[0], X.shape[1], self.D))
        H = np.zeros((X.shape[0], X.shape[1], self.D))
        for t in range(X.shape[1]):
            for k in ['f', 'i', 'o', 'c']:
                activation_function = tanh if k == 'c' else sigmoid
                tmp[k] = activation_function(X[:,t,:] @ W[k] + H[:,t-1,:] @ U[k] if t > 0 else X[:,t,:] @ W[k])
            C[:,t,:] = tmp['i'] * tmp['c'] + tmp['f'] * C[:,t-1,:] if t > 0 else tmp['i'] * tmp['c']
            H[:,t,:] = tmp['o'] * tanh(C[:,t,:])
            
        # Back Propagation Through Time (BPTT)
        non_bias_W = {k: w[:-1] if self.add_bias else w for k, w in W.items()}
        non_bias_F = self.F-1 if self.add_bias else self.F
        W_gradient = {k: np.zeros((self.F, self.D)) for k in ['f', 'i', 'o', 'c']} # W gradient
        U_gradient = {k: np.zeros((self.D, self.D)) for k in ['f', 'i', 'o', 'c']} # W gradient
        C_error = np.zeros((X.shape[0], X.shape[1], self.D)) # Propagated error on C
        X_error = np.zeros((X.shape[0], X.shape[1], non_bias_F)) # Propagated error on X
        for t in range(X.shape[1]-1,0,-1):
            for k in ['f', 'i', 'o', 'c']:
                activation_function = tanh if k == 'c' else sigmoid
                tmp[k] = activation_function(X[:,t,:] @ W[k] + H[:,t-1,:] @ U[k] if t > 0 else X[:,t,:] @ W[k])
            
            # Output gate
            tmp_error = Y_error[:,t,:] * tmp['o'] * (1 - tmp['o']) * tanh(C[:,t,:])
            W_gradient['o'] += X[:,t,:].T @ tmp_error
            X_error[:,t,:] += tmp_error @ non_bias_W['o'].T
            if t > 0:
                U_gradient['o'] += H[:,t-1,:].T @ tmp_error
                Y_error[:,t-1,:] += tmp_error @ U['o'].T
            C_error[:,t,:] += Y_error[:,t,:] * tmp['o'] * (1 - np.square(tanh(C[:,t,:])))
                
            # Concat gate
            tmp_error = C_error[:,t,:] * tmp['i'] * (1 - np.square(tmp['c']))
            W_gradient['c'] += X[:,t,:].T @ tmp_error
            X_error[:,t,:] += tmp_error @ non_bias_W['c'].T
            if t > 0:
                U_gradient['c'] += H[:,t-1,:].T @ tmp_error
                Y_error[:,t-1,:] += tmp_error @ U['c'].T
                
            # Input gate
            tmp_error = C_error[:,t,:] * tmp['i'] * (1 - tmp['i']) * tmp['c']
            W_gradient['i'] += X[:,t,:].T @ tmp_error
            X_error[:,t,:] += tmp_error @ non_bias_W['i'].T
            if t > 0:
                U_gradient['i'] += H[:,t-1,:].T @ tmp_error
                Y_error[:,t-1,:] += tmp_error @ U['i'].T
                
            # Forget gate
            tmp_error = C_error[:,t,:] * tmp['f'] * (1 - tmp['f']) * C[:,t-1,:]
            W_gradient['f'] += X[:,t,:].T @ tmp_error
            X_error[:,t,:] += tmp_error @ non_bias_W['f'].T
            if t > 0:
                U_gradient['f'] += H[:,t-1,:].T @ tmp_error
                Y_error[:,t-1,:] += tmp_error @ U['f'].T
                C_error[:,t-1,:] = C_error[:,t,:] * tmp['f']
                
        gradients = {
            'W_forget_gradient': W_gradient['f'] + 2*l2 * W['f'] / self.F / self.D
            , 'W_input_gradient': W_gradient['i'] + 2*l2 * W['i'] / self.F / self.D
            , 'W_output_gradient': W_gradient['o'] + 2*l2 * W['o'] / self.F / self.D
            , 'W_concat_gradient': W_gradient['c'] + 2*l2 * W['c'] / self.F / self.D
            , 'U_forget_gradient': U_gradient['f'] + 2*l2 * U['f'] / self.D / self.D
            , 'U_input_gradient': U_gradient['i'] + 2*l2 * U['i'] / self.D / self.D
            , 'U_output_gradient': U_gradient['o'] + 2*l2 * U['o'] / self.D / self.D
            , 'U_concat_gradient': U_gradient['c'] + 2*l2 * U['c'] / self.D / self.D
        }
        return gradients, X_error
    
    def update(self, W_forget_gradient, W_input_gradient, W_output_gradient, W_concat_gradient, U_forget_gradient, U_input_gradient, U_output_gradient, U_concat_gradient):
        self.W_forget -= W_forget_gradient
        self.W_input -= W_input_gradient
        self.W_output -= W_output_gradient
        self.W_concat -= W_concat_gradient
        self.U_forget -= U_forget_gradient
        self.U_input -= U_input_gradient
        self.U_output -= U_output_gradient
        self.U_concat -= U_concat_gradient


class BidirectionalRecurrentLayer:
    
    def __init__(self, F, D, add_bias=True, dropout_p=0.0, concat=True):
        if concat:
            if D % 2 != 0:
                raise ValueError(f"Layer output dimension must be a multiple of 2 for concat aggregation (got {D})")
            D = D // 2
            self.agg_layer = ConcatLayer()
        else:
            self.agg_layer = AddLayer()
        self.forward_layer = RecurrentLayer(F, D, add_bias=add_bias, dropout_p=dropout_p)
        self.backward_layer = RecurrentLayer(F, D, add_bias=add_bias, dropout_p=dropout_p)
        
    def forward(self, X, inference=True):
        return self.agg_layer.forward([
            self.forward_layer.forward(X, inference=inference)
            , self.backward_layer.forward(X[:,::-1,:], inference=inference)[:,::-1,:]
        ])
    
    def backward(self, X, Y_error, l2=0.0):
        Y_forward = self.forward_layer.forward(X, inference=False)
        Y_backward = self.backward_layer.forward(X[:,::-1,:], inference=False)[:,::-1,:]
        Y_error_forward, Y_error_backward = self.agg_layer.backward(X, Y_error)
        gradients_forward, X_error_forward = self.forward_layer.backward(X, Y_error_forward, l2=l2)
        gradients_backward, X_error_backward = self.backward_layer.backward(X[:,::-1,:], Y_error_backward[:,::-1,:], l2=l2)
        gradients = {
            'W_gradient_forward': gradients_forward['W_gradient']
            , 'V_gradient_forward': gradients_forward['V_gradient']
            , 'W_gradient_backward': gradients_backward['W_gradient']
            , 'V_gradient_backward': gradients_backward['V_gradient']
        }
        return gradients, (X_error_forward + X_error_backward[:,::-1,:])
    
    def update(self, W_gradient_forward, V_gradient_forward, W_gradient_backward, V_gradient_backward):
        self.forward_layer.update(W_gradient_forward, V_gradient_forward)
        self.backward_layer.update(W_gradient_backward, V_gradient_backward)
    

class BidirectionalLSTMLayer:
    
    def __init__(self, F, D, add_bias=True, dropout_p=0.0, concat=True):
        if concat:
            if D % 2 != 0:
                raise ValueError(f"Layer output dimension must be a multiple of 2 for concat aggregation (got {D})")
            D = D // 2
            self.agg_layer = ConcatLayer()
        else:
            self.agg_layer = AddLayer()
        self.forward_layer = LSTMLayer(F, D, add_bias=add_bias, dropout_p=dropout_p)
        self.backward_layer = LSTMLayer(F, D, add_bias=add_bias, dropout_p=dropout_p)
        
    def forward(self, X, inference=True):
        return self.agg_layer.forward([
            self.forward_layer.forward(X, inference=inference)
            , self.backward_layer.forward(X[:,::-1,:], inference=inference)[:,::-1,:]
        ])
    
    def backward(self, X, Y_error, l2=0.0):
        Y_forward = self.forward_layer.forward(X, inference=False)
        Y_backward = self.backward_layer.forward(X[:,::-1,:], inference=False)[:,::-1,:]
        Y_error_forward, Y_error_backward = self.agg_layer.backward(X, Y_error)
        gradients_forward, X_error_forward = self.forward_layer.backward(X, Y_error_forward, l2=l2)
        gradients_backward, X_error_backward = self.backward_layer.backward(X[:,::-1,:], Y_error_backward[:,::-1,:], l2=l2)
        gradients = {
            'W_forget_gradient_forward': gradients_forward['W_forget_gradient']
            , 'W_input_gradient_forward': gradients_forward['W_input_gradient']
            , 'W_output_gradient_forward': gradients_forward['W_output_gradient']
            , 'W_concat_gradient_forward': gradients_forward['W_concat_gradient']
            , 'U_forget_gradient_forward': gradients_forward['U_forget_gradient']
            , 'U_input_gradient_forward': gradients_forward['U_input_gradient']
            , 'U_output_gradient_forward': gradients_forward['U_output_gradient']
            , 'U_concat_gradient_forward': gradients_forward['U_concat_gradient']
            , 'W_forget_gradient_backward': gradients_backward['W_forget_gradient']
            , 'W_input_gradient_backward': gradients_backward['W_input_gradient']
            , 'W_output_gradient_backward': gradients_backward['W_output_gradient']
            , 'W_concat_gradient_backward': gradients_backward['W_concat_gradient']
            , 'U_forget_gradient_backward': gradients_backward['U_forget_gradient']
            , 'U_input_gradient_backward': gradients_backward['U_input_gradient']
            , 'U_output_gradient_backward': gradients_backward['U_output_gradient']
            , 'U_concat_gradient_backward': gradients_backward['U_concat_gradient']
        }
        return gradients, (X_error_forward + X_error_backward[:,::-1,:])
    
    def update(self
               , W_forget_gradient_forward, W_input_gradient_forward, W_output_gradient_forward, W_concat_gradient_forward
               , U_forget_gradient_forward, U_input_gradient_forward, U_output_gradient_forward, U_concat_gradient_forward
               , W_forget_gradient_backward, W_input_gradient_backward, W_output_gradient_backward, W_concat_gradient_backward
               , U_forget_gradient_backward, U_input_gradient_backward, U_output_gradient_backward, U_concat_gradient_backward):
        self.forward_layer.update(
            W_forget_gradient_forward, W_input_gradient_forward, W_output_gradient_forward, W_concat_gradient_forward
            , U_forget_gradient_forward, U_input_gradient_forward, U_output_gradient_forward, U_concat_gradient_forward)
        self.backward_layer.update(
            W_forget_gradient_backward, W_input_gradient_backward, W_output_gradient_backward, W_concat_gradient_backward
            , U_forget_gradient_backward, U_input_gradient_backward, U_output_gradient_backward, U_concat_gradient_backward)


class EmbeddingLayer:
    
    def __init__(self, F, D, dropout_p=0.0, mask_zero=True):
        self.dropout_p = dropout_p
        self.mask_zero = mask_zero
        self.F = F-1 if self.mask_zero else F
        self.D = D
        self.W = np.random.normal(loc=0, scale=1/self.F, size=(self.F, self.D))
        
    def forward(self, X, inference=True):
        if self.mask_zero:
            X = X-1
        X = dummy_encode(X, self.F)
        if not inference and self.dropout_p > 0.0 and self.dropout_p < 1.0:
            return X @ (self.W * np.where(np.random.random(size=self.W.shape)>self.dropout_p, 1.0, 0.0) / (1-self.dropout_p))
        return X @ self.W
    
    def backward(self, X, Y_error, l2=0.0):
        if self.mask_zero:
            X = X-1
        X = dummy_encode(X, self.F)
        W_gradient = np.sum(np.transpose(X, axes=(1,2,0)) @ np.transpose(Y_error, axes=(1,0,2)), axis=0) + 2*l2 * self.W / self.F / self.D
        return {'W_gradient': W_gradient}, None
    
    def update(self, W_gradient):
        self.W -= W_gradient


class PositionalEmbeddingLayer(EmbeddingLayer):
    
    def __init__(self, F, D, dropout_p=0.0, mask_zero=True):
        if D % 2 == 1:
            raise ValueError("Output dimension of positional encoding needs to be a multiple of 2")
        super().__init__(F, D, dropout_p=dropout_p, mask_zero=mask_zero)
        positions = np.arange(2048)
        self.positional_encoding = np.array([
            np.sin(positions / np.power(10000, 2*i/self.D)) if j == 0 else 
            np.cos(positions / np.power(10000, 2*i/self.D))
            for i in range(self.D//2) for j in range(2)
        ]).T.astype(np.float32)
        
    def forward(self, X, inference=True):
        if self.mask_zero:
            X = X-1
        X = dummy_encode(X, self.F)
        W = self.W
        if not inference and self.dropout_p > 0.0 and self.dropout_p < 1.0:
            W *= np.where(np.random.random(size=W.shape)>self.dropout_p, 1.0, 0.0) / (1-self.dropout_p)
        mask = np.where(np.sum(np.square(X), axis=2) > 0, 1, 0) # (N x T)
        positional_encoding = self.positional_encoding[:X.shape[1],:] # (T x D)
        return X @ W * np.sqrt(self.D) + np.expand_dims(mask, axis=-1) * np.expand_dims(positional_encoding, axis=0)
    
    def backward(self, X, Y_error, l2=0.0):
        if self.mask_zero:
            X = X-1
        X = dummy_encode(X, self.F)
        W_gradient = np.sum(np.transpose(X, axes=(1,2,0)) @ np.transpose(Y_error, axes=(1,0,2)), axis=0) + 2*l2 * self.W / self.F / self.D
        return {'W_gradient': W_gradient * np.sqrt(self.D)}, None


class MultiHeadAttentionLayer:
    
    def __init__(self, F, NH=10, add_bias=True, dropout_p=0.0, use_causal_mask=False):
        if F % NH != 0:
            raise ValueError(f"The input size must be a multiple of the number of heads ({NH} here)")
        self.add_bias = add_bias
        self.dropout_p = dropout_p
        self.use_causal_mask = use_causal_mask
        self.F = F
        self.NH = NH
        self.W_q = np.random.normal(loc=0, scale=1/self.F, size=(self.F, self.F)) # query input weights
        self.W_k = np.random.normal(loc=0, scale=1/self.F, size=(self.F, self.F)) # key input weights
        self.W_v = np.random.normal(loc=0, scale=1/self.F, size=(self.F, self.F)) # value input weights
        
    def split(self, X):
        query_size = self.F // self.NH
        X = np.reshape(X, (X.shape[0], X.shape[1], self.NH, query_size)) # (N x T x NH x F/NH)
        return np.transpose(X, axes=(0,2,1,3)) # (N x NH x T x F/NH)
    
    def merge(self, X):
        X = np.transpose(X, axes=(0,2,1,3)) # (N x T x NH x F/NH)
        return np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]*X.shape[3])) # (N x T x F)
    
    def forward(self, X_q, X_k=None, X_v=None, inference=True):

        def softmax_with_mask(X, mask, axis=-1, safety_threshold=200):
            exp_safety_threshold = np.exp(safety_threshold)
            exp_X = np.where(X>safety_threshold, exp_safety_threshold, np.exp(X))
            return exp_X * mask / (np.sum(exp_X * mask, axis=axis, keepdims=True) + 1e-6)

        if self.use_causal_mask:
            mask = np.repeat(np.expand_dims(np.tril(np.ones((X_q.shape[1], X_q.shape[1]))), axis=0), X_q.shape[0], axis=0) # (N x T x T)
        else:
            mask = np.where(np.sum(np.square(X_q), axis=2) > 0, 1, 0) # (N x T)
            mask = np.array([m @ m.T for m in np.expand_dims(mask, axis=-1)]) # (N x T x T)
        mask = np.expand_dims(mask, axis=1)
        X_k = X_q.copy() if X_k is None else X_k
        X_v = X_q.copy() if X_v is None else X_v
        X = {'Q': X_q, 'K': X_k, 'V': X_v} # (N x T x F)
        W = {'Q': self.W_q, 'K': self.W_k, 'V': self.W_v} # (F x F)
        if not inference and self.dropout_p > 0.0 and self.dropout_p < 1.0:
            for k in ['Q', 'K', 'V']:
                W[k] *= np.where(np.random.random(size=W[k].shape)>self.dropout_p, 1.0, 0.0) / (1-self.dropout_p)
        # Linear transformation
        H = {k: X[k]@W[k] for k in ['Q', 'K', 'V']} # (N x T x F)
        # Data splitting accross heads
        H = {k: self.split(H[k]) for k in H} # (N x NH x T x F/NH)
        # Attention score
        S = H['Q']@np.transpose(H['K'], axes=(0,1,3,2)) # (N x NH x T x T)
        S = softmax_with_mask(S / np.sqrt(self.F // self.NH), mask, axis=(3)) # (N x NH x T x T)
        S = S@H['V'] # (N x NH x T x F/NH)
        # Attention score merge
        S = self.merge(S) # (N x T x F)
        return S
        
    def backward(self, Y_error, X_q, X_k=None, X_v=None, l2=0.0):

        def softmax_with_mask(X, mask, axis=-1, safety_threshold=200):
            exp_safety_threshold = np.exp(safety_threshold)
            exp_X = np.where(X>safety_threshold, exp_safety_threshold, np.exp(X))
            return exp_X * mask / (np.sum(exp_X * mask, axis=axis, keepdims=True) + 1e-6)

        if self.use_causal_mask:
            mask = np.repeat(np.expand_dims(np.tril(np.ones((X_q.shape[1], X_q.shape[1]))), axis=0), X_q.shape[0], axis=0) # (N x T x T)
        else:
            mask = np.where(np.sum(np.square(X_q), axis=2) > 0, 1, 0) # (N x T)
            mask = np.array([m @ m.T for m in np.expand_dims(mask, axis=-1)]) # (N x T x T)
        mask = np.expand_dims(mask, axis=1)
        X_k = X_q.copy() if X_k is None else X_k
        X_v = X_q.copy() if X_v is None else X_v
        X = {'Q': X_q, 'K': X_k, 'V': X_v} # (N x T x F)
        W = {'Q': self.W_q, 'K': self.W_k, 'V': self.W_v} # (F x F)
        # Error splitting accross heads
        Y_error = self.split(Y_error) # (N x NH x T x F/NH)
        # Linear transformation
        H = {k: X[k]@W[k] for k in ['Q', 'K', 'V']} # (N x T x F)
        # Data splitting accross heads
        H = {k: self.split(H[k]) for k in H} # (N x NH x T x F/NH)
        # Attention score (until softmax)
        query_size = self.F // self.NH
        S = H['Q']@np.transpose(H['K'], axes=(0,1,3,2)) # (N x NH x T x T)
        S = softmax_with_mask(S / np.sqrt(query_size), mask, axis=(3)) # (N x NH x T x T)
        # Compute W_q gradient
        W_q_gradient = (S*(1-S)) * (Y_error @ np.transpose(H['V'], axes=(0,1,3,2))) # (N x NH x T x T)
        W_k_gradient = W_q_gradient.copy() # Same starting calculation
        W_q_gradient = W_q_gradient @ H['K'] # (N x NH x T x F/NH)
        W_q_gradient = np.transpose(np.repeat(np.expand_dims(X['Q'], axis=1), self.NH, axis=1), axes=(0,1,3,2)) @ W_q_gradient # (N x NH x F x F/NH)
        W_q_gradient = self.merge(W_q_gradient) # (N x F x F)
        W_q_gradient = np.sum(W_q_gradient, axis=0) / np.sqrt(query_size) # (F x F)
        # Compute X_q error to propagate
        X_q_error = (S*(1-S)) * (Y_error @ np.transpose(H['V'], axes=(0,1,3,2))) # (N x NH x T x T)
        X_k_error = np.transpose(X_q_error, axes=(0,1,3,2)) # Same starting calculation but transposed
        X_q_error = X_q_error @ H['K'] @ np.transpose(self.split(np.repeat(np.expand_dims(W['Q'], axis=0), Y_error.shape[0], axis=0)), axes=(0,1,3,2)) # (N x NH x T x F)
        X_q_error = np.sum(X_q_error, axis=1) / np.sqrt(query_size) # (N x T x F)
        # Compute W_k gradient
        W_k_gradient = np.transpose(W_k_gradient, axes=(0,1,3,2)) @ H['Q'] # (N x NH x T x F/NH)
        W_k_gradient = np.transpose(np.repeat(np.expand_dims(X['K'], axis=1), self.NH, axis=1), axes=(0,1,3,2)) @ W_k_gradient # (N x NH x F x F/NH)
        W_k_gradient = self.merge(W_k_gradient) # (N x F x F)
        W_k_gradient = np.sum(W_k_gradient, axis=0) / np.sqrt(query_size) # (F x F)
        # Compute X_k error to propagate
        X_k_error = X_k_error @ H['Q'] @ np.transpose(self.split(np.repeat(np.expand_dims(W['K'], axis=0), Y_error.shape[0], axis=0)), axes=(0,1,3,2)) # (N x NH x T x F)
        X_k_error = np.sum(X_k_error, axis=1) / np.sqrt(query_size) # (N x T x F)
        # Compute W_v gradient
        W_v_gradient = np.transpose(S, axes=(0,1,3,2)) @ Y_error # (N x NH x T x F/NH)
        W_v_gradient = np.transpose(np.repeat(np.expand_dims(X['V'], axis=1), self.NH, axis=1), axes=(0,1,3,2)) @ W_v_gradient # (N x NH x F x F/NH)
        W_v_gradient = self.merge(W_v_gradient) # (N x F x F)
        W_v_gradient = np.sum(W_v_gradient, axis=0) # (F x F)
        # Compute X_v error to propagate
        X_v_error = Y_error @ np.transpose(self.split(np.repeat(np.expand_dims(W['V'], axis=0), Y_error.shape[0], axis=0)), axes=(0,1,3,2)) # (N x NH x T x F)
        X_v_error = np.transpose(S, axes=(0,1,3,2)) @ X_v_error # (N x NH x T x F)
        X_v_error = np.sum(X_v_error, axis=1) # (N x T x F)
        return {'W_q_gradient': W_q_gradient, 'W_k_gradient': W_k_gradient, 'W_v_gradient': W_v_gradient}, (X_q_error, X_k_error, X_v_error)
    
    def update(self, W_q_gradient, W_k_gradient, W_v_gradient):
        self.W_q -= W_q_gradient
        self.W_k -= W_k_gradient
        self.W_v -= W_v_gradient


class BaseAttentionLayer:
    
    def __init__(self, F, NH=10, add_bias=True, dropout_p=0.0, use_causal_mask=False):
        self.mha_layer = MultiHeadAttentionLayer(F, NH=NH, add_bias=add_bias, dropout_p=dropout_p, use_causal_mask=use_causal_mask)
        self.norm_layer = LayerNormalization()
        self.add_layer = AddLayer()
        
    def forward(self, X_q, X_k=None, X_v=None, X_skip=None, inference=True):
        X = self.mha_layer.forward(X_q, X_k, X_v, inference=inference)
        X = self.add_layer.forward([X, X_q if X_skip is None else X_skip])
        return self.norm_layer.forward(X)
    
    def backward(self, Y_error, X_q, X_k=None, X_v=None, X_skip=None, l2=0.0):
        X_mha = self.mha_layer.forward(X_q, X_k, X_v, inference=False)
        X_add = self.add_layer.forward([X_mha, X_q if X_skip is None else X_skip])
        norm_gradients, Y_error = self.norm_layer.backward(X_add, Y_error, l2=l2)
        Y_error, X_skip_error = self.add_layer.backward([X_mha, X_q if X_skip is None else X_skip], Y_error)
        mha_gradients, (X_q_error, X_k_error, X_v_error) = self.mha_layer.backward(Y_error, X_q, X_k, X_v, l2=l2)
        gradients = {
            'W_q_gradient': mha_gradients['W_q_gradient']
            , 'W_k_gradient': mha_gradients['W_k_gradient']
            , 'W_v_gradient': mha_gradients['W_v_gradient']
            , 'gamma_gradient': norm_gradients['gamma_gradient']
            , 'beta_gradient': norm_gradients['beta_gradient']
        }
        return gradients, (X_q_error, X_k_error, X_v_error, X_skip_error)
    
    def update(self, W_q_gradient, W_k_gradient, W_v_gradient, gamma_gradient, beta_gradient):
        self.mha_layer.update(W_q_gradient, W_k_gradient, W_v_gradient)
        self.norm_layer.update(gamma_gradient, beta_gradient)


class CrossAttentionLayer(BaseAttentionLayer):
    
    def __init__(self, F, NH=10, add_bias=True, dropout_p=0.0, use_causal_mask=False):
        super().__init__(F, NH=NH, add_bias=add_bias, dropout_p=dropout_p, use_causal_mask=use_causal_mask)
        
    def forward(self, X, X_context, inference=True):
        return super().forward(X_q=X, X_k=X_context, X_v=X_context, X_skip=X, inference=inference)
    
    def backward(self, Y_error, X, X_context, l2=0.0):
        gradients, (X_q_error, X_k_error, X_v_error, X_skip_error) = super().backward(
            Y_error, X_q=X, X_k=X_context, X_v=X_context, X_skip=X, l2=l2)
        return gradients, (X_q_error + X_skip_error, X_k_error + X_v_error)


class SelfAttentionLayer(BaseAttentionLayer):
    
    def __init__(self, F, NH=10, add_bias=True, dropout_p=0.0, use_causal_mask=False):
        super().__init__(F, NH=NH, add_bias=add_bias, dropout_p=dropout_p, use_causal_mask=use_causal_mask)
        
    def forward(self, X, inference=True):
        return super().forward(X_q=X, X_k=X, X_v=X, X_skip=X, inference=inference)
    
    def backward(self, Y_error, X, l2=0.0):
        gradients, (X_q_error, X_k_error, X_v_error, X_skip_error) = super().backward(
            Y_error, X_q=X, X_k=X, X_v=X, X_skip=X, l2=l2)
        return gradients, (X_q_error + X_skip_error + X_k_error + X_v_error)
