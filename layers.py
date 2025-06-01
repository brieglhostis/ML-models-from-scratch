
import numpy as np

from normalization import LayerNormalization


def sigmoid(x, safety_threshold=200):
    exp_safety_threshold = np.exp(safety_threshold)
    exp_minus_x = np.where(x<-safety_threshold, exp_safety_threshold, np.exp(-x))
    return 1 / (1+exp_minus_x)

def tanh(x, safety_threshold=200):
    exp_safety_threshold = np.exp(safety_threshold)
    exp_x = np.where(x>safety_threshold, exp_safety_threshold, np.exp(x))
    exp_minus_x = np.where(x<-safety_threshold, exp_safety_threshold, np.exp(-x))
    return (exp_x-exp_minus_x) / (exp_x+exp_minus_x)

def softmax(X, safety_threshold=200, axis=-1):
    exp_safety_threshold = np.exp(safety_threshold)
    exp_X = np.where(X>safety_threshold, exp_safety_threshold, np.exp(X))
    return exp_X / np.sum(exp_X, axis=axis, keepdims=True)

def dummy_encode(X, F=None):
    if len(X.shape) == 1:
        F = np.max(X) if F is None else F
        return np.array([np.where(np.arange(F) == x, 1, 0) for x in X])
    return np.array([dummy_encode(x, F) for x in X])

def one_hot_encode(X, F=None):
    if len(X.shape) == 1:
        F = np.max(X) if F is None else F
        return np.array([np.where(np.arange(1, F) == x, 1, 0) for x in X])
    return np.array([one_hot_encode(x, F) for x in X])


class DenseLayer:
    
    def __init__(self, F, D, add_bias=True, dropout_p=0.0):
        self.add_bias = add_bias
        self.dropout_p = dropout_p
        self.F = F+1 if self.add_bias else F
        self.D = D
        self.W = np.random.normal(loc=0, scale=1/self.F, size=(self.F, self.D))
    
    def forward(self, X, inference=True):
        if self.add_bias:
            X = np.c_[X, np.ones(X.shape[0])]
        if not inference and self.dropout_p > 0.0 and self.dropout_p < 1.0:
            return X@(self.W * np.where(np.random.random(size=self.W.shape)>self.dropout_p, 1.0, 0.0) / (1-self.dropout_p))
        return X@self.W
    
    def backward(self, X, Y_error, l2=0.0):
        if self.add_bias:
            X = np.c_[X, np.ones(X.shape[0])]
        W_gradient = X.T@Y_error + 2*l2 * self.W / self.F / self.D
        error = Y_error@(self.W[:-1] if self.add_bias else self.W).T
        return {'W_gradient': W_gradient}, error
    
    def update(self, W_gradient):
        self.W += W_gradient

        
class RecurrentLayer:
    
    def __init__(self, T, F, D, add_bias=True, dropout_p=0.0):
        self.add_bias = add_bias
        self.dropout_p = dropout_p
        self.T = T
        self.F = F+1 if self.add_bias else F
        self.D = D
        self.W = np.random.normal(loc=0, scale=1/self.F, size=(self.F, self.D)) # current input weights
        self.V = np.random.normal(loc=0, scale=1/self.D, size=(self.D, self.D)) # previous output weights
        
    def forward(self, X, inference=True):
        if self.add_bias:
            X = np.array([np.c_[x, np.ones(self.T)] for x in X])
        if not inference and self.dropout_p > 0.0 and self.dropout_p < 1.0:
            W = self.W * np.where(np.random.random(size=self.W.shape)>self.dropout_p, 1.0, 0.0) / (1-self.dropout_p)
            V = self.V * np.where(np.random.random(size=self.V.shape)>self.dropout_p, 1.0, 0.0) / (1-self.dropout_p)
        else:
            W, V = self.W, self.V
        H = np.zeros((X.shape[0], self.T, self.D))
        for t in range(self.T):
            H[:,t,:] = X[:,t,:]@W + H[:,t-1,:]@V if t > 0 else X[:,t,:]@W
        return H
    
    def backward(self, X, Y_error, l2=0.0):
        if self.add_bias:
            X = np.array([np.c_[x, np.ones(self.T)] for x in X])
            
        N = X.shape[0]
        non_bias_W = self.W[:-1] if self.add_bias else self.W
        non_bias_F = non_bias_W.shape[0]
        
        H = np.zeros((N, self.D)) # Output
        W_gradient = np.zeros((self.F, self.D)) # W gradient
        H_W_gradient = np.zeros((N, self.F, self.D, self.D)) # Derivative of H wrt W
        V_gradient = np.zeros((self.D, self.D)) # V gradient
        H_V_gradient = np.zeros((N, self.D, self.D, self.D)) # Derivative of H wrt V
        error = np.zeros((N, self.T, non_bias_F)) # Propagated error
        H_X_gradient = np.zeros((N, self.T, non_bias_F, self.D)) # Derivative of H wrt X
        for t in range(self.T):
            H_W_gradient = np.expand_dims(X[:,t,:], axis=(-2,-1)) + H_W_gradient @ self.V
            H_V_gradient = np.array([[H[n,d]*np.eye(self.D) for d in range(self.D)] for n in range(N)]) + H_V_gradient @ self.V
            H = X[:,t,:] @ self.W + H @ self.V
            H_X_gradient[:,t,:,:] = np.repeat(np.expand_dims(non_bias_W, axis=0), N, axis=0)
            for s in range(t):
                H_X_gradient[:,s,:,:] = H_X_gradient[:,s,:,:] @ self.V
            W_gradient = W_gradient + np.sum(H_W_gradient*np.expand_dims(Y_error[:,t,:], axis=(1,2)), axis=(0,3))
            V_gradient = V_gradient + np.sum(H_V_gradient*np.expand_dims(Y_error[:,t,:], axis=(1,2)), axis=(0,3))
            error = error + np.sum(H_X_gradient*np.expand_dims(Y_error[:,t,:], axis=(1,2)), axis=-1)
                    
        W_gradient = W_gradient + 2*l2 * self.W / self.F / self.D
        V_gradient = V_gradient + 2*l2 * self.V / self.D / self.D
        return {'W_gradient': W_gradient, 'V_gradient': V_gradient}, error
    
    def update(self, W_gradient, V_gradient):
        self.W += W_gradient
        self.V += V_gradient

        
class LSTMLayer:
    
    def __init__(self, T, F, D, add_bias=True, dropout_p=0.0):
        self.add_bias = add_bias
        self.dropout_p = dropout_p
        self.T = T
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
            X = np.array([np.c_[x, np.ones(self.T)] for x in X])
        N = X.shape[0]
        W = {'f': self.W_forget, 'i': self.W_input, 'o': self.W_output, 'c': self.W_concat}
        U = {'f': self.U_forget, 'i': self.U_input, 'o': self.U_output, 'c': self.U_concat}
        if not inference and self.dropout_p > 0.0 and self.dropout_p < 1.0:
            for k in ['f', 'i', 'o', 'c']:
                W[k] = W[k] * np.where(np.random.random(size=W[k].shape)>self.dropout_p, 1.0, 0.0) / (1-self.dropout_p)
                U[k] = U[k] * np.where(np.random.random(size=U[k].shape)>self.dropout_p, 1.0, 0.0) / (1-self.dropout_p)
        tmp = {k: np.zeros((N, self.D)) for k in ['f', 'i', 'o', 'c']}
        C_t = np.zeros((N, self.D))
        C_t_minus_1 = np.zeros((N, self.D))
        H = np.zeros((N, self.T, self.D))
        for t in range(self.T):
            for k in ['f', 'i', 'o', 'c']:
                activation_function = tanh if k == 'c' else sigmoid
                tmp[k] = activation_function(X[:,t,:]@W[k] + H[:,t-1,:]@U[k] if t > 0 else X[:,t,:]@W[k])
            C_t = tmp['i'] * tmp['c'] + tmp['f'] * C_t_minus_1 if t > 0 else tmp['i'] * tmp['c']
            C_t_minus_1 = C_t
            H[:,t,:] = tmp['o'] * tanh(C_t)
        return H
    
    def backward(self, X, Y_error, l2=0.0):
        if self.add_bias:
            X = np.array([np.c_[x, np.ones(self.T)] for x in X])
            
        N = X.shape[0]
        W = {'f': self.W_forget, 'i': self.W_input, 'o': self.W_output, 'c': self.W_concat}
        U = {'f': self.U_forget, 'i': self.U_input, 'o': self.U_output, 'c': self.U_concat}
        non_bias_W = {k: w[:-1] if self.add_bias else w for k, w in W.items()}
        non_bias_F = self.F-1 if self.add_bias else self.F
        
        tmp = {k: np.zeros((N, self.D)) for k in ['f', 'i', 'o', 'c']}
        C_t = np.zeros((N, self.D)) # Concat at time t
        C_t_minus_1 = np.zeros((N, self.D)) # Concat at time t-1
        H_t = np.zeros((N, self.D)) # Output at time t
        H_t_minus_1 = np.zeros((N, self.D)) # Output at time t-1
        W_gradient = {k: np.zeros((self.F, self.D)) for k in ['f', 'i', 'o', 'c']} # W gradient
        C_W_gradient = {k: np.zeros((N, self.F, self.D, self.D)) for k in ['f', 'i', 'c']} # Derivative of C wrt W
        H_W_gradient = {k: np.zeros((N, self.F, self.D, self.D)) for k in ['f', 'i', 'o', 'c']} # Derivative of H wrt W
        U_gradient = {k: np.zeros((self.D, self.D)) for k in ['f', 'i', 'o', 'c']} # W gradient
        C_U_gradient = {k: np.zeros((N, self.D, self.D, self.D)) for k in ['f', 'i', 'c']} # Derivative of C wrt W
        H_U_gradient = {k: np.zeros((N, self.D, self.D, self.D)) for k in ['f', 'i', 'o', 'c']} # Derivative of H wrt W
        error = np.zeros((N, self.T, non_bias_F)) # Propagated error
        H_X_gradient = {k: np.zeros((N, self.T, non_bias_F, self.D)) for k in ['f', 'i', 'o', 'c', 'C', 'H']} # Derivative of H wrt X
        for t in range(self.T):
            for k in ['f', 'i', 'o', 'c']:
                activation_function = tanh if k == 'c' else sigmoid
                tmp[k] = activation_function(X[:,t,:]@W[k] + H_t@U[k] if t > 0 else X[:,t,:]@W[k])
            C_t = tmp['i'] * tmp['c'] + tmp['f'] * C_t_minus_1
            tanh_c_t = tanh(C_t)
            o_t_tanh_prime_c_t = tmp['o'] * (1-np.square(tanh_c_t))
            H_t = tmp['o'] * tanh_c_t
            
            # Forget gate
            C_W_gradient['f'] =np.expand_dims(tmp['f'], axis=(1,2)) * (
                np.expand_dims((1-tmp['f'])*C_t_minus_1, axis=(1,2)) * (
                    np.expand_dims(X[:,t,:], axis=(2,3)) + 
                    H_W_gradient['f'] @ U['f']) + 
                 C_W_gradient['f'])
            H_W_gradient['f'] = np.expand_dims(o_t_tanh_prime_c_t, axis=(1,2)) * C_W_gradient['f']
            C_U_gradient['f'] = np.expand_dims(tmp['f'], axis=(1,2)) * (
                np.expand_dims((1-tmp['f'])*C_t_minus_1, axis=(1,2)) * (
                    np.array([[H_t_minus_1[n,d]*np.eye(self.D) for d in range(self.D)] for n in range(N)]) + 
                    H_U_gradient['f'] @ U['f']) + 
                C_U_gradient['f'])
            H_U_gradient['f'] = np.expand_dims(o_t_tanh_prime_c_t, axis=(1,2)) * C_U_gradient['f']

            # Input gate
            C_W_gradient['i'] = (
                np.expand_dims(tmp['c']*tmp['i']*(1-tmp['i']), axis=(1,2)) * (
                    np.expand_dims(X[:,t,:], axis=(2,3)) + 
                    H_W_gradient['i'] @ U['i']) + 
                np.expand_dims(tmp['f'], axis=(1,2)) * C_W_gradient['i'])
            H_W_gradient['i'] = np.expand_dims(o_t_tanh_prime_c_t, axis=(1,2)) * C_W_gradient['i']
            C_U_gradient['i'] = (
                np.expand_dims(tmp['c']*tmp['i']*(1-tmp['i']), axis=(1,2)) * (
                    np.array([[H_t_minus_1[n,d]*np.eye(self.D) for d in range(self.D)] for n in range(N)]) + 
                    H_U_gradient['i'] @ U['i']) + 
                np.expand_dims(tmp['f'], axis=(1,2)) * C_U_gradient['i'])
            H_U_gradient['i'] = np.expand_dims(o_t_tanh_prime_c_t, axis=(1,2)) * C_U_gradient['i']

            # Output gate 
            H_W_gradient['o'] = np.expand_dims(tmp['o']*(1-tmp['o']) * tanh_c_t, axis=(1,2)) * (
                np.expand_dims(X[:,t,:], axis=(2,3)) + 
                H_W_gradient['o'] @ U['o'])
            H_U_gradient['o'] = np.expand_dims(tmp['o']*(1-tmp['o']) * tanh_c_t, axis=(1,2)) * (
                np.array([[H_t_minus_1[n,d]*np.eye(self.D) for d in range(self.D)] for n in range(N)]) + 
                H_U_gradient['o'] @ U['o'])

            # Concat gate
            C_W_gradient['c'] = (
                np.expand_dims(tmp['i']*(1-np.square(tmp['c'])), axis=(1,2)) * (
                    np.expand_dims(X[:,t,:], axis=(2,3)) + 
                    H_W_gradient['c'] @ U['c']) + 
                np.expand_dims(tmp['f'], axis=(1,2)) * C_W_gradient['c'])
            H_W_gradient['c'] = np.expand_dims(o_t_tanh_prime_c_t, axis=(1,2)) * C_W_gradient['c']
            C_U_gradient['c'] = (
                np.expand_dims(tmp['i']*(1-np.square(tmp['c'])), axis=(1,2)) * (
                    np.array([[H_t_minus_1[n,d]*np.eye(self.D) for d in range(self.D)] for n in range(N)]) + 
                    H_U_gradient['c'] @ U['c']) + 
                np.expand_dims(tmp['f'], axis=(1,2)) * C_U_gradient['c'])
            H_U_gradient['c'] = np.expand_dims(o_t_tanh_prime_c_t, axis=(1,2)) * C_U_gradient['c']

            # Propagated error
            for k in ['f', 'i', 'o', 'c']:
                activation_derivate = 1-np.square(tmp[k]) if k == 'c' else tmp[k]*(1-tmp[k])
                H_X_gradient[k][:,t,:,:] = np.expand_dims(activation_derivate, axis=1) * np.repeat(np.expand_dims(non_bias_W[k], axis=0), N, axis=0)
                for s in range(t):
                    H_X_gradient[k][:,s,:,:] = np.expand_dims(activation_derivate, axis=1) * (H_X_gradient[k][:,s,:,:] @ U[k])
            H_X_gradient['C'] = (
                H_X_gradient['f'] * np.expand_dims(C_t_minus_1, axis=(1,2)) + 
                H_X_gradient['C'] * np.expand_dims(tmp['f'], axis=(1,2)) + 
                H_X_gradient['i'] * np.expand_dims(tmp['c'], axis=(1,2)) + 
                H_X_gradient['c'] * np.expand_dims(tmp['i'], axis=(1,2)))
            H_X_gradient['H'] = (
                H_X_gradient['o'] * np.expand_dims(tanh_c_t, axis=(1,2)) + 
                H_X_gradient['C'] * np.expand_dims(o_t_tanh_prime_c_t, axis=(1,2)))
            
            # Gradients
            for k in W_gradient:
                W_gradient[k] = W_gradient[k] + np.sum(H_W_gradient[k]*np.expand_dims(Y_error[:,t,:], axis=(1,2)), axis=(0,3))
                U_gradient[k] = U_gradient[k] + np.sum(H_U_gradient[k]*np.expand_dims(Y_error[:,t,:], axis=(1,2)), axis=(0,3))
            error = error + np.sum(H_X_gradient['H']*np.expand_dims(Y_error[:,t,:], axis=(1,2)), axis=-1)
            
            C_t_minus_1 = C_t
            H_t_minus_1 = H_t
                
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
        return gradients, error
    
    def update(self, W_forget_gradient, W_input_gradient, W_output_gradient, W_concat_gradient, U_forget_gradient, U_input_gradient, U_output_gradient, U_concat_gradient):
        self.W_forget += W_forget_gradient
        self.W_input += W_input_gradient
        self.W_output += W_output_gradient
        self.W_concat += W_concat_gradient
        self.U_forget += U_forget_gradient
        self.U_input += U_input_gradient
        self.U_output += U_output_gradient
        self.U_concat += U_concat_gradient


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
            return X@(self.W * np.where(np.random.random(size=self.W.shape)>self.dropout_p, 1.0, 0.0) / (1-self.dropout_p))
        return X@self.W
    
    def backward(self, X, Y_error, l2=0.0):
        if self.mask_zero:
            X = X-1
        X = dummy_encode(X, self.F)
        W_gradient = np.sum(np.transpose(X, axes=(1,2,0))@np.transpose(Y_error, axes=(1,0,2)), axis=0) + 2*l2 * self.W / self.F / self.D
        return {'W_gradient': W_gradient}, None
    
    def update(self, W_gradient):
        self.W += W_gradient


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
        return X@W * np.sqrt(self.D) + np.expand_dims(self.positional_encoding[:X.shape[1],:], axis=0)
    
    def backward(self, X, Y_error, l2=0.0):
        if self.mask_zero:
            X = X-1
        X = dummy_encode(X, self.F)
        W_gradient = np.sum(np.transpose(X, axes=(1,2,0))@np.transpose(Y_error, axes=(1,0,2)), axis=0) + 2*l2 * self.W / self.F / self.D
        return {'W_gradient': W_gradient * np.sqrt(self.D)}, None
    
    def update(self, W_gradient):
        self.W += W_gradient


class AddLayer:
    
    def __init__(self):
        return
    
    def forward(self, X):
        return np.sum(X, axis=0)
    
    def backward(self, X, Y_error):
        return np.repeat(np.expand_dims(Y_error, axis=0), len(X), axis=0)


class MultiHeadAttentionLayer:
    
    def __init__(self, T, F, NH=10, add_bias=True, dropout_p=0.0, use_causal_mask=False):
        if F % NH != 0:
            raise ValueError(f"The input size must be a multiple of the number of heads ({NH} here)")
        self.add_bias = add_bias
        self.dropout_p = dropout_p
        self.T = T
        self.F = F
        self.NH = NH
        self.W_q = np.random.normal(loc=0, scale=1/self.F, size=(self.F, self.F)) # query input weights
        self.W_k = np.random.normal(loc=0, scale=1/self.F, size=(self.F, self.F)) # key input weights
        self.W_v = np.random.normal(loc=0, scale=1/self.F, size=(self.F, self.F)) # value input weights
        self.mask = np.tril(np.ones((self.T, self.T))) if use_causal_mask else np.ones((self.T, self.T))
        
    def split(self, X):
        query_size = self.F // self.NH
        X = np.reshape(X, (X.shape[0], X.shape[1], self.NH, query_size)) # (N x T x NH x F/NH)
        return np.transpose(X, axes=(0,2,1,3)) # (N x NH x T x F/NH)
    
    def merge(self, X):
        X = np.transpose(X, axes=(0,2,1,3)) # (N x T x NH x F/NH)
        return np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]*X.shape[3])) # (N x T x F)
    
    def forward(self, X_q, X_k=None, X_v=None, inference=True):
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
        S = S * np.expand_dims(self.mask, axis=(0,1)) # (N x NH x T x T)
        S = softmax(S / np.sqrt(self.F // self.NH), axis=(2,3)) # (N x NH x T x T)
        S = S@H['V'] # (N x NH x T x F/NH)
        # Attention score merge
        S = self.merge(S) # (N x T x F)
        return S
        
    def backward(self, Y_error, X_q, X_k=None, X_v=None, l2=0.0):
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
        S = S * np.expand_dims(self.mask, axis=(0,1)) # (N x NH x T x T)
        S = softmax(S / np.sqrt(query_size), axis=(2,3)) # (N x NH x T x T)
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
        self.W_q += W_q_gradient
        self.W_k += W_k_gradient
        self.W_v += W_v_gradient


class BaseAttentionLayer:
    
    def __init__(self, T, F, NH=10, add_bias=True, dropout_p=0.0, use_causal_mask=False):
        self.mha_layer = MultiHeadAttentionLayer(T, F, NH=NH, add_bias=add_bias, dropout_p=dropout_p, use_causal_mask=use_causal_mask)
        self.norm_layer = LayerNormalization()
        self.add_layer = AddLayer()
        
    def forward(self, X_q, X_k=None, X_v=None, X_skip=None, inference=True):
        X_mha = self.mha_layer.forward(X_q, X_k, X_v, inference=inference)
        X_add = self.add_layer.forward([X_mha, X_q if X_skip is None else X_skip])
        return self.norm_layer.forward(X_add)
    
    def backward(self, Y_error, X_q, X_k=None, X_v=None, X_skip=None, l2=0.0):
        X_mha = self.mha_layer.forward(X_q, X_k, X_v, inference=False)
        X_add = self.add_layer.forward([X_mha, X_q if X_skip is None else X_skip])
        norm_gradients, X_add_error = self.norm_layer.backward(X_add, Y_error, l2=l2)
        X_mha_error, X_skip_error = self.add_layer.backward([X_mha, X_q if X_skip is None else X_skip], X_add_error)
        mha_gradients, (X_q_error, X_k_error, X_v_error) = self.mha_layer.backward(X_mha_error, X_q, X_k, X_v, l2=l2)
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
    
    def __init__(self, T, F, NH=10, add_bias=True, dropout_p=0.0, use_causal_mask=False):
        super().__init__(T, F, NH=NH, add_bias=add_bias, dropout_p=dropout_p, use_causal_mask=use_causal_mask)
        
    def forward(self, X, X_context, inference=True):
        return super().forward(X_q=X, X_k=X_context, X_v=X_context, X_skip=X, inference=inference)
    
    def backward(self, Y_error, X, X_context, l2=0.0):
        gradients, (X_q_error, X_k_error, X_v_error, X_skip_error) = super().backward(
            Y_error, X_q=X, X_k=X_context, X_v=X_context, X_skip=X, l2=l2)
        return gradients, (X_q_error + X_skip_error, X_k_error + X_v_error)


class SelfAttentionLayer(BaseAttentionLayer):
    
    def __init__(self, T, F, NH=10, add_bias=True, dropout_p=0.0, use_causal_mask=False):
        super().__init__(T, F, NH=NH, add_bias=add_bias, dropout_p=dropout_p, use_causal_mask=use_causal_mask)
        
    def forward(self, X, inference=True):
        return super().forward(X_q=X, X_k=X, X_v=X, X_skip=X, inference=inference)
    
    def backward(self, Y_error, X, l2=0.0):
        gradients, (X_q_error, X_k_error, X_v_error, X_skip_error) = super().backward(
            Y_error, X_q=X, X_k=X, X_v=X, X_skip=X, l2=l2)
        return gradients, (X_q_error + X_skip_error + X_k_error + X_v_error)
