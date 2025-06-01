
import numpy as np



def sigmoid(x, safety_threshold=200):
    exp_safety_threshold = np.exp(safety_threshold)
    exp_minus_x = np.where(x<-safety_threshold, exp_safety_threshold, np.exp(-x))
    return 1 / (1+exp_minus_x)

def tanh(x, safety_threshold=200):
    exp_safety_threshold = np.exp(safety_threshold)
    exp_x = np.where(x>safety_threshold, exp_safety_threshold, np.exp(x))
    exp_minus_x = np.where(x<-safety_threshold, exp_safety_threshold, np.exp(-x))
    return (exp_x-exp_minus_x) / (exp_x+exp_minus_x)

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
        H_W_gradient = np.zeros((N, self.F, self.D)) # Derivative of H wrt W
        V_gradient = np.zeros((self.D, self.D)) # V gradient
        H_V_gradient = np.zeros((N, self.D, self.D)) # Derivative of H wrt V
        error = np.zeros((N, self.T, non_bias_F)) # Propagated error
        H_X_gradient = np.zeros((N, self.T, non_bias_F, self.D)) # Derivative of H wrt X
        for t in range(self.T):
            H_W_gradient = np.repeat(np.expand_dims(X[:,t,:], axis=-1), self.D, axis=-1) + H_W_gradient @ np.diag(np.diag(self.V))
            H_V_gradient = np.repeat(np.expand_dims(H, axis=-1), self.D, axis=-1) + H_V_gradient * np.repeat(np.expand_dims(self.V, axis=0), N, axis=0)
            H = X[:,t,:]@self.W + H@self.V
            H_X_gradient[:,t,:,:] = np.repeat(np.expand_dims(non_bias_W, axis=0), N, axis=0)
            for s in range(t):
                H_X_gradient[:,s,:,:] = H_X_gradient[:,s,:,:]@self.V
            W_gradient = W_gradient + np.sum(np.array([H_W_gradient[n,:,:]*np.expand_dims(Y_error[n,t,:], axis=0) for n in range(N)]), axis=0)
            V_gradient = V_gradient + np.sum(np.array([H_V_gradient[n,:,:]*np.expand_dims(Y_error[n,t,:], axis=0) for n in range(N)]), axis=0)
            error = error + np.sum(np.array([H_X_gradient[:,:,:,d]*np.expand_dims(Y_error[:,t,d], axis=(1,2)) for d in range(self.D)]), axis=0)
                    
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
        C_W_gradient = {k: np.zeros((N, self.F, self.D)) for k in ['f', 'i', 'c']} # Derivative of C wrt W
        H_W_gradient = {k: np.zeros((N, self.F, self.D)) for k in ['f', 'i', 'o', 'c']} # Derivative of H wrt W
        U_gradient = {k: np.zeros((self.D, self.D)) for k in ['f', 'i', 'o', 'c']} # W gradient
        C_U_gradient = {k: np.zeros((N, self.D, self.D)) for k in ['f', 'i', 'c']} # Derivative of C wrt W
        H_U_gradient = {k: np.zeros((N, self.D, self.D)) for k in ['f', 'i', 'o', 'c']} # Derivative of H wrt W
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
            C_W_gradient['f'] = np.repeat(np.expand_dims(tmp['f'], axis=-2), self.F, axis=-2) * (
                np.repeat(np.expand_dims((1-tmp['f'])*C_t_minus_1, axis=-2), self.F, axis=-2) * (
                    np.repeat(np.expand_dims(X[:,t,:], axis=-1), self.D, axis=-1) + 
                    H_W_gradient['f'] @ np.diag(np.diag(U['f']))) + 
                 C_W_gradient['f'])
            H_W_gradient['f'] = np.repeat(np.expand_dims(o_t_tanh_prime_c_t, axis=-2), self.F, axis=-2) * C_W_gradient['f']
            C_U_gradient['f'] = np.repeat(np.expand_dims(tmp['f'], axis=-2), self.D, axis=-2) * (
                np.repeat(np.expand_dims((1-tmp['f'])*C_t_minus_1, axis=-2), self.D, axis=-2) * (
                    np.repeat(np.expand_dims(H_t_minus_1, axis=-1), self.D, axis=-1) + 
                    H_U_gradient['f'] * np.repeat(np.expand_dims(U['f'], axis=0), N, axis=0)) + 
                C_U_gradient['f'])
            H_U_gradient['f'] = np.repeat(np.expand_dims(o_t_tanh_prime_c_t, axis=-2), self.D, axis=-2) * C_U_gradient['f']

            # Input gate
            C_W_gradient['i'] = (
                np.repeat(np.expand_dims(tmp['c']*tmp['i']*(1-tmp['i']), axis=-2), self.F, axis=-2) * (
                    np.repeat(np.expand_dims(X[:,t,:], axis=-1), self.D, axis=-1) + 
                    H_W_gradient['i'] @ np.diag(np.diag(U['i']))) + 
                np.repeat(np.expand_dims(tmp['f'], axis=-2), self.F, axis=-2) * C_W_gradient['i'])
            H_W_gradient['i'] = np.repeat(np.expand_dims(o_t_tanh_prime_c_t, axis=-2), self.F, axis=-2) * C_W_gradient['i']
            C_U_gradient['i'] = (
                np.repeat(np.expand_dims(tmp['c']*tmp['i']*(1-tmp['i']), axis=-2), self.D, axis=-2) * (
                    np.repeat(np.expand_dims(H_t_minus_1, axis=-1), self.D, axis=-1) + 
                    H_U_gradient['i'] * np.repeat(np.expand_dims(U['i'], axis=0), N, axis=0)) + 
                np.repeat(np.expand_dims(tmp['f'], axis=-2), self.D, axis=-2) * C_U_gradient['i'])
            H_U_gradient['i'] = np.repeat(np.expand_dims(o_t_tanh_prime_c_t, axis=-2), self.D, axis=-2) * C_U_gradient['i']

            # Output gate
            H_W_gradient['o'] = np.repeat(np.expand_dims(tmp['o']*(1-tmp['o']) * tanh_c_t, axis=-2), self.F, axis=-2) * (
                np.repeat(np.expand_dims(X[:,t,:], axis=-1), self.D, axis=-1) + 
                H_W_gradient['o'] @ np.diag(np.diag(U['o'])))
            H_U_gradient['o'] = np.repeat(np.expand_dims(tmp['o']*(1-tmp['o']) * tanh_c_t, axis=-2), self.D, axis=-2) * (
                np.repeat(np.expand_dims(H_t_minus_1, axis=-1), self.D, axis=-1) + 
                H_U_gradient['o'] * np.repeat(np.expand_dims(U['o'], axis=0), N, axis=0))

            # Concat gate
            C_W_gradient['c'] = (
                np.repeat(np.expand_dims(tmp['i']*(1-np.square(tmp['c'])), axis=-2), self.F, axis=-2) * (
                    np.repeat(np.expand_dims(X[:,t,:], axis=-1), self.D, axis=-1) + 
                    H_W_gradient['c'] @ np.diag(np.diag(U['c']))) + 
                np.repeat(np.expand_dims(tmp['f'], axis=-2), self.F, axis=-2) * C_W_gradient['c'])
            H_W_gradient['c'] = np.repeat(np.expand_dims(o_t_tanh_prime_c_t, axis=-2), self.F, axis=-2) * C_W_gradient['c']
            C_U_gradient['c'] = (
                np.repeat(np.expand_dims(tmp['i']*(1-np.square(tmp['c'])), axis=-2), self.D, axis=-2) * (
                    np.repeat(np.expand_dims(H_t_minus_1, axis=-1), self.D, axis=-1) + 
                    H_U_gradient['c'] * np.repeat(np.expand_dims(U['c'], axis=0), N, axis=0)) + 
                np.repeat(np.expand_dims(tmp['f'], axis=-2), self.D, axis=-2) * C_U_gradient['c'])
            H_U_gradient['c'] = np.repeat(np.expand_dims(o_t_tanh_prime_c_t, axis=-2), self.D, axis=-2) * C_U_gradient['c']

            # Propagated error
            for k in ['f', 'i', 'o', 'c']:
                activation_derivate = 1-np.square(tmp[k]) if k == 'c' else tmp[k]*(1-tmp[k])
                H_X_gradient[k][:,t,:,:] = np.repeat(np.expand_dims(activation_derivate, axis=1), non_bias_F, axis=1) * np.repeat(np.expand_dims(non_bias_W[k], axis=0), N, axis=0)
                for s in range(t):
                    H_X_gradient[k][:,s,:,:] = np.repeat(np.expand_dims(activation_derivate, axis=1), non_bias_F, axis=1) * (H_X_gradient[k][:,s,:,:]@U[k])
            H_X_gradient['C'] = (
                H_X_gradient['f'] * np.repeat(np.repeat(np.expand_dims(C_t_minus_1, axis=(1,2)), self.T, axis=1), non_bias_F, axis=2) + 
                H_X_gradient['C'] * np.repeat(np.repeat(np.expand_dims(tmp['f'], axis=(1,2)), self.T, axis=1), non_bias_F, axis=2) + 
                H_X_gradient['i'] * np.repeat(np.repeat(np.expand_dims(tmp['c'], axis=(1,2)), self.T, axis=1), non_bias_F, axis=2) + 
                H_X_gradient['c'] * np.repeat(np.repeat(np.expand_dims(tmp['i'], axis=(1,2)), self.T, axis=1), non_bias_F, axis=2))
            H_X_gradient['H'] = (
                H_X_gradient['o'] * np.repeat(np.repeat(np.expand_dims(tanh_c_t, axis=(1,2)), self.T, axis=1), non_bias_F, axis=2) + 
                H_X_gradient['C'] * np.repeat(np.repeat(np.expand_dims(o_t_tanh_prime_c_t, axis=(1,2)), self.T, axis=1), non_bias_F, axis=2))
            
            # Gradients
            for k in W_gradient:
                W_gradient[k] = W_gradient[k] + np.sum(np.array([H_W_gradient[k][n,:,:]*np.expand_dims(Y_error[n,t,:], axis=0) for n in range(N)]), axis=0)
                U_gradient[k] = U_gradient[k] + np.sum(np.array([H_U_gradient[k][n,:,:]*np.expand_dims(Y_error[n,t,:], axis=0) for n in range(N)]), axis=0)
            error = error + np.sum(np.array([H_X_gradient['H'][:,:,:,d]*np.expand_dims(Y_error[:,t,d], axis=(1,2)) for d in range(self.D)]), axis=0)
            
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
