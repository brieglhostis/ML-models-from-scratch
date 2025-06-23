
import numpy as np

from utils import sigmoid, mse, r_square, log_loss, accuracy, precision, recall, f1_score, auc, dummy_encode, random_argmin, random_argmax, frank_wolfe_quadratic_program
from optimizers import GradientDescentOptimizer, AdaGradOptimizer, RMSPropOptimizer, AdamOptimizer
from activations import LinearActivation, ReLuActivation, TanHActivation
from layers import DenseLayer, RecurrentLayer, LSTMLayer
from normalization import BatchNormalization


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


class LogisticRegression:
    """
    Binary logistic regression model:
    Y = 1 / (1 + exp(- X W + b))
    """

    def __init__(self, add_bias=True, l2=0.0):
        """
        Arguments:
         - add_bias (bool) - set to True to add a bias to the equation
         - l2 (float)      - L2 regularization parameter 
        """
        self.add_bias = add_bias
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
        return log_loss(Y, Y_pred) + self.l2 * np.mean(np.square(self.W), axis=0)

    def predict(self, X):
        """
        Compute target probability predictions for a set of input samples 
        Arguments:
         - X (np.ndarray) - input features (NxF)
        """
        if self.W is None:
            raise ValueError('Cannot predict before model is fitted')
        if self.add_bias and X.shape[-1] == self.W.shape[0]-1:
            X = np.c_[X, np.ones(X.shape[0])]
        return sigmoid(X @ self.W)

    def fit(
        self, X, Y, X_val=None, Y_val=None, learning_rate=0.1, epochs=100, batch_size=1000
        , optimizer='Adam', gradient_decay=0.9, gradient_norm_decay=0.999, verbose=True):
        """
        Fit model using gradient descent to minimize loss
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
        elif optimizer == 'AdaGrad':
            optimizer = AdaGradOptimizer()
        elif optimizer == 'RMSProp':
            optimizer = RMSPropOptimizer(gradient_decay=gradient_decay)
        elif optimizer == 'Adam':
            optimizer = AdamOptimizer(gradient_decay=gradient_decay, gradient_norm_decay=gradient_norm_decay)
        else:
            raise ValueError(f"Optimizer must be one of 'Regular', 'AdaGrad', 'RMSProp', or 'Adam'")

        if self.add_bias:
            X = np.c_[X, np.ones(X.shape[0])]

        # Batch creation
        if batch_size >= X.shape[0]:
            X_batches = [X]
            Y_batches = [Y]
        else:
            X_batches = [X[i*batch_size:min(X.shape[0],(i+1)*batch_size)] for i in range(X.shape[0] // batch_size)]
            Y_batches = [Y[i*batch_size:min(X.shape[0],(i+1)*batch_size)] for i in range(X.shape[0] // batch_size)]

        self.W = np.random.normal(loc=0, scale=1/X.shape[-1], size=(X.shape[-1], Y.shape[-1]))
        self.history = []
        for e in range(epochs):
            # Fitting
            for X_batch, Y_batch in zip(X_batches, Y_batches):
                Y_pred = sigmoid(X_batch @ self.W)
                Y_error = (Y_pred - Y_batch) / Y_batch.shape[0]
                W_gradient = X_batch.T @ (Y_error * Y_pred) + self.l2 * self.W / self.W.shape[0]
                self.W -= optimizer.update({'W_gradient': W_gradient}, learning_rate=learning_rate)['W_gradient']

            # Evaluation
            Y_pred = self.predict(X)
            metrics = {
                'train_loss': self.loss(X, Y)
                , 'train_log_loss': log_loss(Y, Y_pred)
                , 'train_accuracy': accuracy(Y, Y_pred)
                , 'train_precision': precision(Y, Y_pred)
                , 'train_recall': recall(Y, Y_pred)
                , 'train_f1_score': f1_score(Y, Y_pred)
                , 'train_auc': auc(Y, Y_pred)
            }
            if X_val is not None and Y_val is not None:
                Y_pred_val = self.predict(X_val)
                metrics['val_loss'] = self.loss(X_val, Y_val)
                metrics['val_log_loss'] = log_loss(Y_val, Y_pred_val)
                metrics['val_accuracy'] = accuracy(Y_val, Y_pred_val)
                metrics['val_precision'] = precision(Y_val, Y_pred_val)
                metrics['val_recall'] = recall(Y_val, Y_pred_val)
                metrics['val_f1_score'] = f1_score(Y_val, Y_pred_val)
                metrics['val_auc'] = auc(Y_val, Y_pred_val)
            self.history.append(metrics)

            # Verbose
            if verbose and (e+1) % 10 == 0:
                print(f"Epoch {e+1}/{epochs} - train loss = {metrics['train_loss'].mean():.4f}" + (f", validation loss = {metrics['val_loss'].mean():.4f}" if 'val_log_loss' in metrics else ""))

        return self.history


class RegressionDecisionTree:
    """
    Regression decision tree
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        Arguments:
         - max_depth (int)         - maximum dept of tree
         - min_samples_split (int) - minimum number of samples to allow a split
         - min_samples_leaf (int)  - minimum number of samples in every leaf
        """
        self.D = None
        self.max_depth = max_depth
        self.min_samples_split = max(min_samples_split, 2*min_samples_leaf)
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def predict_sub_tree(self, X, Y_pred, tree):
        """
        Recursively compute predictions for a set of input samples 
        Arguments:
         - X (np.ndarray) - input features (NxF)
         - Y (np.ndarray) - prediction array to fill (NxD)
         - tree           - sub tree to use for prediction
        """
        split, sub_tree = tree
        left_indexes = X[:,split[0]] <= split[1]
        right_indexes = X[:,split[0]] > split[1]
        if np.sum(left_indexes) > 0:
            if isinstance(sub_tree[0], float) or isinstance(sub_tree[0], int):
                # Set predictions as the leaf value
                Y_pred[left_indexes] = sub_tree[0]
            else:
                # Explore subtree to set predictions
                Y_pred[left_indexes] = self.predict_sub_tree(X[left_indexes], Y_pred[left_indexes], sub_tree[0])
        if np.sum(right_indexes) > 0:
            if isinstance(sub_tree[1], float) or isinstance(sub_tree[1], int):
                # Set predictions as the leaf value
                Y_pred[right_indexes] = sub_tree[1]
            else:
                # Explore subtree to set predictions
                Y_pred[right_indexes] = self.predict_sub_tree(X[right_indexes], Y_pred[right_indexes], sub_tree[1])
        return Y_pred

    def predict(self, X):
        """
        Compute predictions for a set of input samples 
        Arguments:
         - X (np.ndarray) - input features (NxF)
        """
        if self.tree is None:
            raise ValueError('Cannot predict before model is fitted')
        return np.array([self.predict_sub_tree(X, np.zeros(X.shape[0]), self.tree[d]) for d in range(self.D)]).T

    def fit_sub_tree(self, X, Y, D=0):
        """
        Recursively fit model by minimizing variance within each sub tree
        Arguments:
         - X (np.ndarray) - input training features (NxF)
         - Y (np.ndarray) - target training actuals (NxC)
         - D (int)        - current depth
        """
        V_min = np.var(Y)
        if V_min == 0.0:
            # Only one value in Y, so no need to split further
            return np.mean(Y)
        split_min = None
        N = Y.shape[0]
        for f in range(X.shape[-1]):
            # Order X_f and Y according to the order of values in X_f
            order = np.argsort(X[:,f])
            ordered_X = X[order,f]
            ordered_Y = Y[order]
            # Initialize N, M, and V for left and right arrays
            N_left, N_right = 0, 0
            M_left, M_right = 0.0, 0.0
            V_left, V_right = 0.0, 0.0
            # Initialize the weighted average variance
            V = np.zeros(N)
            # Penalize equal values at the end of X_f to ensure that some values are kept in the right arrays
            for n in range(1, N):
                if ordered_X[N-n-1] == ordered_X[N-1]:
                    V[N-n-1] += V_min
                else:
                    break
            # Iteratively compute the weighted average variance
            for n in range(N):
                dM_left, dM_right = (ordered_Y[n] - M_left) / (n+1), (ordered_Y[N-n-1] - M_right) / (n+1)
                M_left += dM_left
                M_right += dM_right
                V_left += ((ordered_Y[n] - M_left - dM_left)**2 - V_left + n * dM_left**2) / (n+1)
                V_right += ((ordered_Y[N-n-1] - M_right - dM_right)**2 - V_right + n * dM_right**2) / (n+1)
                V[n] += n * V_left / N
                V[N-n-1] += n * V_right / N
            # Find optimal split
            n_min = self.min_samples_leaf + random_argmin(V[self.min_samples_leaf:N-self.min_samples_leaf+1])
            if V[n_min] < V_min:
                V_min = V[n_min]
                split_min = (f, X[:,f][order][n_min])
        if split_min is None:
            # No better split found
            return np.mean(Y)
        Y_left = Y[X[:,split_min[0]] <= split_min[1]]
        Y_right = Y[X[:,split_min[0]] > split_min[1]]
        if D+1 == self.max_depth:
            # Generate leaves
            return [split_min, [np.mean(Y_left), np.mean(Y_right)]]
        X_left = X[X[:,split_min[0]] <= split_min[1]]
        X_right = X[X[:,split_min[0]] > split_min[1]]
        # Generate sub tree or leaves depending on number of samples
        return [split_min, [
              self.fit_sub_tree(X_left, Y_left, D+1) if X_left.shape[0] >= self.min_samples_split else np.mean(Y_left)
              , self.fit_sub_tree(X_right, Y_right, D+1) if X_right.shape[0] >= self.min_samples_split else np.mean(Y_right)]]

    def fit(self, X, Y):
        """
        Fit model by minimizing variance within each sub tree
        Arguments:
         - X (np.ndarray) - input training features (NxF)
         - Y (np.ndarray) - target training actuals (NxD)
        """
        self.D = Y.shape[-1]
        self.tree = [self.fit_sub_tree(X, Y[:,d]) for d in range(self.D)]
        return self.tree


class ClassificationDecisionTree:
    """
    Classification decision tree
    """

    INFORMATION_FUNCTIONS = ['gini', 'entropy']

    def __init__(self, information_function='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        Arguments:
         - information_function (str) - information function name, either 'gini' or 'entropy'
         - max_depth (int)            - maximum dept of tree
         - min_samples_split (int)    - minimum number of samples to allow a split
         - min_samples_leaf (int)     - minimum number of samples in every leaf
        """
        assert information_function in self.INFORMATION_FUNCTIONS
        self.information_function = information_function
        self.C = None
        self.max_depth = max_depth
        self.min_samples_split = max(min_samples_split, 2*min_samples_leaf)
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def gini_index(self, Y):
        """
        Compute the Gini index: G = 1 - sum(P_c²)
        Arguments:
         - Y (np.ndarray) - target actuals (NxC)
        """
        return 1 - np.sum(np.square(np.mean(Y, axis=0)), axis=0)

    def self_entropy(self, Y, axis=0):
        """
        Compute the self entropy: entropy = - sum(P_c * log(P_c))
        Arguments:
         - Y (np.ndarray) - target actuals (NxC)
        """
        P = np.mean(Y, axis=0)
        return - np.sum(P*np.log(P+1e-6), axis=0)

    def predict_sub_tree(self, X, Y_pred, tree):
        """
        Recursively compute predictions for a set of input samples 
        Arguments:
         - X (np.ndarray) - input features (NxF)
         - Y (np.ndarray) - prediction array to fill (NxC)
         - tree           - sub tree to use for prediction
        """
        split, sub_tree = tree
        left_indexes = X[:,split[0]] <= split[1]
        right_indexes = X[:,split[0]] > split[1]
        if np.sum(left_indexes) > 0:
            if isinstance(sub_tree[0], int):
                # Set predictions as the leaf value
                Y_pred[left_indexes] = dummy_encode(sub_tree[0] * np.ones(np.sum(left_indexes)), self.C)
            else:
                # Explore subtree to set predictions
                Y_pred[left_indexes] = self.predict_sub_tree(X[left_indexes], Y_pred[left_indexes], sub_tree[0])
        if np.sum(right_indexes) > 0:
            if isinstance(sub_tree[1], int):
                # Set predictions as the leaf value
                Y_pred[right_indexes] = dummy_encode(sub_tree[1] * np.ones(np.sum(right_indexes)), self.C)
            else:
                # Explore subtree to set predictions
                Y_pred[right_indexes] = self.predict_sub_tree(X[right_indexes], Y_pred[right_indexes], sub_tree[1])
        return Y_pred

    def predict(self, X):
        """
        Compute predictions for a set of input samples 
        Arguments:
         - X (np.ndarray) - input features (NxF)
        """
        if self.tree is None:
            raise ValueError('Cannot predict before model is fitted')
        return self.predict_sub_tree(X, np.zeros((X.shape[0], self.C)), self.tree)

    def fit_sub_tree(self, X, Y, D=0):
        """
        Recursively fit model by minimizing information function within each sub tree
        Arguments:
         - X (np.ndarray) - input training features (NxF)
         - Y (np.ndarray) - target training actuals (NxC)
         - D (int)        - current depth
        """
        I_min = self.gini_index(Y) if self.information_function == 'gini' else self.self_entropy(Y)
        if I_min <= 0.0:
            # Only one value in Y, so no need to split further
            return int(random_argmax(np.mean(Y, axis=0)))
        split_min = None
        N = Y.shape[0]
        for f in range(X.shape[-1]):
            # Order X_f and Y according to the order of values in X_f
            order = np.argsort(X[:,f])
            ordered_X = X[order,f]
            ordered_Y = Y[order]
            # Compute probability values for left and right arrays
            P_left = np.cumsum(ordered_Y, axis=0) / np.expand_dims(np.arange(1, N+1), axis=-1)
            P_right = np.cumsum(ordered_Y[::-1], axis=0)[::-1] / np.expand_dims(np.arange(N, 0, -1), axis=-1)
            # Initialize the weighted average information
            I = np.zeros(N)
            # Penalize equal values at the end of X_f to ensure that some values are kept in the right arrays
            for n in range(1, N):
                if ordered_X[N-n-1] == ordered_X[N-1]:
                    I[N-n-1] += I_min
                else:
                    break
            # Iteratively compute the weighted average information
            if self.information_function == 'gini':
                I += np.arange(1, N+1) * (1 - np.sum(np.square(P_left), axis=-1)) / N
                I += np.arange(N, 0, -1) * (1 - np.sum(np.square(P_right), axis=-1)) / N
            else:
                I -= np.arange(1, N+1) * np.sum(P_left * np.log(P_left + 1e-6), axis=-1) / N
                I -= np.arange(N, 0, -1) * np.sum(P_right * np.log(P_right + 1e-6), axis=-1) / N
            # Find optimal split
            n_min = self.min_samples_leaf + random_argmin(I[self.min_samples_leaf:N-self.min_samples_leaf+1])
            if I[n_min] < I_min:
                I_min = I[n_min]
                split_min = (f, X[:,f][order][n_min])
        if split_min is None:
            # No better split found
            return int(random_argmax(np.mean(Y, axis=0)))
        Y_left = Y[X[:,split_min[0]] <= split_min[1]]
        Y_right = Y[X[:,split_min[0]] > split_min[1]]
        if D+1 == self.max_depth:
            # Generate leaves
            return [split_min, [int(random_argmax(np.mean(Y_left, axis=0))), int(random_argmax(np.mean(Y_right, axis=0)))]]
        X_left = X[X[:,split_min[0]] <= split_min[1]]
        X_right = X[X[:,split_min[0]] > split_min[1]]
        # Generate sub tree or leaves depending on number of samples
        return [split_min, [
              self.fit_sub_tree(X_left, Y_left, D+1) if X_left.shape[0] >= self.min_samples_split else int(random_argmax(np.mean(Y_left, axis=0)))
              , self.fit_sub_tree(X_right, Y_right, D+1) if X_right.shape[0] >= self.min_samples_split else int(random_argmax(np.mean(Y_right, axis=0)))]]

    def fit(self, X, Y):
        """
        Fit model by minimizing information function within each sub tree
        Arguments:
         - X (np.ndarray) - input training features (NxF)
         - Y (np.ndarray) - target training actuals (NxC)
        """
        self.C = Y.shape[-1]
        self.tree = self.fit_sub_tree(X, Y)
        return self.tree


class RegressionRandomForest:
    """
    Regression random forest
    """

    def __init__(self, T=10, F=None, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        Arguments:
         - T (int)                 - number of trees
         - F (int)                 - number of features per tree, if None then uses the square root of the total number of features
         - max_depth (int)         - maximum dept of tree
         - min_samples_split (int) - minimum number of samples to allow a split
         - min_samples_leaf (int)  - minimum number of samples in every leaf
        """
        self.D = None
        self.T = T
        self.F = F
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.feature_masks = None
        self.trees = None
        
    def predict(self, X):
        """
        Compute predictions for a set of input samples by averaging the predictions from all trees
        Arguments:
         - X (np.ndarray) - input features (NxF)
        """
        if self.feature_masks is None or self.trees is None:
            raise ValueError('Cannot predict before model is fitted')
        return np.mean(np.array([
            tree.predict(X[:,feature_mask]) 
            for feature_mask, tree, in zip(self.feature_masks, self.trees)
        ]), axis=0)
    
    def fit(self, X, Y):
        """
        Fit decision trees using sample and feature bagging
        Arguments:
         - X (np.ndarray) - input training features (NxF)
         - Y (np.ndarray) - target training actuals (NxD)
        """
        self.D = Y.shape[-1]
        self.F = int(np.sqrt(X.shape[-1])) if self.F is None else self.F
        # Compute bagging features
        self.feature_masks = [
            np.random.choice(np.arange(X.shape[-1]), self.F, replace=False) 
            if self.F < X.shape[-1] else np.arange(X.shape[-1]) 
            for t in range(self.T)]
        # Initialize decision trees
        self.trees = [
            RegressionDecisionTree(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
                , min_samples_leaf=self.min_samples_leaf) 
            for t in range(self.T)]
        # Fit each decision tree using bootstrap on samples
        for t in range(self.T):
            sample_mask = np.random.choice(np.arange(X.shape[0]), X.shape[0], replace=True)
            self.trees[t].fit(X[sample_mask,:][:,self.feature_masks[t]], Y[sample_mask,:])
        return self.trees


class ClassificationRandomForest:
    """
    Classification random forest
    """

    INFORMATION_FUNCTIONS = ['gini', 'entropy']

    def __init__(self, T=10, F=None, information_function='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        Arguments:
         - T (int)                    - number of trees
         - F (int)                    - number of features per tree, if None then uses the square root of the total number of features
         - information_function (str) - information function name, either 'gini' or 'entropy'
         - max_depth (int)            - maximum dept of tree
         - min_samples_split (int)    - minimum number of samples to allow a split
         - min_samples_leaf (int)     - minimum number of samples in every leaf
        """
        assert information_function in self.INFORMATION_FUNCTIONS
        self.C = None
        self.T = T
        self.F = F
        self.information_function = information_function
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.feature_masks = None
        self.trees = None
        
    def predict(self, X):
        """
        Compute predictions for a set of input samples by finding the most common prediction
        Arguments:
         - X (np.ndarray) - input features (NxF)
        """
        if self.feature_masks is None or self.trees is None:
            raise ValueError('Cannot predict before model is fitted')
        return dummy_encode(random_argmax(np.mean(np.array([
            tree.predict(X[:,feature_mask]) 
            for feature_mask, tree, in zip(self.feature_masks, self.trees)
        ]), axis=0), axis=-1), self.C)
    
    def fit(self, X, Y):
        """
        Fit decision trees using sample and feature bagging
        Arguments:
         - X (np.ndarray) - input training features (NxF)
         - Y (np.ndarray) - target training actuals (NxC)
        """
        self.C = Y.shape[-1]
        self.F = int(np.sqrt(X.shape[-1])) if self.F is None else self.F
        # Compute bagging features
        self.feature_masks = [
            np.random.choice(np.arange(X.shape[-1]), self.F, replace=False) 
            if self.F < X.shape[-1] else np.arange(X.shape[-1]) 
            for t in range(self.T)]
        # Initialize decision trees
        self.trees = [
            ClassificationDecisionTree(
                information_function=self.information_function
                , max_depth=self.max_depth, min_samples_split=self.min_samples_split
                , min_samples_leaf=self.min_samples_leaf) 
            for t in range(self.T)]
        # Fit each decision tree using bootstrap on samples
        for t in range(self.T):
            sample_mask = np.random.choice(np.arange(X.shape[0]), X.shape[0], replace=True)
            self.trees[t].fit(X[sample_mask,:][:,self.feature_masks[t]], Y[sample_mask,:])
        return self.trees


class SupportVectorClassifier:
    """
    Support vector classifier:
    Y = sign(X W - b)
    """
    
    KERNELS = ['polynomial', 'gaussian']
    
    def __init__(self, l2=0.001, kernel='polynomial', kernel_polynomial_order=1, kernel_polynomial_offset=0.0, kernel_gaussian_sigma=1.0):
        """
        Arguments:
         - l2 (float)                       - L2 regularization parameter 
         - kernel (str)                     - name of the kernel, either 'polynomial' or 'gaussian'
         - kernel_polynomial_order (int)    - polynomial kernel order
         - kernel_polynomial_offset (float) - polynomial kernel offset
         - kernel_gaussian_sigma (float)    - gaussian radial kernel sigma
        """
        assert kernel in self.KERNELS
        self.l2 = l2
        self.kernel = kernel
        self.kernel_polynomial_order = kernel_polynomial_order
        self.kernel_polynomial_offset = kernel_polynomial_offset
        self.kernel_gaussian_sigma = kernel_gaussian_sigma
        self.alpha = None
        self.X = None
        self.Y = None
        self.b = None
        
    def kernel_function(self, X_1, X_2):
        """
        Compute the kernel function for two samples X
        Arguments:
         - X_1 (np.ndarray) - first input features (NxF)
         - X_2 (np.ndarray) - second input features (NxF)
        """
        if self.kernel == 'polynomial':
            return np.power((X_1 @ X_2.T) + self.kernel_polynomial_offset, self.kernel_polynomial_order)
        elif self.kernel == 'gaussian':
            if len(X_1.shape) == 1 and len(X_2.shape) == 1:
                return np.exp(- (X_1 - X_2) @ (X_1 - X_2).T / 2 / self.kernel_gaussian_sigma**2)
            elif len(X_1.shape) == 2:
                return np.array([self.kernel_function(X_1[i], X_2) for i in range(X_1.shape[0])])
            elif len(X_2.shape) == 2:
                return np.array([self.kernel_function(X_1, X_2[j]) for j in range(X_2.shape[0])])
        
    def predict(self, X):
        """
        Compute predictions for a set of input samples 
        Arguments:
         - X (np.ndarray) - input features (NxF)
        """
        if self.alpha is None or self.b is None:
            raise ValueError('Cannot predict before model is fitted')
        Y_pred = np.sign(self.kernel_function(X, self.X) @ (self.alpha * self.Y) - self.b)
        return 0.5 + Y_pred / 2
    
    def fit(self, X, Y, max_steps=10):
        """
        Fit model by maximizing the dual problem
        Arguments:
         - X (np.ndarray)  - input training features (NxF)
         - Y (np.ndarray)  - target training actuals (NxD)
         - max_steps (int) - maximum number of steps in the Frank Wolfe algorithm
        """
        N = X.shape[0]
        Y = 2.0 * Y[:,:1] - 1.0 # Transform labels to -1 and 1
        
        # Compute Q using kernel
        Q = Y * self.kernel_function(X, X) * Y.T
        c = - np.ones((N,1))
        
        # Inequality constraints: A X <= b
        A = 1.0 * np.eye(N)
        b = 1/(2*N*self.l2) * np.ones((N, 1))
        
        # Equality constraints: A X = b
        A_eq = Y.T
        b_eq = np.zeros((1, 1))
        
        # alpha initialization with a random vector
        alpha_0 = np.zeros((N, 1))
        alpha_0[np.random.choice(np.where(Y[:,0] == 1.0)[0])] = 1/(2*N*self.l2)
        alpha_0[np.random.choice(np.where(Y[:,0] == -1.0)[0])] = 1/(2*N*self.l2)
        
        # Dual problem solving using the Frank Wolfe quadratic program
        self.alpha = frank_wolfe_quadratic_program(alpha_0, Q, c, A, b, A_eq, b_eq, max_steps=max_steps)
        
        # Weights and border width calculation
        self.X, self.Y = X, Y
        self.b = self.kernel_function(X[np.argmax(self.alpha)], self.X) @ (self.alpha * self.Y) - Y[np.argmax(self.alpha)]


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
