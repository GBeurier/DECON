from sklearn.cross_decomposition import PLSRegression
import numpy as np

def lw_pls(X_train, y_train, X_test, n_components=2, weighting='bisquare'):
    # Fit a PLS model
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)
    
    # Calculate the distance between each test sample and the training samples
    D = np.sum((X_train - X_test[:, np.newaxis])**2, axis=-1)
    
    # Apply a weighting scheme
    if weighting == 'bisquare':
        w = (1 - (D / np.max(D))**2)**2
    elif weighting == 'tricube':
        w = (1 - np.abs(D / np.max(D))**3)**3
    else:
        w = 1 / D
    
    # Build local models for each test sample
    y_pred = []
    for i in range(X_test.shape[0]):
        pls.fit(X_train, y_train, sample_weight=w[:, i])
        y_pred.append(pls.predict(X_test[i].reshape(1, -1)))
    
    return np.array(y_pred).ravel()



# Orthogonal Projections to Latent Structures (OPLS): OPLS is a variant of PLS that aims to separate the variation in the data that is predictive of the response variable from the variation that is orthogonal to it. This can help improve the interpretability of the model and reduce the risk of overfitting.

import numpy as np
from pyChemometrics.ChemometricsPLS import ChemometricsPLS
from pyChemometrics.ChemometricsScaler import ChemometricsScaler

# Load the NIRS data
# X_train: predictor variables for the training set (samples x variables)
# y_train: response variable for the training set (samples x 1)
# X_test: predictor variables for the test set (samples x variables)
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')

# Scale the data
x_scaler = ChemometricsScaler()
y_scaler = ChemometricsScaler(with_std=False)
X_train = x_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train)
X_test = x_scaler.transform(X_test)

# Fit an OPLS model
opls = ChemometricsPLS(n_components=2, deflation='orthogonal')
opls.fit(X_train, y_train)

# Make predictions on the test data
y_pred = opls.predict(X_test)

# Inverse transform the predictions
y_pred = y_scaler.inverse_transform(y_pred)

# Evaluate the model performance
rmse = np.sqrt(np.mean((y_pred - y_test)**2))
r2 = 1 - np.sum((y_pred - y_test)**2) / np.sum((y_test - np.mean(y_test))**2)

print(f'RMSE: {rmse:.4f}')
print(f'R2: {r2:.4f}')


# Sparse PLS: Sparse PLS is a variant of PLS that incorporates a sparsity constraint on the loading vectors. This can help improve the interpretability of the model by selecting only a small number of important variables.

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso
import numpy as np

def sparse_pls(X_train, y_train, X_test, n_components=2, alpha=1.0):
    # Fit a PLS model
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)
    
    # Extract the loading vectors
    W = pls.x_loadings_
    
    # Apply a sparsity constraint to the loading vectors
    lasso = Lasso(alpha=alpha)
    W_sparse = lasso.fit(W.T, X_train.T).coef_.T
    
    # Calculate the scores
    T = X_train @ W_sparse
    
    # Fit a linear regression model to the scores and the response variable
    beta = np.linalg.pinv(T.T @ T) @ T.T @ y_train
    
    # Make predictions on the test data
    T_test = X_test @ W_sparse
    y_pred = T_test @ beta
    
    return y_pred.ravel()



# Robust PLS: Robust PLS is a variant of PLS that is more resistant to outliers and noise in the data. It achieves this by using robust estimation methods to calculate the loading vectors and scores.

# Nonlinear PLS: Nonlinear PLS is a variant of PLS that can be used to model nonlinear relationships between the predictor and response variables. It achieves this by incorporating nonlinear transformations of the data into the PLS algorithm.

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

def npls(X_train, y_train, X_test, n_components=2, gamma=1.0):
    # Map the data into a higher-dimensional space using a kernel function
    K_train = rbf_kernel(X_train, gamma=gamma)
    K_test = rbf_kernel(X_test, X_train, gamma=gamma)
    
    # Fit a PLS model to the kernel matrix
    pls = PLSRegression(n_components=n_components)
    pls.fit(K_train, y_train)
    
    # Make predictions on the test data
    y_pred = pls.predict(K_test)
    
    return y_pred.ravel()





import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array

class LWPLS(BaseEstimator, RegressorMixin):
    def __init__(self, X_test, max_component_number=10,
                 lambda_in_similarity=0.1):
        self.X_test = X_test
        self.max_component_number = max_component_number
        self.lambda_in_similarity = lambda_in_similarity

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        y = np.reshape(y, (len(y), 1))
        X_test = self.X_test
        nb_test_samples = X_test.shape[0]
        estimated_y_test = np.zeros(
            (nb_test_samples, self.max_component_number))
        distance_matrix = cdist(X, X_test, 'euclidean')
        distance_std = distance_matrix.std(ddof=1)
        y_std = y.std(ddof=1)
        y_mean = y.mean()
        
        for test_sample_number in range(nb_test_samples):
            query_x_test = X_test[test_sample_number, :]
            query_x_test = np.reshape(query_x_test, (1, len(query_x_test)))
            distance = distance_matrix[:, test_sample_number]
            similarity = np.exp(-distance / distance_std / self.lambda_in_similarity)
            
            y_w = y.T.dot(similarity) / similarity.sum()
            x_w = np.reshape(X.T.dot(similarity) /
                              similarity.sum(), (1, X.shape[1]))
            centered_y = y - y_w
            centered_x = X - x_w
            centered_query_x_test = query_x_test - x_w
            estimated_y_test[test_sample_number, :] += y_w

            for component_number in range(self.max_component_number):
                w_a = np.reshape(centered_x.T.dot(similarity).dot(
                    centered_y) / np.linalg.norm(centered_x.T.dot(similarity).dot(centered_y)), (X.shape[1], 1))
                t_a = np.reshape(centered_x.dot(w_a), (X.shape[0], 1))
                p_a = np.reshape(centered_x.T.dot(similarity).dot(
                    t_a) / t_a.T.dot(similarity).dot(t_a), (X.shape[1], 1))
                q_a = centered_y.T.dot(similarity).dot(
                    t_a) / t_a.T.dot(similarity).dot(t_a)
                t_q_a = centered_query_x_test.dot(w_a)
                estimated_y_test[test_sample_number,
                                 component_number:] = estimated_y_test[test_sample_number, component_number:] + t_q_a * q_a
                if component_number != self.max_component_number:
                    centered_x = centered_x - t_a.dot(p_a.T)
                    centered_y = centered_y - t_a * q_a
                    centered_query_x_test = centered_query_x_test - \
                        t_q_a.dot(p_a.T)

        self.y_test_predict_ = estimated_y_test[:,
                                                component_number - 1] * y_std + y_mean

    def predict(self, X):
        check_array(X)
        return self.y_test_predict_
