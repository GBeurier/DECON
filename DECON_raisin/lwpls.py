import numpy as np

from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class LWPLS(BaseEstimator, RegressorMixin):
    """
    Locally-Weighted Partial Least Squares (LWPLS)

    Predict y-values of test samples using LWPLS

    Parameters
    ----------
    x_train: numpy.array or pandas.DataFrame
        autoscaled m x n matrix of X-variables of training data,
        m is the number of training sammples and
        n is the number of X-variables
    y_train: numpy.array or pandas.DataFrame
        autoscaled m x 1 vector of a Y-variable of training data
    x_test: numpy.array or pandas.DataFrame
        k x n matrix of X-variables of test data, which is autoscaled with training data,
        and k is the number of test samples
    max_component_number: int
        number of maximum components
    lambda_in_similarity: float
        parameter in similarity matrix

    Returns
    -------
    estimated_y_test : numpy.array
        k x 1 vector of estimated y-values of test data
    """

    def __init__(self, max_component_number, lambda_in_similarity, X_test, y_test):
        self.max_component_number = max_component_number
        self.lambda_in_similarity = lambda_in_similarity
        self.X_test = X_test
        self.y_test = y_test

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        y = np.reshape(y, (len(y), 1))
        X_test = self.X_test
        nb_test_samples = X_test.shape[0]
        estimated_y_test = np.zeros(
            (nb_test_samples, self.max_component_number))
        distance_matrix = cdist(X, X_test, 'euclidean')

        for test_sample_number in range(nb_test_samples):
            query_x_test = X_test[test_sample_number, :]
            query_x_test = np.reshape(query_x_test, (1, len(query_x_test)))
            distance = distance_matrix[:, test_sample_number]
            similarity = np.diag(
                np.exp(-distance / distance.std(ddof=1) / self.lambda_in_similarity))
            # similarity_matrix = np.diag(similarity) ## option

            y_w = y.T.dot(np.diag(similarity)) / similarity.sum()
            x_w = np.reshape(X.T.dot(np.diag(similarity)) /
                             similarity.sum(), (1, X.shape[1]))
            centered_y = y - y_w
            centered_x = X - np.ones((X.shape[0], 1)).dot(x_w)
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
                                                component_number - 1] * y.std(ddof=1) + y.mean()

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        return self.y_test_predict_
