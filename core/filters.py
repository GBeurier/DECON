import numpy as np
from scipy import signal, sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES


def crop(spectra, start=0, end=0, length=0):
    if length != 0:
        end = start + length
    return spectra[:, start:end]


class Crop(TransformerMixin, BaseEstimator):
    def __init__(self, start=0, end=0, length=0, *, copy=True):
        self.copy = copy
        self.start = start
        self.end = end
        self.length = length

    def _reset(self):
        pass

    def fit(self, X, y=None):
        if sparse.issparse(X):
            raise ValueError("Crop does not support sparse input")
        return self

    def transform(self, X, copy=None):
        if sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')

        X = self._validate_data(
            X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self
        )

        return crop(X, start=self.start, end=self.end, length=self.length)

    def _more_tags(self):
        return {"allow_nan": False}


def uniform_ft_resample(spectra, resample_size):
    resampled = []
    for i in range(len(spectra)):
        resampled.append(signal.resample(spectra[i], resample_size))
    return np.array(resampled)


class Uniform_FT_Resample(TransformerMixin, BaseEstimator):
    def __init__(self, resample_size, *, copy=True):
        self.copy = copy
        self.resample_size = resample_size

    def _reset(self):
        pass

    def fit(self, X, y=None):
        if sparse.issparse(X):
            raise ValueError("Crop does not support sparse input")
        return self

    def transform(self, X, copy=None):
        if sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!"')

        X = self._validate_data(
            X, reset=False, copy=self.copy, dtype=FLOAT_DTYPES, estimator=self
        )
        return uniform_ft_resample(X, resample_size=self.resample_size)

    def _more_tags(self):
        return {"allow_nan": False}
