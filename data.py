from pathlib import Path
import numpy as np
import pandas as pd
import os
import re
from scipy import signal

# from pinard.utils import load_csv


class WrongFormatError(Exception):
    """Exception raised when X et Y datasets are invalid."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        msg = ""
        if type(x) is np.ndarray:
            msg += "Invalid X shape: {}".format(x.shape) + " "
        if type(y) is np.ndarray:
            msg += "Invalid Y shape: {}".format(y.shape)
        super().__init__(msg)


def load_csv(x_fname, y_fname=None, y_cols=0, *, sep=None, x_hdr=None, y_hdr=None, x_index_col=None, y_index_col=None, autoremove_na=True):
    assert y_fname is not None or y_cols is not None
    # TODO - add assert/exceptions on non-numerical columns
    # TODO - better management of NaN and Null (esp exception msg)

    x_df = pd.read_csv(x_fname, sep=sep, header=x_hdr, index_col=x_index_col, skip_blank_lines=False)
    x_df = x_df.replace(r"^\s*$", np.nan, regex=True).apply(pd.to_numeric, args=("coerce",))

    x_data = x_df.astype(np.float32).values
    x_rows_del = []
    if autoremove_na:
        if np.isnan(x_data).any():
            x_rows_del, _ = np.where(np.isnan(x_data))
            print("Missing X:", x_rows_del)
            x_data = np.delete(x_data, x_rows_del, axis=0)

    if len(x_data.shape) != 2 or len(x_data) == 0:
        raise WrongFormatError(x_data, None)

    y_data = None
    if y_fname is None:
        y_data = x_data[:, y_cols]
        x_data = np.delete(x_data, y_cols, axis=1)
    else:
        y_df = pd.read_csv(y_fname, sep=sep, header=y_hdr, index_col=y_index_col, skip_blank_lines=False)
        y_df = y_df.replace(r"^\s*$", np.nan, regex=True).apply(pd.to_numeric, args=("coerce",))
        y_data = y_df.astype(np.float32).values
        if autoremove_na:
            if len(x_rows_del) > 0:
                y_data = np.delete(y_data, x_rows_del, axis=0)

            if np.isnan(y_data).any():
                y_rows_del, _ = np.where(np.isnan(y_data))
                # print("Missing Y:", y_rows_del)

                # print("NULLL", np.where(np.isnull(y_data)))
                y_data = np.delete(y_data, y_rows_del, axis=0)
                x_data = np.delete(x_data, y_rows_del, axis=0)

        if len(y_data.shape) != 2:
            raise WrongFormatError(x_data, y_data)

        if y_cols != -1:
            y_data = y_data[:, y_cols]

    if len(x_data) != len(y_data):
        raise WrongFormatError(x_data, y_data)

    return x_data, y_data.reshape(-1, 1)


def load_data(path, resampling=None, resample_size=0):
    print("(", resampling, resample_size, ")", end=" ")
    projdir = Path(path)
    files = tuple(next(projdir.glob(n)) for n in ["*Xcal*", "*Ycal*"])
    # X_train, y_train = load_csv(files[0], files[1], x_hdr=0, y_hdr=0, sep=";")
    X_train, y_train = load_csv(files[0], files[1], x_hdr=None, y_hdr=None, sep=";")

    if resampling == "crop":
        X_train = X_train[:, :resample_size]
    elif resampling == "resample":
        X_train_rs = []
        for i in range(len(X_train)):
            X_train_rs.append(signal.resample(X_train[i], resample_size))
        X_train = np.array(X_train_rs)

    X_valid, y_valid = np.empty(X_train.shape), np.empty(y_train.shape)
    regex = re.compile(".*Xval.*")
    for file in os.listdir(path):
        if regex.match(file):
            files = tuple(next(projdir.glob(n)) for n in ["*Xval*", "*Yval*"])
            X_valid, y_valid = load_csv(files[0], files[1], x_hdr=0, y_hdr=0, sep=";")

            if resampling == "crop":
                X_valid = X_valid[:, :resample_size]
            elif resampling == "resample":
                X_valid_rs = []
                for i in range(len(X_valid)):
                    X_valid_rs.append(signal.resample(X_valid[i], resample_size))
                X_valid = np.array(X_valid_rs)

    # X_train, X_valid = X_train[:,0:1024], X_valid[:,0:1024]

    return X_train, y_train, X_valid, y_valid


def load_multiple_data(path, resampling=None, resample_size=0):
    print("(", resampling, resample_size, ")", end=" ")
    projdir = Path(path)
    files = tuple(next(projdir.glob(n)) for n in ["*Xcal*", "*Ycal*"])
    # X_train, y_train = load_csv(files[0], files[1], x_hdr=0, y_hdr=0, sep=";")
    X_train, y_train = load_csv(files[0], files[1], x_hdr=None, y_hdr=None, sep=";")

    if resampling == "crop":
        X_train = X_train[:, :resample_size]
    elif resampling == "resample":
        X_train_rs = []
        for i in range(len(X_train)):
            X_train_rs.append(signal.resample(X_train[i], resample_size))
        X_train = np.array(X_train_rs)

    X_valid, y_valid = np.empty(X_train.shape), np.empty(y_train.shape)
    regex = re.compile(".*Xval.*")
    for file in os.listdir(path):
        if regex.match(file):
            files = tuple(next(projdir.glob(n)) for n in ["*Xval*", "*Yval*"])
            X_valid, y_valid = load_csv(files[0], files[1], x_hdr=0, y_hdr=0, sep=";")

            if resampling == "crop":
                X_valid = X_valid[:, :resample_size]
            elif resampling == "resample":
                X_valid_rs = []
                for i in range(len(X_valid)):
                    X_valid_rs.append(signal.resample(X_valid[i], resample_size))
                X_valid = np.array(X_valid_rs)

    # X_train, X_valid = X_train[:,0:1024], X_valid[:,0:1024]

    return X_train, y_train, X_valid, y_valid
