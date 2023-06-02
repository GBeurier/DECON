import csv
from functools import reduce
import gzip
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import traceback
import re


DATACACHE = {}


def data_hash(data):
    """
    The function generates a hash value for a given data input, which can be a list, tuple, or numpy
    array.
    
    :param data: The input data that needs to be hashed. It can be of any data type such as string,
    integer, float, list, tuple, or numpy array
    :return: The function `data_hash` takes in a data object and returns a hexadecimal string
    representing the hash value of the object. If the object is a list or tuple, it recursively hashes
    each element and concatenates the results before hashing the concatenated string. If the object is a
    numpy array, it hashes the string representation of the array. For all other objects, it hashes the
    string representation of the object.
    """
    if isinstance(data, list) or isinstance(data, tuple):
        return hex(abs(hash("".join([data_hash(x) for x in data]))))[2:]
    elif isinstance(data, np.ndarray):
        return hex(abs(hash(str(data))))[2:]

    return hex(abs(hash(str(data))))[2:]


def get_properties(file_path):
    """
    This function reads the first two lines of a file and determines if there is a header and the
    delimiter used in the file.
    
    :param file_path: The path to the file that we want to extract properties from
    :return: a tuple containing two values: a boolean value indicating whether the file has a header row
    or not, and a string value indicating the delimiter used in the file.
    """
    file = None
    if str(file_path).endswith(".gz"):
        file = gzip.open(file_path, 'rb')
    else:
        file = open(file_path, 'rb')

    first_line = file.readline().decode('utf-8')
    second_line = file.readline().decode('utf-8')
    file.close()

    header = False
    if re.search(r'[a-df-zA-DF-Z]', first_line):
        header = True
    if "." not in first_line:
        header = True
    
    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(second_line)
    # logging.info(dialect.delimiter)
    return header, dialect.delimiter


def load_csv(file_path):
    """
    This function loads a CSV file, infers its properties, corrects the delimiter if necessary, reads
    the file into a pandas dataframe, cleans the data by dropping empty columns and rows, removes rows
    with missing values, and returns the cleaned data as a numpy array along with a list of rows that
    were removed.
    
    :param file_path: The file path of the CSV file to be loaded
    :return: a tuple containing the loaded data as a numpy array and a list of row indices that contain
    missing values.
    """
    logging.info("Loading file: %s", file_path)
    data = None
    header, delimiter = get_properties(file_path)
    logging.info("Header inference: %s", header)
    logging.info("Delimiter inference: %s", delimiter)
    if re.search(r"[0-9\.]", delimiter):
        delimiter = "\n"
        logging.warning("Delimiter correction: \\n")
    try:
        if header:
            data = pd.read_csv(file_path, header=0, na_filter=False, sep=delimiter, engine='python')
        else:
            data = pd.read_csv(file_path, header=None, na_filter=False, sep=delimiter, engine='python')
    except Exception as e:
        logging.error("Error loading file: %s", file_path)
        logging.error("Exception %s", e)
        logging.error(traceback.format_exc())
        return None

    data = data.replace(r"^\s*$", "", regex=True)
    data = data.apply(pd.to_numeric, args=("coerce",))

    logging.info("Data shape: %s", data.shape)
    data = data.dropna(how='all', axis=1)
    logging.info("Data shape after dropna(all) cols: %s", data.shape)
    data = data.dropna(how='all', axis=0)
    logging.info("Data shape after dropna(all) rows: %s", data.shape)

    na_coords = np.argwhere(pd.isna(data).values)
    na_row_list = []
    if len(na_coords) > 0:
        na_row_list = list(set(na_coords[:, 0]))
        logging.info("Rows to remove after filtering: %s", na_row_list)
        # data = data.dropna(how='any', axis=0)
        # logging.info("Data shape after dropna(any) rows:", data.shape)

    data = data.astype(np.float32).values
    logging.info("data_sample:\n%s", data[:2, :2])

    return data, na_row_list


def register_dataset(dataset_config):
    """
    This function registers a dataset by loading and processing its X and y files, removing rows with
    missing values, and storing the resulting data in a cache.
    
    :param dataset_config: The configuration of the dataset to be registered. It can be either a string
    representing the path to the dataset directory or a tuple containing the path to the dataset
    directory and the index of the column containing the target variable
    :return: a tuple containing the cache hash, dataset name, and cache.
    """
    logging.info("Registering dataset: %s", dataset_config)
    y_files = [("y_train", "*ycal*"), ("y_test", "*ytest*"), ("y_val", "*yval*")]
    X_files = [("X_train", "*Xcal*"), ("X_test", "*Xtest*"), ("X_val", "*Xval*")]
    files = X_files.copy()

    dataset_path = ""
    if isinstance(dataset_config, str):
        files.extend(y_files)
        dataset_path = dataset_config
    elif isinstance(dataset_config, tuple):
        dataset_path, y_cols = dataset_config
        if isinstance(y_cols, int):
            y_cols = [y_cols]

    dataset_dir = Path(dataset_path)
    dataset_name = dataset_dir.name

    cache = {
        "X_train": None,
        "X_test": None,
        "X_val": None,
        "y_train": None,
        "y_test": None,
        "y_val": None
    }

    removed_rows = {
        "X_train": [],
        "X_test": [],
        "X_val": [],
        "y_train": [],
        "y_test": [],
        "y_val": []
    }

    for key, pattern in files:
        files = list(dataset_dir.glob(pattern))
        csv_list = [load_csv(f) for f in files if f.is_file()]
        cache[key] = [x[0] for x in csv_list]
        removed_rows[key] = [x[1] for x in csv_list]
        
    logging.info("cache state: %s", json.dumps({k: np.array(v).shape if v is not None else None for k, v in cache.items()}))

    # Get y from a col of X
    if isinstance(dataset_config, tuple):
        assert len(cache["X_train"]) == 1, "Cannot initialize y columns (%s). More than one X_train file found for %s." % (y_cols, dataset_name)
        
        logging.info("Getting y cols %s from X for %s.", y_cols, dataset_name)
        for i, (y_key, pattern) in enumerate(y_files):
            x_key = X_files[i][0]
            if len(cache[x_key])  == 0:
                logging.warning("Unable to init %s, no %s data found for %s.", y_key, x_key, dataset_name)
            else:
                cache[y_key] = cache[x_key][0][:, y_cols]
                cache[x_key] = [np.delete(cache[x_key][0], y_cols, axis=1)]
                logging.info("%s, %s shapes: %s, %s", x_key, y_key, np.array(cache[x_key]).shape, np.array(cache[y_key]).shape)
            
    logging.info("cache state: %s", json.dumps({k: np.array(v).shape if v is not None else None for k, v in cache.items()}))
    
    # Remove rows with missing values
    for i, (y_key, pattern) in enumerate(y_files):
        x_key = X_files[i][0]
        
        logging.info("Probing if rows with missing values from %s and %s should be removed.", x_key, y_key)
        x_removed_rows = [removed_rows[x_key][j] for j in range(len(removed_rows[x_key])) if len(removed_rows[x_key][j]) > 0]
        logging.info("x_removed_rows: %s", x_removed_rows)
        y_removed_rows = [removed_rows[y_key][j] for j in range(len(removed_rows[y_key])) if len(removed_rows[y_key][j]) > 0]
        logging.info("y_removed_rows: %s", y_removed_rows)
        
        if len(x_removed_rows) > 0 or len(y_removed_rows) > 0:
            indexes_to_remove = np.array(reduce(np.union1d, (x_removed_rows + y_removed_rows))).astype(np.int32)
            logging.info("indexes_to_remove: %s", indexes_to_remove)
            
            logging.info("cache state: %s", json.dumps({k: np.array(v).shape if v is not None else None for k, v in cache.items()}))
            logging.warning("Removing rows: %s", str(indexes_to_remove))
            new_cache = []
            for k, dataset in enumerate(cache[x_key]):
                cache[x_key][k] = list(np.delete(np.array(dataset), indexes_to_remove, axis=0))
            cache[y_key][0] = np.delete(cache[y_key][0], indexes_to_remove, axis=0)


    logging.info("cache state: %s", json.dumps({k: np.array(v).shape if v is not None else None for k, v in cache.items()}))
    # check if cache X and y are consistent
    for i, (y_key, pattern) in enumerate(y_files):
        x_key = X_files[i][0]
        y_dataset = cache[y_key]
        cache[y_key] = np.array(cache[y_key])
        cache[x_key] = np.array(cache[x_key])
        
        for dataset in cache[x_key]:
            if y_dataset is None:
                assert dataset is None, "X and y are not consistent for %s. %s is None but not %s" % (dataset_name, y_key, x_key)
            else:
                assert dataset is not None, "X and y are not consistent for %s. %s is not None but %s is" % (dataset_name, y_key, x_key)
                assert dataset.shape[0] == y_dataset.shape[0], "X and y are not consistent for %s. %s has %s rows but %s has %s." % (dataset_name, x_key, dataset.shape[0], y_key, y_dataset.shape[0])

    cache_hash = data_hash([cache[key] for key in sorted(cache.keys())])
    cache["path"] = dataset_path

    DATACACHE[dataset_name] = {}
    DATACACHE[dataset_name][cache_hash] = cache

    return cache_hash, dataset_name, cache


# def get(self, dataset, *, indexes=None):
#     return self._cache.get(dataset)

# def set(self, dataset, data):
#     self._cache[dataset] = data

# def clear(self):
#     self._cache.clear()

# def has(self, dataset):
#     return dataset in self._cache

# def free(self, uid):
#     pass







# 
# import numpy as np
# import pandas as pd
# import os
# import re
# from scipy import signal

# # from pinard.utils import load_csv


# class WrongFormatError(Exception):
#     """Exception raised when X et Y datasets are invalid."""

#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#         msg = ""
#         if type(x) is np.ndarray:
#             msg += "Invalid X shape: {}".format(x.shape) + " "
#         if type(y) is np.ndarray:
#             msg += "Invalid Y shape: {}".format(y.shape)
#         super().__init__(msg)


# def load_csv(x_fname, y_fname=None, y_cols=0, *, sep=None, x_hdr=None, y_hdr=None, x_index_col=None, y_index_col=None, autoremove_na=True):
#     assert y_fname is not None or y_cols is not None
#     # TODO - add assert/exceptions on non-numerical columns
#     # TODO - better management of NaN and Null (esp exception msg)

#     x_df = pd.read_csv(x_fname, sep=sep, header=x_hdr, index_col=x_index_col, skip_blank_lines=False)
    

#     x_data = x_df.astype(np.float32).values
#     x_rows_del = []
#     if autoremove_na:
#         if np.isnan(x_data).any():
#             x_rows_del, _ = np.where(np.isnan(x_data))
#             logging.info("Missing X:", x_rows_del)
#             x_data = np.delete(x_data, x_rows_del, axis=0)

#     if len(x_data.shape) != 2 or len(x_data) == 0:
#         raise WrongFormatError(x_data, None)

#     y_data = None
#     if y_fname is None:
#         y_data = x_data[:, y_cols]
#         x_data = np.delete(x_data, y_cols, axis=1)
#     else:
#         y_df = pd.read_csv(y_fname, sep=sep, header=y_hdr, index_col=y_index_col, skip_blank_lines=False)
#         y_df = y_df.replace(r"^\s*$", np.nan, regex=True).apply(pd.to_numeric, args=("coerce",))
#         y_data = y_df.astype(np.float32).values
#         if autoremove_na:
#             if len(x_rows_del) > 0:
#                 y_data = np.delete(y_data, x_rows_del, axis=0)

#             if np.isnan(y_data).any():
#                 y_rows_del, _ = np.where(np.isnan(y_data))
#                 # logging.info("Missing Y:", y_rows_del)

#                 # logging.info("NULLL", np.where(np.isnull(y_data)))
#                 y_data = np.delete(y_data, y_rows_del, axis=0)
#                 x_data = np.delete(x_data, y_rows_del, axis=0)

#         if len(y_data.shape) != 2:
#             raise WrongFormatError(x_data, y_data)

#         if y_cols != -1:
#             y_data = y_data[:, y_cols]

#     if len(x_data) != len(y_data):
#         raise WrongFormatError(x_data, y_data)

#     return x_data, y_data.reshape(-1,1)


# def load_csv_multiple(x_fname, y_fname=None, y_cols=0, *, sep=None, x_hdr=None, y_hdr=None, x_index_col=None, y_index_col=None, autoremove_na=True):
#     assert y_fname is not None or y_cols is not None
#     # TODO - add assert/exceptions on non-numerical columns
#     # TODO - better management of NaN and Null (esp exception msg)


#     x_data = []
#     for i in range(len(x_fname)):
#         x_df = pd.read_csv(x_fname[i], sep=sep, header=x_hdr, index_col=x_index_col, skip_blank_lines=False)
#         x_df = x_df.replace(r"^\s*$", np.nan, regex=True).apply(pd.to_numeric, args=("coerce",))
#         x_data.append(x_df.astype(np.float32).values)
#     x_data = np.array(x_data)
#     logging.info(x_data.shape)

#     x_rows_del = []
#     if autoremove_na:
#         for i in range(len(x_data)):
#             if np.isnan(x_data[i]).any():
#                 x_del, _ = np.where(np.isnan(x_data[i]))
#                 x_rows_del = np.union1d(x_rows_del, x_del)
#                 logging.info("Missing X:", x_rows_del)
#         x_data = np.delete(x_data, x_rows_del, axis=1)

#     if len(x_rows_del) > 0:
#         logging.info("X rows deleted: ", x_rows_del)

#     if len(x_data.shape) != 3 or len(x_data[0]) == 0:
#         raise WrongFormatError(x_data, None)

#     y_data = None
#     if y_fname is None:
#         y_data = x_data[:, y_cols]
#         x_data = np.delete(x_data, y_cols, axis=2)
#     else:
#         y_df = pd.read_csv(y_fname, sep=sep, header=y_hdr, index_col=y_index_col, skip_blank_lines=False)
#         y_df = y_df.replace(r"^\s*$", np.nan, regex=True).apply(pd.to_numeric, args=("coerce",))
#         y_data = y_df.astype(np.float32).values
#         if autoremove_na:
#             if len(x_rows_del) > 0:
#                 y_data = np.delete(y_data, x_rows_del, axis=0)

#             if np.isnan(y_data).any():
#                 y_rows_del, _ = np.where(np.isnan(y_data))
#                 y_data = np.delete(y_data, y_rows_del, axis=0)
#                 x_data = np.delete(x_data, y_rows_del, axis=1)
#                 if len(y_rows_del) > 0:
#                     logging.info("Y rows deleted: ", y_rows_del)

#         if len(y_data.shape) != 2:
#             raise WrongFormatError(x_data, y_data)

#         if y_cols != -1:
#             y_data = y_data[:, y_cols]

#     if len(x_data[0]) != len(y_data):
#         raise WrongFormatError(x_data, y_data)

#     return x_data, y_data.reshape(-1,1)




# def load_data(path, resampling=None, resample_size=0):

#     if resampling is not None:
#         logging.info('(', resampling, resample_size, ')', end=" ")

#     projdir = Path(path)
#     files = tuple(next(projdir.glob(n)) for n in ["*Xcal*", "*Ycal*"])
#     X_train, y_train = load_csv(files[0], files[1], x_hdr=None, y_hdr=None, sep=";")

#     if resampling == "crop":
#         X_train = X_train[:, :resample_size]
#     elif resampling == "resample":
#         X_train_rs = []
#         for i in range(len(X_train)):
#             X_train_rs.append(signal.resample(X_train[i], resample_size))
#         X_train = np.array(X_train_rs)

#     X_valid, y_valid = np.empty(X_train.shape), np.empty(y_train.shape)
#     regex = re.compile(".*Xval.*")
#     for file in os.listdir(path):
#         if regex.match(file):
#             files = tuple(next(projdir.glob(n)) for n in ["*Xval*", "*Yval*"])
#             X_valid, y_valid = load_csv(files[0], files[1], x_hdr=0, y_hdr=0, sep=";")

#             if resampling == "crop":
#                 X_valid = X_valid[:, :resample_size]
#             elif resampling == "resample":
#                 X_valid_rs = []
#                 for i in range(len(X_valid)):
#                     X_valid_rs.append(signal.resample(X_valid[i], resample_size))
#                 X_valid = np.array(X_valid_rs)

#     # X_train, X_valid = X_train[:,0:1024], X_valid[:,0:1024]
    
#     return X_train, y_train, X_valid, y_valid


# def load_data_multiple(path, resampling=None, resample_size=0):

#     if resampling is not None:
#         logging.info('(', resampling, resample_size, ')', end=" ")

#     projdir = Path(path)
    
#     x_files = []
#     for x_file in projdir.glob("*Xcal*"):
#         x_files.append(x_file)
#     y_file = next(projdir.glob("*Ycal*"))
#     X_train, y_train = load_csv_multiple(x_files, y_file, x_hdr=None, y_hdr=None, sep=";")

#     if resampling == "crop":
#         X_train = X_train[:, :, :resample_size]
#     elif resampling == "resample":
#         new_X_train = []
#         for j in range(len(X_train)):
#             X_train_rs = []
#             for i in range(len(X_train[j])):
#                 X_train_rs.append(signal.resample(X_train[j][i], resample_size))
#             new_X_train.append(X_train_rs)
#         X_train = np.array(new_X_train)
        
#     X_valid, y_valid = np.empty(X_train.shape), np.empty(y_train.shape)
#     regex = re.compile(".*Xval.*")
#     for file in os.listdir(path):
#         if regex.match(file):
#             x_files = []
#             for x_file in projdir.glob("*Xval*"):
#                 x_files.append(x_file)
#             y_file = next(projdir.glob("*Yval*"))
#             X_valid, y_valid = load_csv_multiple(x_files, y_file, x_hdr=None, y_hdr=None, sep=";")

#             if resampling == "crop":
#                 X_valid = X_valid[:, :, :resample_size]
#             elif resampling == "resample":
#                 new_X_valid = []
#                 for j in range(len(X_valid)):
#                     X_valid_rs = []
#                     for i in range(len(X_valid[j])):
#                         X_valid_rs.append(signal.resample(X_valid[j][i], resample_size))
#                     new_X_valid.append(X_valid_rs)
#                 X_valid = np.array(new_X_valid)
                
#     return X_train, y_train, X_valid, y_valid
