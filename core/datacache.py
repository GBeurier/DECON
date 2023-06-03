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


def hash_pipeline(pipeline):  # TODO test the type of Mixin (str, class, function, instance) and provide a identical hash for each
    """
    The function takes a pipeline as input, converts it to a string, hashes it, takes the absolute
    value, and returns the hexadecimal representation of the hash without the '0x' prefix.

    :param pipeline: The parameter "pipeline" is a variable that is expected to contain a pipeline
    object or a string representation of a pipeline object. The function "hash_pipeline" takes this
    parameter and returns a hexadecimal representation of the hash value of the string representation of
    the pipeline object
    :return: The function `hash_pipeline` takes a `pipeline` argument and returns a hexadecimal string
    representation of the absolute hash value of the string representation of the `pipeline` argument.
    The `[2:]` at the end of the return statement is used to remove the first two characters of the
    hexadecimal string, which are always '0x'.
    """
    return hex(abs(hash(str(pipeline))))[2:]


def get_properties(file_path):
    """
    This function reads the first two lines of a file and determines if there is a header and the
    delimiter used in the file.

    :param file_path: The path to the file that needs to be analyzed. It can be a regular file or a
    compressed file with a ".gz" extension
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
    This function loads a CSV file, cleans and processes the data, and returns it as a numpy array along
    with a list of rows that contain missing values.

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
    This function registers a dataset by loading files, removing rows with missing values, and checking
    if the X and y data are consistent.

    :param dataset_config: The configuration of the dataset to be registered, which can be either a
    string representing the path to the dataset directory or a tuple containing the path to the dataset
    directory and the indices of the columns to be used as the target variable
    :return: a tuple containing the cache hash, dataset name, and cache.
    """

    logging.info("Registering dataset: %s", dataset_config)

    x_files_re = [("X_train", "*Xcal*"), ("X_test", "*Xtest*"), ("X_val", "*Xval*")]
    y_files_re = [("y_train", "*ycal*"), ("y_test", "*ytest*"), ("y_val", "*yval*")]

    dataset_dir = None
    if isinstance(dataset_config, str):
        dataset_dir = Path(dataset_config)
        assert dataset_dir.is_dir(), "dataset_dir must be a directory but is of type %s: %s" % (type(dataset_dir), dataset_dir)
    elif isinstance(dataset_config, tuple):
        path, y_cols = dataset_config
        dataset_dir = Path(path)
        assert isinstance(y_cols, list), "y_cols must be a list but is of type %s: %s" % (type(y_cols), y_cols)
        y_cols = [y_cols]
        if dataset_dir.is_file():
            x_files_re = [("X_train", str(dataset_dir))]

    if not dataset_dir.exists():
        logging.error("Path does not exist: %s", path)
        return None, None, None

    dataset_name = dataset_dir.name

    # Load Files
    cache = {"X_train": None, "X_test": None, "X_val": None, "y_train": None, "y_test": None, "y_val": None}
    removed_rows = {"X_train": [], "X_test": [], "X_val": [], "y_train": [], "y_test": [], "y_val": []}

    for key, regex in x_files_re:
        files = list(dataset_dir.glob(regex))
        csv_list = [load_csv(f) for f in files if f.is_file()]
        cache[key] = [x[0] for x in csv_list]
        removed_rows[key] = [x[1] for x in csv_list]

    if not isinstance(dataset_config, tuple):
        for key, regex in y_files_re:
            files = list(dataset_dir.glob(regex))

            if len(files) == 0:
                logging.warning("No y file %s - %s found for %s.", key, regex, dataset_name)
                continue
            assert len(files) == 1, "Cannot initialize %s %s dataset. More than one (%s) file found. Multiple y should be available in the next version." % (dataset_name, key, regex)

            csv_list = [load_csv(f) for f in files if f.is_file()]
            cache[key], removed_rows[key] = csv_list[0]

    logging.info("cache state: %s", json.dumps({k: np.array(v).shape if v is not None else None for k, v in cache.items()}))

    # Reconstruct nested Y data ###
    if isinstance(dataset_config, tuple):
        assert len(cache["X_train"]) == 1, "Cannot initialize %s %s columns. More than one X_train file found." % (y_cols, dataset_name)
        logging.info("Getting y cols %s from X for %s.", y_cols, dataset_name)

        for i, (y_key, _) in enumerate(y_files_re):
            x_key = x_files_re[i][0]
            if len(cache[x_key]) > 0:
                cache[y_key] = cache[x_key][0][:, y_cols]
                cache[x_key] = [np.delete(cache[x_key][0], y_cols, axis=1)]
            else:
                logging.warning("Unable to init %s %s, no %s data found.", dataset_name, y_key, x_key)

    logging.info("cache state: %s", json.dumps({k: np.array(v).shape if v is not None else None for k, v in cache.items()}))

    # Remove rows with missing values
    for i, (y_key, _) in enumerate(y_files_re):
        x_key = x_files_re[i][0]
        x_removed_rows = [removed_rows[x_key][j] for j in range(len(removed_rows[x_key])) if len(removed_rows[x_key][j]) > 0]
        y_removed_rows = [removed_rows[y_key][j] for j in range(len(removed_rows[y_key])) if len(removed_rows[y_key][j]) > 0]

        logging.info("Probing if rows with missing values from %s and %s should be removed.", x_key, y_key)
        logging.info("x_removed_rows: %s", x_removed_rows)
        logging.info("y_removed_rows: %s", y_removed_rows)

        if len(x_removed_rows) > 0 or len(y_removed_rows) > 0:
            indexes_to_remove = np.array(reduce(np.union1d, (x_removed_rows + y_removed_rows))).astype(np.int32)

            logging.info("cache state: %s", json.dumps({k: np.array(v).shape if v is not None else None for k, v in cache.items()}))
            logging.warning("Removing rows: %s in %s %s", indexes_to_remove, dataset_name, x_key)

            for k, dataset in enumerate(cache[x_key]):
                cache[x_key][k] = list(np.delete(np.array(dataset), indexes_to_remove, axis=0))
            cache[y_key] = np.delete(cache[y_key], indexes_to_remove, axis=0)

    logging.info("cache state: %s", json.dumps({k: np.array(v).shape if v is not None else None for k, v in cache.items()}))

    # check if cache X and y are consistent
    for i, (y_key, pattern) in enumerate(y_files_re):
        x_key = x_files_re[i][0]
        y_dataset = cache[y_key]
        cache[y_key] = np.array(cache[y_key])
        cache[x_key] = np.array(cache[x_key])

        for dataset in cache[x_key]:
            if y_dataset is None:
                assert dataset is None, "X and y are not consistent for %s. %s is None but not %s" % (dataset_name, y_key, x_key)
            else:
                assert dataset is not None, "X and y are not consistent for %s. %s is not None but %s is" % (dataset_name, y_key, x_key)
                assert dataset.shape[0] == y_dataset.shape[0], "X and y are not consistent for %s. %s has %s rows but %s has %s." % (
                    dataset_name, x_key, dataset.shape[0], y_key, y_dataset.shape[0])

    cache_hash = data_hash([cache[key] for key in sorted(cache.keys())])
    cache["path"] = str(dataset_dir)
    cache["origin"] = None

    DATACACHE[dataset_name] = {}
    DATACACHE[dataset_name][cache_hash] = cache

    return cache_hash, dataset_name


def get_data_from_uid(dataset, uid):
    """
    This function retrieves data from a cache based on a given dataset and unique identifier.

    :param dataset: a string representing the name of a dataset that is stored in a global variable
    called DATACACHE
    :param uid: uid stands for "user ID" and is a unique identifier for a specific user in a dataset.
    The function "get_data_from_uid" takes in a dataset and a uid as parameters and returns the data
    associated with that uid in the dataset
    :return: data associated with a specific uid from a dataset stored in a global variable called
    DATACACHE.
    """
    assert dataset in DATACACHE, "Unknown dataset %s" % dataset
    assert uid in DATACACHE[dataset], "Unknown uid %s for dataset %s" % (uid, dataset)
    return DATACACHE[dataset][uid]


def fake_transform(p, X):  # TODO: replace by a call to the pipeliner
    logging.info("Fake transform %s", p)
    return X


def get_data(dataset, from_uid=None, previous_pipelines=None, pipeline=None):
    """
    This function retrieves data from a cache based on a given dataset, previous pipelines, and a
    current pipeline.

    :param dataset: a string representing the name of the dataset to retrieve data from. It should be a
    key in the DATACACHE dictionary
    :param from_uid: The unique identifier of the data to start the pipeline from. If None, the function
    will look for data with no origin in the cache and use that as the starting point
    :param previous_pipelines: A list of pipelines that have already been applied to the dataset. Each
    pipeline is represented as a dictionary of transformation functions and their parameters
    :param pipeline: A pipeline is a sequence of data processing steps. In this function, it refers to a
    specific pipeline that is applied to the data in the cache. If a pipeline is provided, it will be
    applied to the data before returning it
    :return: the data corresponding to the specified dataset, after applying any previous pipelines and
    the current pipeline (if provided).
    """
    assert dataset in DATACACHE, "Unknown dataset %s" % dataset

    cache = DATACACHE[dataset]
    if from_uid is None:
        for k, v in cache.items():
            if v["origin"] is None:
                from_uid = k
                break

    assert from_uid is not None, "No data found for dataset %s" % dataset

    if previous_pipelines is not None:
        for p in previous_pipelines:
            if p is None:
                continue

            from_uid += hash_pipeline(p)
            if from_uid not in cache:
                # instanciate pipeline p
                new_data = fake_transform(p, cache[from_uid])
                cache[from_uid] = new_data

    assert from_uid is not None, "Failed to apply previous pipelines" % previous_pipelines

    if pipeline is not None:
        from_uid += hash_pipeline(pipeline)
        if from_uid not in cache:
            # instanciate pipeline pipeline
            new_data = fake_transform(pipeline, cache[from_uid])
            cache[from_uid] = new_data

    return cache[from_uid]


def clear(dataset=None):
    if dataset is None:
        DATACACHE.clear()
    else:
        DATACACHE[dataset].clear()


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
