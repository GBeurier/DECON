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

import core.pipeliner as pipeliner

DATACACHE = {}  # TODO convert datacache and cache to dataclass


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

    header = False
    if re.search(r'[a-df-zA-DF-Z]', first_line):
        header = True
    # if "." not in first_line:
    #     header = True

    sniffer = csv.Sniffer()
    line_to_sniff = first_line
    while re.search(r'[a-df-zA-DF-Z]', line_to_sniff) is not None or not line_to_sniff.strip() or not re.search(r'[0-9]', line_to_sniff):
        line_to_sniff = file.readline().decode('utf-8')
    dialect = sniffer.sniff(line_to_sniff)
    # print(dialect.delimiter)

    if header is False:
        header_test = first_line.split(dialect.delimiter)
        if len(header_test) > 50:
            s = 0
            for v in header_test:
                if v.strip() == "":
                    continue
                try:
                    v = float(v)
                except Exception as e:
                    logging.error("Error while converting to float: %s", v)
                    logging.error(e)
                    logging.error(traceback.format_exc())
                if v < s:
                    header = False
                    break
                s = v
                header = True
    # print("header is", header)

    line_to_sniff = second_line
    while re.search(r'[a-df-zA-DF-Z]', line_to_sniff) is not None or not line_to_sniff.strip() or not re.search(r'[0-9]', line_to_sniff):
        line_to_sniff = file.readline().decode('utf-8')

    number_delimiter = "."
    if "," in line_to_sniff and dialect.delimiter != ",":
        number_delimiter = ","
    delimiter = dialect.delimiter
    logging.info("Header inference: %s", header)
    logging.info("Delimiter inference: %s", delimiter)
    logging.info("Number delimiter inference: %s", number_delimiter)
    if re.search(r"[0-9\.]", delimiter):
        delimiter = " "
        logging.warning("Delimiter correction for %s: \\n", file_path)

    file.close()
    return header, delimiter, number_delimiter


def load_csv(file_path):
    """
    This function loads a CSV file, cleans and processes the data, and returns it as a numpy array along
    with a list of rows that contain missing values.

    :param file_path: The file path of the CSV file to be loaded
    :return: a tuple containing the loaded data as a numpy array and a list of row indices that contain
    missing values.
    """
    logging.info("Loading file: %s", file_path)
    # print("loading file", file_path)
    data = None
    header, delimiter, nb_delimiter = get_properties(file_path)

    try:
        if header:
            data = pd.read_csv(file_path, header=0, na_filter=False, sep=delimiter, engine='python', skip_blank_lines=False, decimal=nb_delimiter)
        else:
            data = pd.read_csv(file_path, header=None, na_filter=False, sep=delimiter, engine='python', skip_blank_lines=False, decimal=nb_delimiter)
    except Exception as e:
        logging.error("Error loading file: %s", file_path)
        logging.error("Exception %s", e)
        logging.error(traceback.format_exc())
        return None

    logging.info("Data shape after read_csv: %s", data.shape)
    data = data.replace(r"^\s*$", "", regex=True)
    data = data.apply(pd.to_numeric, args=("coerce",))

    logging.info("Data shape: %s", data.shape)
    data = data.dropna(how='all', axis=1)
    logging.info("Data shape after dropna(all) cols: %s", data.shape)
    # data = data.dropna(how='all', axis=0)
    logging.info("Data shape after dropna(all) rows: %s", data.shape)

    na_coords = np.argwhere(pd.isna(data).values)
    # logging.info("NA coordinates: %s", na_coords)
    na_row_list = []
    if len(na_coords) > 0:
        na_row_list = list(set(na_coords[:, 0]))
        logging.info("Rows to remove after filtering: %s", na_row_list)
        # data = data.dropna(how='any', axis=0)
        # logging.info("Data shape after dropna(any) rows:", data.shape)

    data = data.astype(np.float32).values
    logging.info("data_sample:\n%s", data[:2, :2])

    return data, na_row_list


def _prepare_dataset(dataset_config):
    """
    This function prepares the dataset for loading by setting up file patterns and checking the validity
    of the dataset directory.

    :param dataset_config: The configuration for the dataset, which can be either a string representing
    the path to the dataset directory or a tuple containing the path to the dataset directory and a list
    of column names for the target variable(s)
    :return: a tuple containing dataset_dir, x_files_re, y_files_re, and y_cols.
    """
    # Prepare loading
    x_files_re = [("X_train", "*Xcal*"), ("X_test", "*Xtest*"), ("X_val", "*Xval*")]
    y_files_re = [("y_train", "*ycal*"), ("y_test", "*ytest*"), ("y_val", "*yval*")]
    y_cols = None
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

    return dataset_dir, x_files_re, y_files_re, y_cols


def _load_dataset(dataset_dir, x_files_re, y_files_re, dataset_config, dataset_name):
    """
    This function loads files from a given directory based on specified regular expressions and returns
    a cache and removed rows.

    :param dataset_dir: The directory where the dataset files are stored
    :param x_files_re: A list of tuples where each tuple contains a key and a regular expression to
    match the X files in the dataset directory
    :param y_files_re: A list of tuples containing the regular expression for matching the y files and
    the corresponding key to store the loaded data in the cache dictionary
    :param dataset_config: The dataset configuration, which is expected to be a tuple. If it is not a
    tuple, then the function assumes that there is only one y file and loads it accordingly
    :param dataset_name: The name of the dataset being loaded
    :return: a tuple containing two dictionaries: `cache` and `removed_rows`.
    """
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
            # cache[key] = [x[0] for x in csv_list]
            # removed_rows[key] = [x[1] for x in csv_list]
            cache[key], removed_rows[key] = csv_list[0]

    return cache, removed_rows


def _build_y_data(cache, x_files_re, y_files_re, y_cols, dataset_name):
    """
    This function extracts specific columns from X_train to use as y data for a given dataset.

    :param cache: A dictionary containing cached data
    :param x_files_re: A regular expression pattern used to match the X data files in the cache
    :param y_files_re: A regular expression pattern used to match the names of the y data files
    :param y_cols: The columns to be extracted from the X data to form the y data
    :param dataset_name: The name of the dataset being processed
    """
    assert len(cache["X_train"]) == 1, "Cannot initialize %s %s columns. More than one X_train file found." % (y_cols, dataset_name)
    logging.info("Getting y cols %s from X for %s.", y_cols, dataset_name)

    for i, (y_key, _) in enumerate(y_files_re):
        x_key = x_files_re[i][0]
        if len(cache[x_key]) > 0:
            cache[y_key] = cache[x_key][0][:, y_cols]
            cache[x_key] = [np.delete(cache[x_key][0], y_cols, axis=1)]
        else:
            logging.warning("Unable to init %s %s, no %s data found.", dataset_name, y_key, x_key)


def _clean_dataset(cache, x_files_re, y_files_re, removed_rows, dataset_name):
    """
    This function removes rows with missing values from the cache for a given dataset.

    :param cache: A dictionary containing the datasets to be cleaned
    :param x_files_re: A regular expression pattern used to match the filenames of the X dataset files
    :param y_files_re: A regular expression pattern used to match the names of the files containing the
    target variables in a dataset
    :param removed_rows: A dictionary containing information about which rows have been removed from
    each dataset. The keys are the names of the datasets and the values are lists of lists, where each
    inner list contains the indexes of the rows that have been removed
    :param dataset_name: The name of the dataset being cleaned
    """
    for i, (y_key, _) in enumerate(y_files_re):
        x_key = x_files_re[i][0]
        logging.info("Removing rows with missing values from %s and %s.", x_key, y_key)
        logging.info("removed_rows: %s", removed_rows)
        # logging.info(removed_rows[x_key], removed_rows[y_key])
        x_removed_rows = [removed_rows[x_key][j] for j in range(len(removed_rows[x_key])) if len(removed_rows[x_key][j]) > 0]
        y_removed_rows = removed_rows[y_key] if len(removed_rows[y_key]) > 0 else []

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


def _validate_dataset(cache, x_files_re, y_files_re, dataset_name):
    """
    This function validates the consistency of X and y datasets in a cache for a given dataset name and
    regular expressions for X and y keys.

    :param cache: a dictionary containing the datasets to be validated
    :param x_files_re: a regular expression pattern for matching the keys of the input data files
    :param y_files_re: A regular expression pattern used to match the keys of the y datasets in the
    cache
    :param dataset_name: a string representing the name of the dataset being validated
    """
    for i, (y_key, _) in enumerate(y_files_re):
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
                    dataset_name, x_key, dataset.shape, y_key, y_dataset.shape)


def register_dataset(dataset_config):
    """
    This function registers a dataset by preparing, loading, cleaning, validating, and storing it in the
    DATACACHE.

    :param dataset_config: A dictionary or tuple containing configuration information for the dataset
    :return: a tuple containing the cache hash and the dataset name.
    """

    logging.info("Registering dataset: %s", dataset_config)
    dataset_dir, x_files_re, y_files_re, y_cols = _prepare_dataset(dataset_config)
    dataset_name = dataset_dir.name

    cache, removed_rows = _load_dataset(dataset_dir, x_files_re, y_files_re, dataset_config, dataset_name)
    logging.info("cache state: %s", json.dumps({k: np.array(v).shape if v is not None else None for k, v in cache.items()}))
    # print("cache state: %s", json.dumps({k: np.array(v).shape if v is not None else None for k, v in cache.items()}))

    # Reconstruct nested Y data ###
    if isinstance(dataset_config, tuple):
        _build_y_data(cache, x_files_re, y_files_re, y_cols, dataset_name)
        logging.info("cache state: %s", json.dumps({k: np.array(v).shape if v is not None else None for k, v in cache.items()}))

    # Remove rows with missing values
    _clean_dataset(cache, x_files_re, y_files_re, removed_rows, dataset_name)
    logging.info("cache state: %s", json.dumps({k: np.array(v).shape if v is not None else None for k, v in cache.items()}))

    # check if cache X and y are consistent
    _validate_dataset(cache, x_files_re, y_files_re, dataset_name)

    # Put data in DATACACHE
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


def _apply_pipeline(dataset, pipeline, from_uid):
    cache = DATACACHE[dataset]
    next_uid = from_uid + hash_pipeline(pipeline)
    if next_uid not in cache:
        new_data = pipeliner.apply_pipeline(pipeline, cache[from_uid])
        cache[next_uid] = new_data
        from_uid = next_uid

    return from_uid


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
            from_uid = _apply_pipeline(dataset, p, from_uid)
            assert from_uid is not None, "Failed to apply previous pipeline" % p

    if pipeline is not None:
        from_uid = _apply_pipeline(dataset, pipeline, from_uid)
        assert from_uid is not None, "Failed to apply pipeline" % pipeline

    return cache[from_uid]


def clear(dataset=None):
    """
    This function clears either the entire DATACACHE or a specific dataset within it.

    :param dataset: The dataset parameter is an optional argument that specifies the name of the dataset
    to be cleared from the DATACACHE dictionary. If no dataset name is provided, the entire DATACACHE
    dictionary will be cleared
    """
    if dataset is None:
        DATACACHE.clear()
    else:
        DATACACHE[dataset].clear()
