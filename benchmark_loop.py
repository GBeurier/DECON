import core.datacache as datacache
import datetime
# import glob
import json
# import math
import numpy as np
# from pathlib import Path
import time
import os
from collections import OrderedDict
import random
# import # logging
from scipy import signal
import joblib

# import joblib
# import pickle

# import signal
# import time
# import sys

from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    explained_variance_score,
    median_absolute_error,
)
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
import tensorflow as tf

# from data import load_data, load_data_multiple
from preprocessings import transform_test_data
import regressors
from pinard import augmentation, sklearn, model_selection


class regressor_stratified_cv:
    def __init__(self, n_splits=10, n_repeats=2, group_count=10, random_state=0, strategy='quantile'):
        self.group_count = group_count
        self.strategy = strategy
        self.cvkwargs = dict(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.cv = RepeatedStratifiedKFold(**self.cvkwargs)
        self.discretizer = KBinsDiscretizer(n_bins=self.group_count, encode='ordinal', strategy=self.strategy)

    def split(self, X, y, groups=None):
        kgroups = self.discretizer.fit_transform(y[:, None])[:, 0]
        return self.cv.split(X, kgroups, groups)

    def get_n_splits(self, X, y, groups=None):
        return self.cv.get_n_splits(X, y, groups)


def get_datasheet(dataset_name, model_name, path, SEED, y_valid, y_pred):
    return {
        "model": model_name,
        "dataset": dataset_name,
        "seed": str(SEED),
        "RMSE": str(mean_squared_error(y_valid, y_pred, squared=False)),
        "MAPE": str(mean_absolute_percentage_error(y_valid, y_pred)),
        "R2": str(r2_score(y_valid, y_pred)),
        "MAE": str(mean_absolute_error(y_valid, y_pred)),
        "MSE": str(mean_squared_error(y_valid, y_pred, squared=True)),
        "MedAE": str(median_absolute_error(y_valid, y_pred)),
        "EVS": str(explained_variance_score(y_valid, y_pred)),
        "run": datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
        "path": path,
    }


def log_run(dataset_name, model_name, path, SEED, y_valid, y_pred, elapsed_time, model_type):
    datasheet = get_datasheet(dataset_name, model_name, path, SEED, y_valid, y_pred)
    datasheet["type"] = model_type
    datasheet["training_time"] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    # Save data
    folder = os.path.join("results", dataset_name)
    if not os.path.isdir(folder):
        os.makedirs(folder)

    canon_name = folder + "/" + model_name

    # save predictions
    np.savetxt(canon_name + ".csv", np.column_stack((y_valid, y_pred)), delimiter=";")

    # save main metrics globally
    result_file = open(folder + "/_runs.txt", "a")
    log = (
        datasheet["RMSE"]
        + "  ---  "
        + model_name
        + " in "
        + time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        + " ("
        + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        + ")\n"
    )
    result_file.write(log)
    result_file.close()

    # save pipeline
    # joblib.dump(estimator, canon_name + '.pkl')

    return datasheet


current_estimator = None
current_X_test = None
current_y_test = None
current_path = None
current_y_scaler = None


def get_callback_predict(dataset_name, model_name, path, SEED, target_RMSE, best_current_model, discretizer, *, model_type):

    def callback_predict(epoch, val_loss):
        global current_estimator
        if current_estimator is None:
            return

        y_pred = current_estimator.predict(current_X_test)
        if discretizer is not None:
            y_pred = discretizer.inverse_transform(y_pred)

        print("*" * 10, current_X_test.shape, current_y_test.shape, y_pred.shape)

        res = get_datasheet("", "", current_path, -1, current_y_test, y_pred)
        print("Epoch:", epoch, "> RMSE:", res["RMSE"], " (", target_RMSE, "|", best_current_model, ") - R²:", res["R2"], " val_loss", val_loss)

        if float(res["RMSE"]) / 1.1 < float(target_RMSE):
            log_run(dataset_name, model_name, path, SEED, current_y_test, y_pred, 0, model_type)

        return res

    return callback_predict


def evaluate_pipeline(desc, model_name, data, transformers, target_RMSE, best_current_model, discretizer=None):
    start_time = time.time()

    # Unpack args
    X_train, y_train, X_valid, y_valid, X_test_pp, y_test_pp = data
    dataset_name, path, global_result_file, results, SEED = desc
    global current_path
    current_path = path
    y_scaler, transformer_pipeline, regressor, model_desc, callbacks = transformers

    # Construct pipeline
    # pipeline = Pipeline([("transformation", transformer_pipeline), (model_name, regressor)])

    # # Fit estimator
    # if discretizer is None:
    #     estimator = TransformedTargetRegressor(regressor=pipeline, transformer=y_scaler)
    # else:
    #     estimator = pipeline

    X_train_ = transformer_pipeline.transform(X_train)

    global current_estimator
    current_estimator = regressor
    global current_X_test
    current_X_test = X_test_pp
    global current_y_test
    current_y_test = y_valid
    print(regressor)

    regressor.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
    params = model_desc[1]
    regressor.fit(
        X_train_,
        y_train,
        epochs=params["epoch"],
        batch_size=params["batch_size"],
        verbose=params["verbose"],
        callbacks=callbacks,
        validation_data=(X_test_pp, y_test_pp))

    y_pred = regressor.predict(X_valid)
    y_pred = y_scaler.inverse_transform(y_pred)
    elapsed_time = time.time() - start_time

    datasheet = log_run(dataset_name, model_name, path, SEED, y_valid, y_pred, elapsed_time, "NN")
    results[model_name] = datasheet

    # Save results
    results = OrderedDict(sorted(results.items(), key=lambda k_v: float(k_v[1]["RMSE"])))
    with open(global_result_file, "w") as fp:
        json.dump(results, fp, indent=4)

    folder = os.path.join("results", dataset_name)
    canon_name = folder + "/" + model_name

    # print("Saving model to", canon_name)
    joblib.dump(transformer_pipeline, canon_name + "_tf.pkl")
    # print(regressor)
    # if isinstance(model[0], regressors.NN_NIRS_Regressor):
    regressor.model_.save(canon_name + ".h5")
    # pass
    # else:
    # joblib.dump(estimator.regressor_[1], canon_name + "_reg.pkl")

    joblib.dump(y_scaler, canon_name + "y_scaler.pkl")

    return y_pred, datasheet


def get_augmentation(augmentation_config):
    if augmentation_config is None:
        return "NoAug", None
    augmentation_array = []
    name_augmentation = "Aug"
    for aug in augmentation_config:
        if augmentation is not None:
            transfo_name_b = str(type(aug[1])).split('.')[-1].split('_')
            transfo_name = "".join([c[0] for c in transfo_name_b])
            name_augmentation += "_" + str(aug[0]) + transfo_name
            augmentation_array.append((aug[0], name_augmentation, aug[1]))

    return name_augmentation, sklearn.SampleAugmentation(augmentation_array)


def init_log(path):
    # DATASET in the form (name)_RMSE(val)
    dataset_name = os.path.split(path)[-1]

    # Get global results database for this dataset
    global_result_file = "results/" + dataset_name + "_results.json"
    results = {}
    if os.path.isfile(global_result_file):
        with open(global_result_file) as json_file:
            results = json.load(json_file)

    return dataset_name, results, global_result_file


def benchmark_dataset(
        dataset_list, split_configs, cv_configs, augmentations, preprocessings, models, SEED, *, bins=None, resampling=None, resample_size=0, weight_config=False,
        skip_existing=True):

    if SEED == -1:
        SEED = np.random.randint(0, 10000)

    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED, version=2)

    classification_mode = False if bins is None else True
    name_classif = "" if classification_mode is False else '_Cl' + str(bins) + '_'

    for path in dataset_list:
        # Infos
        # logging.info("Dataset: " + path)
        dataset_name, results, results_file = init_log(path)
        previous_models = []
        if os.path.exists(results_file) and skip_existing:
            previous_results = json.load(open(results_file))
            previous_models = [k[:-18] for k in previous_results.keys()]
        # print(previous_models)
        # return

        target_RMSE, best_current_model = "-1", "None"
        if len(results.keys()) > 0:
            best_current_model = list(results.keys())[0]
            target_RMSE = results[best_current_model]["RMSE"]

        folder = os.path.join("results", dataset_name)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        desc = (dataset_name, path, results_file, results, SEED)
        # logging.info("Dataset: %s, %s, %s, %s, %s" % desc)

        # Load data
        print("=" * 10, str(dataset_name).upper(), end=" ")
        # print("Loading data...", path)

        # X, y, X_valid, y_valid = load_data(path, resampling, resample_size)
        # print("="*10, X.shape, y.shape, X_valid.shape, y_valid.shape)

        dataset_hash, dataset_name, cache = datacache.register_dataset(path)
        # print("Dataset hash", dataset_hash, dataset_name)
        # cache = datacache.get_data_from_uid(dataset_name, dataset_hash)
        X, y, X_valid, y_valid = cache["X_train"][0], cache["y_train"], cache["X_val"][0], cache["y_val"]

        if X.shape[-1] <= 256:
            new_X = []
            for x in X:
                new_X.append(signal.resample(x, 256))
            X = np.array(new_X)

        if X_valid.shape[-1] <= 256:
            new_X_valid = []
            for x in X_valid:
                new_X_valid.append(signal.resample(x, 256))
            X_valid = np.array(new_X_valid)

        print("=" * 10, X.shape, y.shape, X_valid.shape, y_valid.shape)
        # logging.info("=" * 10 + str(X.shape) + str(y.shape) + str(X_valid.shape) + str(y_valid.shape))
        # Split data
        for split_config in split_configs:
            # print("Split >", split_config)
            X_train, y_train, X_test, y_test = X, y, X_valid, y_valid
            discretizer = None
            if classification_mode:
                discretizer = KBinsDiscretizer(n_bins=bins, encode='onehot-dense', strategy='uniform')
                discretizer.fit(y_train)
                y_train = discretizer.transform(y_train.reshape((-1, 1)))
                y_test = discretizer.transform(y_test.reshape((-1, 1)))
                print("Discretized shape", y_train.shape, y_test.shape)
            # else:
            #     y_train = y_train.reshape((-1, 1))
            #     y_test = y_test.reshape((-1, 1))

            name_split = "NoSpl"
            if split_config is not None:
                train_index, test_index = model_selection.train_test_split_idx(X, y=y, **split_config)
                X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]

                # Replace validation set by test set if validation set is empty
                if X_valid.size == 0:
                    X_valid, y_valid = X_test, y_test

                name_split = 'Spl_' + str(split_config['method']) + "_" + str(hash(frozenset(split_config)))[0:3]

            # print("Split >", X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_valid.shape, y_valid.shape)

            # Generate training sets
            for cv_config in cv_configs:
                # print("CV >", cv_config)

                folds = iter([([], [])])
                folds_size = 1
                name_cv = "NoCV"
                time_str_cv = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
                if cv_config is not None:
                    cv = RepeatedKFold(random_state=SEED, **cv_config)
                    folds = cv.split(X_train)
                    # cv = regressor_stratified_cv(**cv_config, group_count=5, random_state=SEED, strategy='uniform')
                    # folds = cv.split(X_train, y_train)
                    folds_size = cv.get_n_splits()
                    name_cv = "CV_" + str(cv_config['n_splits']) + "_" + str(cv_config['n_repeats'])

                # Loop data
                fold_i = 1
                cv_predictions = {}
                for train_index, test_index in folds:
                    if len(train_index) > 0:
                        X_tr, y_tr, X_te, y_te = X_train[train_index], y_train[train_index], X_train[test_index], y_train[test_index]
                    else:
                        X_tr, y_tr, X_te, y_te = X_train, y_train, X_test, y_test

                    name_fold = "Fold_" + str(fold_i) + '(' + str(folds_size) + ')'
                    # print(name_fold, X_tr.shape, y_tr.shape, X_te.shape, y_te.shape)
                    fold_i += 1

                    # Loop augmentations
                    for augmentation_config in augmentations:
                        name_augmentation, augmentation_pipeline = get_augmentation(augmentation_config)
                        if augmentation_pipeline is not None:
                            X_tr, y_tr = augmentation_pipeline.transform(X_tr, y_tr)
                        # print(name_augmentation, X_tr.shape, y_tr.shape)
                        data = (X_tr, y_tr, X_valid, y_valid)

                        # Loop preprocessing
                        for preprocessing in preprocessings:
                            # print(preprocessing)
                            # continue
                            name_preprocessing = 'NoPP_'
                            if preprocessing is not None:
                                if isinstance(preprocessing, tuple):
                                    name_preprocessing = preprocessing[0]
                                elif len(preprocessing) == 1:
                                    name_preprocessing = preprocessing[0][0]
                                    # if len(preprocessing) == 1:
                                # name_preprocessing = 'PP_' + str(len(preprocessing))  # + "_" + str(hash(frozenset(preprocessing)))[0:5]
                            # logging.info("Preprocessing " + name_preprocessing)
                            for model in models:
                                if SEED == -1:
                                    SEED = np.random.randint(0, 10000)
                                    np.random.seed(SEED)
                                    tf.random.set_seed(SEED)
                                    random.seed(SEED, version=2)

                                name_model = model[0].name()
                                time_str = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
                                run_name = "-".join([name_model, name_classif, name_split, name_cv, name_fold, name_augmentation, name_preprocessing, str(SEED), time_str])
                                run_key = "-".join([name_model, name_classif, name_split, name_cv, name_augmentation, name_preprocessing, str(SEED), time_str_cv])
                                # logging.info("RUN " + run_name)
                                if run_name[:-18] in previous_models:
                                    continue

                                print("RUN", dataset_name, run_name.split("-")[0], run_name[:-18].split("-")[-2])
                                # print(name_model, "is Neural Net")
                                X_test_pp, y_test_pp, transformer_pipeline, y_scaler = transform_test_data(preprocessing, X_tr, y_tr, X_te, y_te, type="augmentation", classification_mode=classification_mode)
                                data = (X_tr, y_tr, X_valid, y_valid, X_test_pp, y_test_pp)
                                # print("RUN", run_name, X_tr.shape, y_tr.shape, X_test_pp.shape, y_test_pp.shape)
                                # continue
                                # print(">"*10, X_tr[0], y_test_pp[0])
                                model_type = "ML"
                                if isinstance(model[0], regressors.NN_NIRS_Regressor):
                                    model_type = "NN"
                                cb_predict = get_callback_predict(dataset_name, name_model, path, SEED, target_RMSE, best_current_model, discretizer, model_type=model_type)

                                # joblib.dump(transformer_pipeline, "transformer_pipeline.pkl")
                                # logging.info("Starting training" + str(X_tr.shape) + str(y_tr.shape) + str(X_test_pp.shape) + str(y_test_pp.shape))
                                regressor, callbacks = model[0].model(X_tr, y_tr, X_test_pp, y_test_pp, run_name=run_name, cb=cb_predict, params=model[1], desc=desc, discretizer=discretizer)

                                binary_prefix = "-".join([name_model, name_classif, name_split, name_cv, name_fold, name_augmentation, name_preprocessing])

                                # if isinstance(model[0], regressors.NN_NIRS_Regressor) and len(model) == 3:
                                #     weight_config = model[2]
                                #     if isinstance(weight_config, str):
                                #         regressor.load_weights(weight_config)
                                #         # print("loaded weights from", weight_config)
                                #     elif isinstance(weight_config, bool) and weight_config:
                                #         # search for last created weight file that names starts with run_name in results folder
                                #         regex = os.path.join("results", desc[0], binary_prefix + '*' + '.h5')
                                #         weight_files = glob.glob(regex)
                                #         if len(weight_files) == 0:
                                #             print("No weight file found for", binary_prefix)
                                #         else:
                                #             weight_file = max(weight_files, key=os.path.getctime)
                                #             regressor.model = tf.keras.models.load_model(weight_file)
                                #             print("loaded weights from", weight_file)
                                # else:
                                #     print("No weights to load")

                                # print(model)
                                transformers = (y_scaler, transformer_pipeline, regressor, model, callbacks)
                                y_pred, datasheet = evaluate_pipeline(desc, run_name, data, transformers, target_RMSE, best_current_model, discretizer)
                                print(dataset_name, run_name, datasheet["RMSE"], " (", target_RMSE, ") in", datasheet["training_time"])  # "|", best_current_model,
                                if name_cv != "NoCV":
                                    if run_key not in cv_predictions:
                                        cv_predictions[run_key] = []
                                    cv_predictions[run_key].append({'pred': y_pred, 'RMSE': float(datasheet['RMSE'])})

                # if name_cv != "NoCV":
                #     for key, val in cv_predictions.items():
                #         RMSE_TOT = sum(item['RMSE'] for item in val)
                #         factor = 1. / (len(val) - 1)
                #         weights = [factor * (RMSE_TOT - item['RMSE']) / RMSE_TOT for item in val]
                #         y_pred = np.sum([weights[i]*val[i]['pred'] for i in range(len(val))], axis=0)
                #         datasheet = get_datasheet(dataset_name, key, path, SEED, y_valid, y_pred)
                #         results[key + "_CV"] = datasheet
                print("Finished", dataset_name, "with", len(results), "results")
                # logging.info("Finished", dataset_name, "with", len(results), "results")
                results = OrderedDict(sorted(results.items(), key=lambda k_v: float(k_v[1]["RMSE"])))
                with open(results_file, "w") as fp:
                    json.dump(results, fp, indent=4)
