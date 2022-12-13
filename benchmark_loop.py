import datetime
import json
import math
import numpy as np
from pathlib import Path
import time
import os
from collections import OrderedDict

from contextlib import redirect_stdout

# import joblib
# import pickle

from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    explained_variance_score,
    mean_squared_log_error,
    median_absolute_error,
)
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
import tensorflow as tf

from data import load_data
from preprocessings import transform_test_data
import regressors
from pinard import augmentation, sklearn, model_selection

tf.get_logger().setLevel("ERROR")
tf.keras.mixed_precision.set_global_policy("mixed_float16")


def get_datasheet(dataset_name, model_name, path, SEED, y_valid, y_pred):
    return {
        "model": model_name,
        "dataset": dataset_name,
        "seed": str(SEED),
        "targetRMSE": str(float(os.path.split(path)[-1].split("_")[-1].split("RMSE")[-1])),
        "RMSE": str(mean_squared_error(y_valid, y_pred, squared=False)),
        "MAPE": str(mean_absolute_percentage_error(y_valid, y_pred)),
        "R2": str(r2_score(y_valid, y_pred)),
        "MAE": str(mean_absolute_error(y_valid, y_pred)),
        "MSE": str(mean_squared_error(y_valid, y_pred, squared=True)),
        "MedAE": str(median_absolute_error(y_valid, y_pred)),
        "EVS": str(explained_variance_score(y_valid, y_pred)),
        # "MSLE":str(mean_squared_log_error(y_valid, y_pred)),
        "run": datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
    }


def log_run(dataset_name, model_name, path, SEED, y_valid, y_pred, elapsed_time):
    datasheet = get_datasheet(dataset_name, model_name, path, SEED, y_valid, y_pred)
    # Save data
    folder = "results/" + dataset_name
    if not os.path.isdir(folder):
        os.makedirs(folder)

    canon_name = folder + "/" + model_name

    # save predictions
    np.savetxt(canon_name + ".csv", np.column_stack((y_valid, y_pred)))

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


def callback_predict(epoch, val_loss):
    if current_estimator is None:
        return

    y_pred = current_estimator.predict(current_X_test)
    res = get_datasheet("", "", current_path, -1, current_y_test, y_pred)
    print("Epoch:", epoch, "> RMSE:", res["RMSE"], "(", res["targetRMSE"], ") - R²:", res["R2"], " val_loss", val_loss)


def evaluate_pipeline(desc, model_name, data, transformers):
    print("<", model_name, ">")
    start_time = time.time()

    # Unpack args
    X_train, y_train, X_valid, y_valid = data
    dataset_name, path, global_result_file, results, SEED = desc
    global current_path
    current_path = path
    y_scaler, transformer_pipeline, regressor = transformers

    # Construct pipeline
    pipeline = Pipeline([("transformation", transformer_pipeline), (model_name, regressor)])

    # Fit estimator
    estimator = TransformedTargetRegressor(regressor=pipeline, transformer=y_scaler)
    global current_estimator
    current_estimator = estimator
    global current_X_test
    current_X_test = X_valid
    global current_y_test
    current_y_test = y_valid
    estimator.fit(X_train, y_train)
    # Evaluate estimator
    y_pred = estimator.predict(X_valid)
    elapsed_time = time.time() - start_time
    datasheet = log_run(dataset_name, model_name, path, SEED, y_valid, y_pred, elapsed_time)
    datasheet["training_time"] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    results[model_name] = datasheet

    # Save results
    results = OrderedDict(sorted(results.items(), key=lambda k_v: float(k_v[1]["RMSE"])))
    with open(global_result_file, "w") as fp:
        json.dump(results, fp, indent=4)

    print(datasheet["RMSE"], " (", datasheet["targetRMSE"], ") in", datasheet["training_time"])
    return y_pred


# Data X_Y (train_test) - datapath + names ?
# SEED
# Augmentation_list
# PP list
# Model
### Fit (single, CV)
# Data results - resultpath


def augment_dataset():
    if augmentations == None or len(augmentations) == 0:
        augmentations = [None]

    for augmentation in augmentation:
        if augmentation is not None:
            augmentation_pipeline = sklearn.SampleAugmentation(augmentation)
            print(X_train.shape, y_train.shape)
            X_train, y_train = augmentation_pipeline.transform(X_train, y_train)
            print("augmented to:", X_train.shape, y_train.shape)
        preprocess_dataset()


def init_log(path):
    ### DATASET in the form (name)_RMSE(val)
    dataset_name = ("_").join(os.path.split(path)[-1].split("_")[:-1])

    ### Get global results database for this dataset
    global_result_file = "results/" + dataset_name + "_results.json"
    results = {}
    if os.path.isfile(global_result_file):
        with open(global_result_file) as json_file:
            results = json.load(json_file)

    return dataset_name, results, global_result_file


def benchmark_dataset(dataset_list, split_configs, cv_configs, augmentations, preprocessings, models, SEED):
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    for path in dataset_list:
        ## Infos
        dataset_name, results, results_file = init_log(path)
        desc = (dataset_name, path, results_file, results)

        ## Load data
        print("=" * 10, str(dataset_name).upper(), end=" ")
        X, y, X_valid, y_valid = load_data(path)
        print(X.shape, y.shape, X_valid.shape, y_valid.shape, "=" * 10)

        ## Split data
        for split_config in split_configs:
            print("split>", split_config)
            X_train, y_train, X_test, y_test = X, y, X_valid, y_valid
            if split_config is not None:
                train_index, test_index = model_selection.train_test_split_idx(X, y=y, **split_config)
                X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]

                ## Replace validation set by test set if validation set is empty
                if X_valid.size == 0:
                    X_valid, y_valid = X_test, y_test

            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_valid.shape, y_valid.shape)

        #     ## Generate training sets
        #     for cv_config in cv_configs:
        #         fold = iter([])

        # ## Generate train/test
        # datasets_indices = []
        # if "cross_validation" not in training_config:
        #     train, test = np.arange(len(y), len(y_valid))
        #     datasets.append((train, test))
        # else:
        #     pass

        # if augment:
        #     augmentation_pipeline = sklearn.SampleAugmentation(
        #         [
        #             (2, "rot_tr", augmentation.Rotate_Translate()),
        #             (1, "rd_mult", augmentation.Random_X_Operation()),
        #             (1, "simpl", augmentation.Random_Spline_Addition()),
        #         ]
        #     )
        #     print(X_train.shape, y_train.shape)
        #     X_train, y_train = augmentation_pipeline.transform(X_train, y_train)
        #     print("augmented to:", X_train.shape, y_train.shape)

        # data = (X_train, y_train, X_valid, y_valid)
        # for preprocessing in preprocessing_list:
        #     ##### DEEP LEARNING #####
        #     X_test_pp, y_test_pp, transformer_pipeline, y_scaler = transform_test_data(
        #         preprocessing, X_train, y_train, X_test, y_test, type="augmentation"
        #     )
        #     for model_desc in nn_list():
        #         model_name = model_desc.__name__ + "-" + preprocessing.__name__ + "-" + str(SEED)
        #         if os.path.isfile("results/" + dataset_name + "/" + model_name + ".csv"):
        #             # print("Skipping", model_name)
        #             continue

        #         # batch_size = 3000
        #         # if preprocessing.__name__ == "dumb_set":
        #         #     batch_size = 3
        #         regressor = get_keras_model(
        #             dataset_name + "_" + model_name,
        #             model_desc,
        #             4096,
        #             batch_size,
        #             X_test_pp,
        #             y_test_pp,
        #             transfer=True,
        #             callback_func=callback_predict,
        #             verbose=0,
        #             seed=SEED,
        #         )
        #         transformers = (y_scaler, transformer_pipeline, regressor)
        #         evaluate_pipeline(desc, model_name, data, transformers)

        #     # ##### MACHINE LEARNING #####
        #     # X_test_pp, y_test_pp, transformer_pipeline, y_scaler = transform_test_data(preprocessing, X_train, y_train, X_test, y_test, type="union")
        #     # for regressor, mdl_name in ml_list(SEED, X_test_pp, y_test_pp):
        #     #     model_name = mdl_name + "-" + preprocessing.__name__ + "-" + str(SEED)
        #     #     if os.path.isfile(  "results/" + dataset_name + "/" + model_name + '.csv'):
        #     #        # print("Skipping", model_name)
        #     #         continue
        #     #     transformers = (y_scaler, transformer_pipeline, regressor)
        #     #     evaluate_pipeline(desc, model_name, data, transformers)

        # #########################

        # #########################
        # ### CROSS VALIDATION TRAINING
        # cv_predictions = {}
        # for preprocessing in preprocessing_list():
        #     fold = RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED)
        #     fold_index = 0
        #     for train_index, test_index in fold.split(X):
        #         X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]
        #         data = (X_train, y_train, X_valid, y_valid)

        #         ##### DEEP LEARNING #####
        #         X_test_pp, y_test_pp, transformer_pipeline, y_scaler = transform_test_data(
        #             preprocessing, X_train, y_train, X_test, y_test, type="augmentation"
        #         )
        #         for model_desc in nn_list():
        #             model_name = model_desc.__name__ + "-" + preprocessing.__name__ + "-" + str(SEED)
        #             fold_name = model_name + "-F" + str(fold_index)
        #             if os.path.isfile("results/" + dataset_name + "/" + fold_name + ".csv"):
        #                 # print("Skipping", model_name)
        #                 continue
        #             regressor = get_keras_model(dataset_name + "_" + fold_name, model_desc, 7500, 750, X_test_pp, y_test_pp, verbose=0, seed=SEED)
        #             y_pred = evaluate_pipeline(desc, fold_name, data, (y_scaler, transformer_pipeline, regressor))
        #             cv_predictions[model_name] = cv_predictions[model_name] + y_pred if model_name in cv_predictions else y_pred

        #         ##### MACHINE LEARNING #####
        #         X_test_pp, y_test_pp, transformer_pipeline, y_scaler = transform_test_data(
        #             preprocessing, X_train, y_train, X_test, y_test, type="union"
        #         )
        #         for regressor, mdl_name in ml_list(SEED, X_test_pp, y_test_pp):
        #             model_name = mdl_name + "-" + preprocessing.__name__ + "-" + str(SEED)
        #             fold_name = model_name + "-F" + str(fold_index)
        #             if os.path.isfile("results/" + dataset_name + "/" + fold_name + ".csv"):
        #                 #  print("Skipping", model_name)
        #                 continue
        #             y_pred = evaluate_pipeline(desc, fold_name, data, (y_scaler, transformer_pipeline, regressor))
        #             cv_predictions[model_name] = cv_predictions[model_name] + y_pred if model_name in cv_predictions else y_pred

        #         fold_index += 1

        # for key, val in cv_predictions.items():
        #     y_pred = val / fold.get_n_splits()
        #     datasheet = get_datasheet(dataset_name, key, path, SEED, y_valid, y_pred)
        #     results[key + "_CV"] = datasheet

        # results = OrderedDict(sorted(results.items(), key=lambda k_v: float(k_v[1]["RMSE"])))
        # with open(global_result_file, "w") as fp:
        #     json.dump(results, fp, indent=4)

        # # #########################
