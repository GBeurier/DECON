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
    # print("<", model_name, ">")
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
        # Infos
        dataset_name, results, results_file = init_log(path)
        desc = (dataset_name, path, results_file, results, SEED)

        # Load data
        print("="*10, str(dataset_name).upper(), end=" ")
        X, y, X_valid, y_valid = load_data(path)
        print("="*10, X.shape, y.shape, X_valid.shape, y_valid.shape)

        # Split data
        for split_config in split_configs:
            # print("Split >", split_config)
            X_train, y_train, X_test, y_test = X, y, X_valid, y_valid
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

                folds = iter([([],[])])
                folds_size = 1
                name_cv = "NoCV"
                if cv_config is not None:
                    folder = RepeatedKFold(random_state=SEED, **cv_config)
                    folds_size = folder.get_n_splits()
                    folds = folder.split(X_train)
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
                            name_preprocessing = 'PP_' + str(len(preprocessing)) + "_" + str(hash(frozenset(preprocessing)))[0:5]

                            for model in models:
                                name_model = model[0].name()
                                if isinstance(model[0], regressors.NN_NIRS_Regressor):
                                    # print(name_model, "is Neural Net")
                                    X_test_pp, y_test_pp, transformer_pipeline, y_scaler = transform_test_data(
                                        preprocessing, X_tr, y_tr, X_te, y_te, type="augmentation"
                                    )
                                else:
                                    # print(name_model, "is Machine Learning")
                                    X_test_pp, y_test_pp, transformer_pipeline, y_scaler = transform_test_data(
                                        preprocessing, X_tr, y_tr, X_te, y_te, type="union"
                                    )

                                run_desc = "-".join([name_model, name_split, name_cv, name_fold, name_augmentation, name_preprocessing, str(SEED)])
                                run_key = "-".join([name_model, name_split, name_cv, name_augmentation, name_preprocessing, str(SEED)])

                                run_name = run_desc + "_" + datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
                                print(run_name, X_tr.shape, y_tr.shape, X_test_pp.shape, y_test_pp.shape)

                                regressor = model[0].model(X_tr, y_tr, X_test_pp, y_test_pp, run_name=run_name, cb=callback_predict, params=model[1])
                                transformers = (y_scaler, transformer_pipeline, regressor)

                                y_pred = evaluate_pipeline(desc, run_name, data, transformers)
                                cv_predictions[run_key] = cv_predictions[run_key] + y_pred if run_key in cv_predictions else y_pred
                

                for key, val in cv_predictions.items():
                    y_pred = val / folds_size
                    datasheet = get_datasheet(dataset_name, key, path, SEED, y_valid, y_pred)
                    results[key + "_CV"] = datasheet

                results = OrderedDict(sorted(results.items(), key=lambda k_v: float(k_v[1]["RMSE"])))
                with open(results_file, "w") as fp:
                    json.dump(results, fp, indent=4)

