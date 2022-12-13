from pathlib import Path
import datetime
import json
import math
import numpy as np
import os

from contextlib import redirect_stdout
# import joblib
# import pickle

from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics \
    import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,\
    r2_score, explained_variance_score, mean_squared_log_error, median_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
import tensorflow as tf

from data import load_data
from preprocessings import preprocessing_list, transform_test_data
from ml_regressors import ml_list
from nn_regressors import get_keras_model, nn_list

tf.get_logger().setLevel('ERROR')
tf.keras.mixed_precision.set_global_policy('mixed_float16')


def get_datasheet(dataset_name, model_name, path, y_valid, y_pred):
    return {
        "model": model_name,
        "dataset": dataset_name,
        "targetRMSE": str(float(os.path.split(path)[-1].split('_')[-1].split("RMSE")[-1])),
        "RMSE": str(mean_squared_error(y_valid, y_pred, squared=True)),
        "MAPE": str(mean_absolute_percentage_error(y_valid, y_pred)),
        "R2": str(r2_score(y_valid, y_pred)),
        "MAE": str(mean_absolute_error(y_valid, y_pred)),
        "MSE": str(mean_squared_error(y_valid, y_pred, squared=False)),
        "MedAE": str(median_absolute_error(y_valid, y_pred)),
        "EVS": str(explained_variance_score(y_valid, y_pred)),
        "MSLE": str(mean_squared_log_error(y_valid, y_pred)),
    }


def log_run(dataset_name, model_name, path, estimator, y_valid, y_pred):
    datasheet = get_datasheet(dataset_name, model_name, path, y_valid, y_pred)
    # Save data
    folder = "results/" + dataset_name
    if not os.path.isdir(folder):
        os.makedirs(folder)

    canon_name = folder + "/" + model_name

    # save predictions
    np.savetxt(canon_name + '.csv', np.column_stack((y_valid, y_pred)))

    # save main metrics globally
    result_file = open(folder + "/_runs.txt", "a")
    log = datasheet["RMSE"] + "  ---  " + model_name + ' '*10 + \
        datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S") + '\n'
    result_file.write(log)
    result_file.close()

    # save pipeline
    # joblib.dump(estimator, canon_name + '.pkl')

    return datasheet


def evaluate_pipeline(desc, model_name, data, transformers):
    # Unpack args
    X_train, y_train, X_valid, y_valid = data
    dataset_name, path = desc
    y_scaler, transformer_pipeline, regressor = transformers

    # Construct pipeline
    pipeline = Pipeline([
        ('transformation', transformer_pipeline),
        (model_name, regressor)
    ])

    # Fit estimator
    estimator = TransformedTargetRegressor(
        regressor=pipeline, transformer=y_scaler)
    estimator.fit(X_train, y_train)
    # Evaluate estimator
    y_pred = estimator.predict(X_valid)

    datasheet = log_run(dataset_name, model_name, path,
                        estimator, y_valid, y_pred)
    RMSE = datasheet["RMSE"]
    target_RMSE = datasheet["targetRMSE"]

    print(model_name, ">", RMSE, " (", target_RMSE, ")")

    return datasheet, y_pred


def benchmark_dataset(path, SEED):
    results = {}
    dataset_name = ('_').join(os.path.split(path)[-1].split('_')[:-1])
    print("="*10, str(dataset_name).upper(), "="*10)

    X, y, X_valid, y_valid = load_data(path)
    print("Data >", X.shape, y.shape, X_valid.shape, y_valid.shape)

    # # SINGLE RUN TRAINING
    # X_train, y_train, X_test, y_test = X, y, X_valid, y_valid
    # for preprocessing in preprocessing_list():
    #     data = (X_train, y_train, X_valid, y_valid)
    #     desc = (dataset_name, path)

    #     ##### DEEP LEARNING #####
    #     X_test_pp, y_test_pp, transformer_pipeline, y_scaler = transform_test_data(
    #         preprocessing, X_train, y_train, X_test, y_test, type="augmentation")
    #     for model_desc in nn_list():
    #         model_name = model_desc.__name__ + "-" + \
    #             preprocessing.__name__ + "-" + str(SEED)
    #         regressor = get_keras_model(
    #             dataset_name + '_' + model_name, model_desc, 5, 750, X_test_pp, y_test_pp, verbose=0, seed=SEED)
    #         transformers = (y_scaler, transformer_pipeline, regressor)
    #         results[model_name], _ = evaluate_pipeline(
    #             desc, model_name, data, transformers)

    #     ##### MACHINE LEARNING #####
    #     X_test_pp, y_test_pp, transformer_pipeline, y_scaler = transform_test_data(
    #         preprocessing, X_train, y_train, X_test, y_test, type="union")
    #     for regressor, mdl_name in ml_list(SEED, X_test_pp, y_test_pp):
    #         model_name = mdl_name + "-" + \
    #             preprocessing.__name__ + "-" + str(SEED)
    #         transformers = (y_scaler, transformer_pipeline, regressor)
    #         results[model_name], _ = evaluate_pipeline(
    #             desc, model_name, data, transformers)

    # CROSS VALIDATION TRAINING
    cv_predictions = {}
    for preprocessing in preprocessing_list():
        fold = RepeatedKFold(n_splits=3, n_repeats=2, random_state=SEED)
        fold_index = 0
        for train_index, test_index in fold.split(X):
            X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]
            data = (X_train, y_train, X_valid, y_valid)
            desc = (dataset_name, path)

            ##### DEEP LEARNING #####
            X_test_pp, y_test_pp, transformer_pipeline, y_scaler = transform_test_data(
                preprocessing, X_train, y_train, X_test, y_test, type="augmentation")
            for model_desc in nn_list():
                model_name = model_desc.__name__ + "-" + \
                    preprocessing.__name__ + "-" + str(SEED)
                fold_name = model_name + "-F" + str(fold_index)
                regressor = get_keras_model(
                    dataset_name + '_' + fold_name, model_desc, 5, 750, X_test_pp, y_test_pp, verbose=0, seed=SEED)
                results[fold_name], y_pred = evaluate_pipeline(
                    desc, fold_name, data, (y_scaler, transformer_pipeline, regressor))
                cv_predictions[model_name] = cv_predictions[model_name] + \
                    y_pred if model_name in cv_predictions else y_pred

            # ##### MACHINE LEARNING #####
            # X_test_pp, y_test_pp, transformer_pipeline, y_scaler = transform_test_data(
            #     preprocessing, X_train, y_train, X_test, y_test, type="union")
            # for regressor, mdl_name in ml_list(SEED, X_test_pp, y_test_pp):
            #     model_name = mdl_name + "-" + \
            #         preprocessing.__name__ + "-" + str(SEED)
            #     fold_name = model_name + "-F" + str(fold_index)
            #     results[fold_name], y_pred = evaluate_pipeline(
            #         desc, fold_name, data, (y_scaler, transformer_pipeline, regressor))
            #     cv_predictions[model_name] = cv_predictions[model_name] + \
            #         y_pred if model_name in cv_predictions else y_pred

            fold_index += 1

    for key, val in cv_predictions.items():
        y_pred = val / fold.get_n_splits()
        datasheet = get_datasheet(dataset_name, key, path, y_valid, y_pred)
        results[key + "_CV"] = datasheet

    results = sorted(results.items(), key=lambda k_v: float(k_v[1]['RMSE']))
    with open("results/" + dataset_name + "_" + str(SEED) + '_global_results.json', 'w') as fp:
        json.dump(results, fp, indent=4)


rootdir = Path('data/regression')
folder_list = [f for f in rootdir.glob('**/*') if f.is_dir()]

SEED = ord('D') + 31373
np.random.seed(SEED)
# tf.random.set_seed(SEED)

for folder in folder_list:
    benchmark_dataset(folder, SEED)
