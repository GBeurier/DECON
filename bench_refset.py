import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
# from pinard import augmentation, sklearn, model_selection
# from preprocessings import transform_test_data
from pinard.sklearn import FeatureAugmentation
from pinard import preprocessing as pp
from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import KBinsDiscretizer
# from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
import core.datacache as datacache
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, Input, SpatialDropout1D, SeparableConv1D, BatchNormalization, MaxPool1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, Callback, LearningRateScheduler  # , ReduceLROnPlateau
import os.path
import sys
import pinard.preprocessing as pp
import regressors
import preprocessings
from preprocessings import preprocessing_list
from benchmark_loop import benchmark_dataset
from pathlib import Path
# import random
# import tensorflow as tf
# import traceback
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    explained_variance_score,
    median_absolute_error,
)
import json


def str_to_class(classname):
    return getattr(sys.modules['pinard.preprocessing'], classname)


def get_dataset_list(path):
    datasets = []
    for r, d, _ in os.walk(path):
        for folder in d:
            # print(r, folder)
            path = os.path.join(r, folder)
            if os.path.isdir(path):
                print(path)
                # if len(datasets) < 3:
                datasets.append(str(path))
                # break
    return datasets


split_configs = [None]

augmentations = [None]

preprocessings_list = [
    # None,
    [("id", pp.IdentityTransformer())],
    # [("baseline", pp.StandardNormalVariate())],
    # [("savgol", pp.SavitzkyGolay())],
    # [("gaussian1", pp.Gaussian(order=1, sigma=2))],
    # [("gaussian2", pp.Gaussian(order=2, sigma=1))],
    # [("haar", pp.Wavelet("haar"))],
    # [("coif3", pp.Wavelet("coif3"))],
    # [("detrend", pp.Detrend())],
    # [("msc", pp.MultiplicativeScatterCorrection(scale=False))],
    # [("dv1", pp.Derivate(1, 1))],
    # [("dv2", pp.Derivate(2, 1))],
    # [("dv3", pp.Derivate(2, 2))],
    ("simple", [("id", pp.IdentityTransformer()), ('haar', pp.Haar()), ('savgol', pp.SavitzkyGolay())]),
    ("small_set", preprocessings.small_set()),
    ("transf_set", preprocessings.transf_set()),
    ("bacon_set", preprocessings.bacon_set()),
    ("decon_set", preprocessings.decon_set()),
    # ("all2D_set", preprocessings.optimal_set_2D()),
    # preprocessings.id_preprocessing(),
    # preprocessings.fat_set(),
]

cv_configs = [
    None,
]

# folder = "data"
# folders = get_dataset_list(folder)

len_cv_configs = 0
for c in cv_configs:
    if c is None:
        len_cv_configs += 1
    else:
        len_cv_configs += (c['n_splits'] * c['n_repeats'])

models = [
    [
        (regressors.Decon(), {'batch_size': 100, 'epoch': 20000, 'verbose': 0, 'patience': 400, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-6, "max_lr": 1e-3, "cycle_length": 256}),
        (regressors.Transformer(), {'batch_size': 30, 'epoch': 300, 'verbose': 0, 'patience': 30, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-5, "max_lr": 1e-2, "cycle_length": 32}),
        (regressors.Transformer_LongRange(), {'batch_size': 30, 'epoch': 300, 'verbose': 0, 'patience': 30, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-5, "max_lr": 1e-2, "cycle_length": 32}),
        (regressors.MLP(), {'batch_size': 1000, 'epoch': 20000, 'verbose': 0, 'patience': 2000, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-6, "max_lr": 1e-3, "cycle_length": 256}),
        (regressors.FFT_Conv(), {'batch_size': 500, 'epoch': 20000, 'verbose': 0, 'patience': 1000, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-6, "max_lr": 1e-3, "cycle_length": 256}),
        (regressors.CONV_LSTM(), {'batch_size': 1000, 'epoch': 20000, 'verbose': 0, 'patience': 2000, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-6, "max_lr": 1e-3, "cycle_length": 256}),
    ],
    [
        (regressors.XCeption1D(), {'batch_size': 500, 'epoch': 10000, 'verbose': 0, 'patience': 1200, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-6, "max_lr": 1e-3, "cycle_length": 256}),
        (regressors.Inception1D(), {'batch_size': 500, 'epoch': 10000, 'verbose': 0, 'patience': 1200, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-6, "max_lr": 1e-3, "cycle_length": 256}),
        (regressors.ResNetV2(), {'batch_size': 500, 'epoch': 10000, 'verbose': 0, 'patience': 1200, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-6, "max_lr": 1e-3, "cycle_length": 256}),
        (regressors.UNET(), {'batch_size': 500, 'epoch': 10000, 'verbose': 0, 'patience': 1200, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-6, "max_lr": 1e-3, "cycle_length": 256}),
        (regressors.UNet_NIRS(), {'batch_size': 500, 'epoch': 10000, 'verbose': 0, 'patience': 1200, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-6, "max_lr": 1e-3, "cycle_length": 256}),
    ],
    [
        (regressors.Decon_Sep(), {'batch_size': 500, 'epoch': 20000, 'verbose': 0, 'patience': 1000, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-6, "max_lr": 1e-3, "cycle_length": 256}),
        (regressors.Decon_Sep(), {'batch_size': 500, 'epoch': 20000, 'verbose': 0, 'patience': 1000, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-6, "max_lr": 1e-3, "cycle_length": 256}),
        (regressors.Decon_Sep(), {'batch_size': 500, 'epoch': 20000, 'verbose': 0, 'patience': 1000, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-6, "max_lr": 1e-3, "cycle_length": 256}),
        (regressors.Decon_Sep(), {'batch_size': 500, 'epoch': 20000, 'verbose': 0, 'patience': 1000, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-6, "max_lr": 1e-3, "cycle_length": 256}),
        (regressors.Decon(), {'batch_size': 100, 'epoch': 20000, 'verbose': 0, 'patience': 400, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-6, "max_lr": 1e-3, "cycle_length": 256}),
        (regressors.Decon(), {'batch_size': 100, 'epoch': 20000, 'verbose': 0, 'patience': 400, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-6, "max_lr": 1e-3, "cycle_length": 256}),
        (regressors.Bacon(), {'batch_size': 500, 'epoch': 20000, 'verbose': 0, 'patience': 1000, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-6, "max_lr": 1e-3, "cycle_length": 256}),
        (regressors.Bacon(), {'batch_size': 500, 'epoch': 20000, 'verbose': 0, 'patience': 1000, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-6, "max_lr": 1e-3, "cycle_length": 256}),
    ]
]


# from data import load_data, load_data_multiple

def scale_fn(x):
    # return 1. ** x
    return 1 / (2.0 ** (x - 1))


def get_clr(params):
    min_lr = params.get("min_lr", 0.0001)
    max_lr = params.get("max_lr", 0.05)
    cycle_length = params.get("cycle_length", 256)

    def clr(epoch):
        # return 0.05
        cycle_params = {
            "MIN_LR": min_lr,
            "MAX_LR": max_lr,
            "CYCLE_LENGTH": cycle_length,
        }
        MIN_LR, MAX_LR, CYCLE_LENGTH = (
            cycle_params["MIN_LR"],
            cycle_params["MAX_LR"],
            cycle_params["CYCLE_LENGTH"],
        )
        initial_learning_rate = MIN_LR
        maximal_learning_rate = MAX_LR
        step_size = CYCLE_LENGTH
        step_as_dtype = float(epoch)
        cycle = math.floor(1 + step_as_dtype / (2 * step_size))
        x = abs(step_as_dtype / step_size - 2 * cycle + 1)
        mode_step = cycle  # if scale_mode == "cycle" else step
        return initial_learning_rate + (maximal_learning_rate - initial_learning_rate) * max(0, (1 - x)) * scale_fn(mode_step)
    return clr


class Auto_Save(Callback):
    best_weights = []

    def __init__(self, model_name, shape, cb_func=None):
        super(Auto_Save, self).__init__()
        self.model_name = model_name
        self.shape = shape
        self.best = np.Inf
        self.cb = cb_func

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss")
        if np.less(current_loss, self.best):
            self.best = current_loss
            Auto_Save.best_weights = self.model.get_weights()
            self.best_epoch = epoch
            print("Best so far >", self.best, self.model_name)

    def on_train_end(self, logs=None):
        self.model.set_weights(Auto_Save.best_weights)
        # self.model.save_weights(self.model_name + ".hdf5")
        self.model.save(self.model_name + ".h5")
        print(self.model.summary())


def process_data(X, y, X_valid, y_valid, preprocessing):
    y_scaler = MinMaxScaler()
    y_scaler.fit(y.reshape((-1, 1)))
    y_train = y_scaler.transform(y.reshape((-1, 1)))
    y_test = y_scaler.transform(y_valid.reshape((-1, 1)))

    transformer_pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("preprocessing", FeatureAugmentation(preprocessing)),
    ])

    transformer_pipeline.fit(X)
    X_train = transformer_pipeline.transform(X)
    X_test = transformer_pipeline.transform(X_valid)

    return X_train, y_train, X_test, y_test, y_scaler, transformer_pipeline


def main():
    folder = sys.argv[1]
    dataset_name = sys.argv[1]
    print(f"Processing folder: {folder}")

    dataset_hash, dataset_name, cache = datacache.register_dataset(folder)
    X, y, X_valid, y_valid = cache["X_train"][0], cache["y_train"], cache["X_val"][0], cache["y_val"]
    X_train, y_train, X_test, y_test, y_scaler, transformer_pipeline = process_data(X, y, X_valid, y_valid, preprocessings.decon_set())
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    model_config = {'batch_size': 500, 'epoch': 20000, 'verbose': 0, 'patience': 1000, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-6, "max_lr": 1e-3, "cycle_length": 256}
    model = Sequential()
    model.add(Input(shape=X_train.shape[1:]))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(units=2048, activation="relu"))
    model.add(Dense(units=1024, activation="relu"))
    model.add(Dense(units=512, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(units=256, activation="relu"))
    model.add(Dense(units=512, activation="relu"))
    model.add(Dense(units=1024, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(units=256, activation="relu"))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=32, activation="relu"))
    model.add(Dense(units=16, activation="sigmoid"))
    model.add(Dense(units=1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="mse", metrics=["mse", "mae"])

    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    totalParams = trainableParams + nonTrainableParams
    print("--- Trainable:", trainableParams, "- untrainable:", nonTrainableParams, ">", totalParams,)

    auto_save = Auto_Save("weights.h5", X_valid.shape)
    early_stop = EarlyStopping(monitor="val_loss", patience=model_config["patience"], verbose=1, mode="min", min_delta=0)
    lrScheduler = LearningRateScheduler(get_clr(model_config))
    model.fit(X_train, y_train, batch_size=model_config['batch_size'], epochs=model_config['epoch'], verbose=0, validation_data=(X_test, y_test), callbacks=[auto_save, early_stop, lrScheduler])

    y_pred = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred.reshape((1, -1)))
    y_pred = y_pred.reshape((-1,))

    print(y_pred.shape, y_valid.shape)
    datasheet = {
        "dataset": dataset_name,
        "RMSE": str(mean_squared_error(y_valid, y_pred, squared=False)),
        "MAPE": str(mean_absolute_percentage_error(y_valid, y_pred)),
        "R2": str(r2_score(y_valid, y_pred)),
        "MAE": str(mean_absolute_error(y_valid, y_pred)),
        "MSE": str(mean_squared_error(y_valid, y_pred, squared=True)),
        "MedAE": str(median_absolute_error(y_valid, y_pred)),
        "EVS": str(explained_variance_score(y_valid, y_pred)),
    }

    print(json.dumps(datasheet, indent=4))
    with open(f"/lus/home/PERSO/grp_dcros/dcros/scratch/gbeurier/{dataset_name}_sheet.json", "w") as f:
        json.dump(datasheet, f, indent=4)
    np.savetxt(f"/lus/home/PERSO/grp_dcros/dcros/scratch/gbeurier/{dataset_name}_ypred.csv", y_pred, delimiter=";")


if __name__ == "__main__":
    main()
