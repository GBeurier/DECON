import os.path
import sys
import pinard.preprocessing as pp
import regressors
import preprocessings
import logging
from pathlib import Path
from preprocessings import preprocessing_list

from benchmark_loop import benchmark_dataset
import random
import tensorflow as tf


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
    if c == None:
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

# (regressors.ResNetV2(), {'batch_size': 200, 'epoch': 20000, 'verbose': 0, 'patience': 300,
#  'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-6, "max_lr": 1e-3, "cycle_length": 256}),


# from lwpls import LWPLS
# from regressors import NonlinearPLSRegressor

# for i in range(5,150,5):
#     models. append(
#         (regressors.ML_Regressor(NonlinearPLSRegressor, name=f"NL_RBF_PLS_{i}"), {"n_components":i, "poly_degree":2, "gamma":0.1})
#     )
#     models. append(
#         (regressors.ML_Regressor(PLSRegression, name=f"PLS_{i}"), {"n_components":i})
#     )

# models.append(
#     (regressors.ML_Regressor(LWPLS, name=f"LWPLS_0-05_45"), {"max_component_number":45, "lambda_in_similarity":0.05})
# )

# for i in range(10,100,50):
#     models.append(
#         (regressors.ML_Regressor(LWPLS, name=f"LWPLS_0-1_{i}"), {"max_component_number":i, "lambda_in_similarity":0.05})
#     )
# models.append(
#     (regressors.ML_Regressor(LWPLS, name=f"LWPLS_0-5_{i}"), {"max_component_number":i, "lambda_in_similarity":0.5})
# )
# models.append( (regressors.ML_Regressor(PLSRegression, name=f"PLS_5"), {"n_components":5}) )
# for i in range(1,20,1):
#     models.append( (regressors.ML_Regressor(PLSRegression, name=f"PLS_{i}"), {"n_components":i}) )
# for i in range(21,120,3):
#     models.append( (regressors.ML_Regressor(PLSRegression, name=f"PLS_{i}"), {"n_components":i}) )


def main():
    # Read the folder path parameter
    folder = sys.argv[1]
    # models_index = int(sys.argv[2])
    print(f"Processing folder: {folder}")

    benchmark_size = len(split_configs) * len_cv_configs * len(augmentations) * len(preprocessings_list) * len(models)
    print("Benchmarking", benchmark_size, "runs.")

    tf.get_logger().setLevel("INFO")

    logging.basicConfig(
        level=logging.INFO,
        format="'%(name)s - %(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(folder + ".log"),
            logging.StreamHandler()
        ]
    )

    print("*" * 20, folder, "*" * 20)
    SEED = ord('D') + 31373
    # if models_index == 2:
    #     SEED = -1
    benchmark_dataset([folder], split_configs, cv_configs, augmentations, preprocessings_list, models[0], SEED)  # , resampling='resample', resample_size=2048) #bins=5)


if __name__ == "__main__":
    main()


# for folder in folders:


# benchmark_dataset(folders, split_configs, cv_configs, augmentations, preprocessings_list, models, SEED, resampling='crop', resample_size=2150)


# for folder in folder_list:
    # # print(ord(str(folder)[17]), ord('A'), ord('M'))
    # if ord(str(folder)[16]) < ord("L") or ord(str(folder)[16]) > ord("M"):
    #     continue
    # benchmark_dataset(folder, SEED, preprocessing_list(), 20, augment=False)
