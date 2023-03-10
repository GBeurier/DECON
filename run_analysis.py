from pathlib import Path
from preprocessings import preprocessing_list

from benchmark_loop import benchmark_dataset

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
tf.keras.mixed_precision.set_global_policy("mixed_float16")

rootdir = Path('data/regression')
folder_list = [f for f in rootdir.glob('**/*') if f.is_dir()]

SEED = ord('D') + 31373

# tf.keras.utils.set_random_seed(SEED)
# tf.config.experimental.enable_op_determinism()


import preprocessings
import regressors
import pinard.preprocessing as pp
from pinard import augmentation, model_selection
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor
import sys
import os.path

def str_to_class(classname):
    return getattr(sys.modules['pinard.preprocessing'], classname)

# print(str_to_class('SavitzkyGolay'))




def get_dataset_list(path):
    datasets = []
    for r, d, _ in os.walk(path):
        for folder in d:
            path = os.path.join(r, folder)
            if os.path.isdir(path):
                # if len(datasets) < 3:
                datasets.append(str(path))
    return datasets

split_configs = [
    None,
    # {'test_size':None, 'method':"random", 'random_state':SEED},
    # {'test_size':None, 'method':"stratified", 'random_state':SEED, 'n_bins':5},
    # {'test_size':0.25, 'method':"spxy", 'random_state':SEED, 'metric':"euclidean", 'pca_components':250},
]

augmentations = [
    None,
    # [(6, augmentation.Rotate_Translate())],
    # [(3, augmentation.Rotate_Translate()),(2, augmentation.Random_X_Operation()),(1, augmentation.Random_Spline_Addition()),],
    # [(3, augmentation.Rotate_Translate()),(2, augmentation.Random_X_Operation()),(2, augmentation.Random_Spline_Addition()),]
]

preprocessings_list = [
    # None,
    # preprocessings.id_preprocessing(),
    [("id", pp.IdentityTransformer()), ('haar', pp.Haar()), ('savgol', pp.SavitzkyGolay())],
    # preprocessings.decon_set(),
    # preprocessings.bacon_set(),
    # preprocessings.small_set(),
    preprocessings.transf_set(),
    # preprocessings.optimal_set_2D(),
    # preprocessings.fat_set(),
]



cv_configs = [
    None,
    # {'n_splits':5, 'n_repeats':4},
    # {'n_splits':4, 'n_repeats':2},
    {'n_splits':4, 'n_repeats':1},
]

# import os
folder = "data/regression"
folders = get_dataset_list(folder)
print(folders)
# folders = ["data/regression/Cassava_TBC_3556_Davrieux_RMSE1.02"]

len_cv_configs = 0
for c in cv_configs:
    if c == None:
        len_cv_configs += 1
    else:
        len_cv_configs += (c['n_splits'] * c['n_repeats'])

models = [
    # (regressors.ML_Regressor(XGBRegressor), {"n_estimators":200, "max_depth":50, "seed":SEED}),
    # (regressors.ML_Regressor(PLSRegression), {"n_components":50}),
    (regressors.Transformer_VG(), {'batch_size':50, 'epoch':10000, 'verbose':0, 'patience':250, 'optimizer':'adam', 'loss':'mse'}),
    # (regressors.Decon_SepPo(), {'batch_size':50, 'epoch':10000, 'verbose':0, 'patience':1000, 'optimizer':'adam', 'loss':'mse'}),
    # (regressors.FFT_Conv(), {'batch_size':500, 'epoch':20000, 'verbose':0, 'patience':1000, 'optimizer':'adam', 'loss':'mse'}),
    # (regressors.Decon(), {'batch_size':50, 'epoch':30000, 'verbose':0, 'patience':2000, 'optimizer':'adam', 'loss':'mse'}),
    # (regressors.Decon_Sep(), {'batch_size':5, 'epoch':30000, 'verbose':0, 'patience':100, 'optimizer':'adam', 'loss':'mse'}),
    # (regressors.ResNetV2(), {'batch_size':200, 'epoch':20000, 'verbose':0, 'patience':300, 'optimizer':'adam', 'loss':'mse'}),
    # (regressors.MLP(), {'batch_size':1000, 'epoch':20000, 'verbose':0, 'patience':2000, 'optimizer':'adam', 'loss':'mse'}),
    # (regressors.CONV_LSTM(), {'batch_size':1000, 'epoch':20000, 'verbose':0, 'patience':2000, 'optimizer':'adam', 'loss':'mse'}),
    # (regressors.XCeption1D(), {'batch_size':500, 'epoch':10000, 'verbose':0, 'patience':1200, 'optimizer':'adam', 'loss':'mse'}),
    # (regressors.Transformer(), {'batch_size':10, 'epoch':300, 'verbose':0, 'patience':30, 'optimizer':'Adam', 'loss':'mse'}),
]

benchmark_size = len(folders) * len(split_configs) * len_cv_configs * len(augmentations) * len(preprocessings_list) * len(models)
print("Benchmarking", benchmark_size, "runs.")


# benchmark_dataset(folders, split_configs, cv_configs, augmentations, preprocessings_list, models, SEED, resampling='crop', resample_size=2150)
benchmark_dataset(folders, split_configs, cv_configs, augmentations, preprocessings_list, models, SEED) #, resampling='resample', resample_size=2048) #bins=5)
# benchmark_dataset(folders, split_configs, cv_configs, augmentations, preprocessings_list, models, 42)
# benchmark_dataset(folders, split_configs, cv_configs, augmentations, preprocessings_list, models, 666)
# benchmark_dataset(folders, split_configs, cv_configs, augmentations, preprocessings_list, models, 1234567890)


# for folder in folder_list:
    # # print(ord(str(folder)[17]), ord('A'), ord('M'))
    # if ord(str(folder)[16]) < ord("L") or ord(str(folder)[16]) > ord("M"):
    #     continue
    # benchmark_dataset(folder, SEED, preprocessing_list(), 20, augment=False)