import os.path
import sys
from xgboost import XGBRegressor
from sklearn.cross_decomposition import PLSRegression
from pinard import augmentation, model_selection
import pinard.preprocessing as pp
import preprocessings
import logging
from pathlib import Path
from preprocessings import preprocessing_list

from benchmark_loop import benchmark_dataset
import regressors

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
tf.keras.mixed_precision.set_global_policy("mixed_float16")


logging.basicConfig(
    level=logging.WARNING,
    format="'%(name)s - %(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)


rootdir = Path('data/regression')
folder_list = [f for f in rootdir.glob('**/*') if f.is_dir()]

SEED = ord('D') + 31373

# tf.keras.utils.set_random_seed(SEED)
# tf.config.experimental.enable_op_determinism()


def str_to_class(classname):
    return getattr(sys.modules['pinard.preprocessing'], classname)

# print(str_to_class('SavitzkyGolay'))


split_configs = [
    None,
    # {'test_size':None, 'method':"random", 'random_state':SEED},
    # {'test_size':None, 'method':"stratified", 'random_state':SEED, 'n_bins':5},
    # {'test_size':0.25, 'method':"spxy", 'random_state':SEED, 'metric':"euclidean", 'pca_components':250},
]

augmentations = [
    None,
    # [(6, augmentation.Rotate_Translate())],
    # [(3, augmentation.Rotate_Translate()),(2, augmentation.Random_X_Operation()),(1, augmentation.Random_Spline_Addition())],
    # [(3, augmentation.Rotate_Translate()),(2, augmentation.Random_X_Operation()),(2, augmentation.Random_Spline_Addition())],
]

preprocessings_list = [
    # None,
    # [("id", pp.IdentityTransformer())],
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

    ("small_set", preprocessings.small_set()),
    ("simple", [("id", pp.IdentityTransformer()), ('haar', pp.Haar()), ('savgol', pp.SavitzkyGolay())]),
    ("transf_set", preprocessings.transf_set()),
    ("bacon_set", preprocessings.bacon_set()),
    ("decon_set", preprocessings.decon_set()),
    # preprocessings.decon_set(),
    # preprocessings.optimal_set_2D(),
    # preprocessings.fat_set(),
]

# for p in preprocessings_list:
#     if isinstance(p, tuple):
#         h = str(hash(frozenset(p[1])))[0:5]
#         print(h)
#     else:
#         h = str(hash(frozenset(p)))[0:5]
#         print(h)


cv_configs = [
    None,
    # {'n_splits':5, 'n_repeats':4},
    # {'n_splits':4, 'n_repeats':2},
    # {'n_splits':3, 'n_repeats':1},
]

# import os
# folder = "data/regression"
# folder = "data/Paprica_2D"
folder = "data/_RefSet"


def get_dataset_list(path, filter=""):
    datasets = []
    for r, d, _ in os.walk(path):
        for folder in d:
            path = os.path.join(r, folder)
            if os.path.isdir(path):
                if str(path).lower()[13] >= filter:
                    datasets.append(str(path))
                # break
    return datasets


folders = get_dataset_list(folder)
# folders = folders[0:1]
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

    # (regressors.Transformer(), {'batch_size': 1, 'epoch': 1000, 'verbose': 0, 'patience': 150, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-6, "max_lr": 1e-2, "cycle_length": 32}),
    # (regressors.Transformer(), {'batch_size':15, 'epoch':400, 'verbose':0, 'patience':60, 'optimizer':'adam', 'loss':'mse'}),
    # (regressors.Transformer_VG(), {'batch_size':15, 'epoch':400, 'verbose':0, 'patience':30, 'optimizer':'adam', 'loss':'mse'}),
    # (regressors.Transformer_NIRS(), {'batch_size':15, 'epoch':400, 'verbose':0, 'patience':30, 'optimizer':'adam', 'loss':'mse'}),


    # (regressors.FFT_Conv(), {'batch_size':1, 'epoch':10000, 'verbose':0, 'patience':300, 'optimizer':'adam', 'loss':'mse'}),
    # (regressors.ResNetV2(), {'batch_size':1, 'epoch':10000, 'verbose':0, 'patience':300, 'optimizer':'adam', 'loss':'mse'}),
    # (regressors.XCeption1D(), {'batch_size':1, 'epoch':10000, 'verbose':0, 'patience':600, 'optimizer':'adam', 'loss':'mse'}),


    # (regressors.Decon_SepPo(), {'batch_size':50, 'epoch':10000, 'verbose':0, 'patience':1000, 'optimizer':'adam', 'loss':'mse'}),
    (regressors.Decon(), {'batch_size': 100, 'epoch': 20000, 'verbose': 0, 'patience': 400, 'optimizer': 'adam', 'loss': 'mse', "min_lr": 1e-6, "max_lr": 1e-2, "cycle_length": 32}),
    # (regressors.Decon_Sep(), {'batch_size':100, 'epoch':20, 'verbose':2, 'patience':500, 'optimizer':'adam', 'loss':'mse'}),
    # (regressors.Decon_SepPo(), {'batch_size':100, 'epoch':20000, 'verbose':2, 'patience':500, 'optimizer':'adam', 'loss':'mse'}),
    # (regressors.Decon_Sep_VG(), {'batch_size':100, 'epoch':20000, 'verbose':2, 'patience':500, 'optimizer':'adam', 'loss':'mse'}),
    # (regressors.MLP(), {'batch_size':1000, 'epoch':20000, 'verbose':0, 'patience':2000, 'optimizer':'adam', 'loss':'mse'}),
    # (regressors.Bacon(), {'batch_size':100, 'epoch':10, 'verbose':0, 'patience':10, 'optimizer':'adam', 'loss':'mse'}, True),
]


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
# for i in range(1,50,10):
#     models.append( (regressors.ML_Regressor(XGBRegressor, name=f"XGB_{i}"), {"n_estimators":i, "max_depth":max(30, int(i/2)), "seed":SEED}) )

# models.append((regressors.Decon(), {'batch_size':100, 'epoch':20000, 'verbose':0, 'patience':400, 'optimizer':'adam', 'loss':'mse'}))
# benchmark_size = len(folders) * len(split_configs) * len_cv_configs * len(augmentations) * len(preprocessings_list) * len(models)
# print("Benchmarking", benchmark_size, "runs.")
# benchmark_dataset(folders[:-1], split_configs, cv_configs, augmentations, preprocessings_list, models, SEED) #, resampling='resample', resample_size=2048) #bins=5)


# benchmark_dataset([folders[12]], split_configs, cv_configs, augmentations, preprocessings_list, models, SEED)
for folder in folders[:-1]:
    print("adding folder", folder)
    benchmark_dataset([folder], split_configs, cv_configs, augmentations, preprocessings_list, models, SEED)


# for folder in folders[:-1]:
#     print("adding folder", folder)
#     benchmark_dataset([folders[i]], split_configs, cv_configs, augmentations, preprocessings_list, [(regressors.CONV_LSTM(), {'batch_size':10, 'epoch':1, 'verbose':0, 'patience':2000, 'optimizer':'adam', 'loss':'mse'}),], SEED)


# import threading
# import queue

# def worker(q):
#     while True:
#         item = q.get()
#         if item is None:
#             break
#         # process item
#         # print(f'Processing {item}')
#         benchmark_dataset(*item)
#         q.task_done()

# q = queue.Queue()
# num_worker_threads = 1
# threads = []
# for i in range(num_worker_threads):
#     t = threading.Thread(target=worker, args=(q,))
#     t.start()
#     threads.append(t)

# # add items to the queue
# for folder in folders[:-1]:
#     print("adding folder", folder)
#     q.put(([folder], split_configs, cv_configs, augmentations, preprocessings_list, models, SEED))

# # block until all tasks are done
# q.join()

# # stop workers
# for i in range(num_worker_threads):
#     q.put(None)


# for t in threads:
#     t.join()


print("Benchmarking finished.")


# benchmark_dataset(folders, split_configs, cv_configs, augmentations, preprocessings_list, models, SEED) #, resampling='resample', resample_size=2048) #bins=5)


# benchmark_dataset(folders, split_configs, cv_configs, augmentations, preprocessings_list, models, SEED, resampling='crop', resample_size=2150)


# for folder in folder_list:
# # print(ord(str(folder)[17]), ord('A'), ord('M'))
# if ord(str(folder)[16]) < ord("L") or ord(str(folder)[16]) > ord("M"):
#     continue
# benchmark_dataset(folder, SEED, preprocessing_list(), 20, augment=False)
