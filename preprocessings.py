from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler

from pinard import preprocessing as pp
from pinard.sklearn import FeatureAugmentation


def id_preprocessing():
    return [("id", pp.IdentityTransformer())]


def savgol_only():
    return [("savgol", pp.SavitzkyGolay())]


def haar_only():
    return [("haar", pp.Wavelet("haar"))]


def bacon_set():
    preprocessing = [
        ("id", pp.IdentityTransformer()),
        ("savgol1", pp.SavitzkyGolay(window_length=17, polyorder=2, deriv=2)),
        ("savgol2", pp.SavitzkyGolay(window_length=5, polyorder=2)),
        ("gaussian1", pp.Gaussian(order=1, sigma=2)),
        ("gaussian2", pp.Gaussian(order=2, sigma=1)),
        ("gaussian3", pp.Gaussian(order=0, sigma=2)),
        ("baseline", pp.StandardNormalVariate()),
        ("msc", pp.MultiplicativeScatterCorrection(scale=False)),
        ("detrend", pp.Detrend()),
        ("derivate", pp.Derivate(2, 1)),
        ("dv2", pp.Derivate(2, 1)),
        ("haar", pp.Wavelet("haar")),
    ]
    return preprocessing


def decon_set():
    preprocessing = [
        ("id", pp.IdentityTransformer()),
        ("detrend", pp.Detrend()),
        ("msc", pp.MultiplicativeScatterCorrection(scale=False)),
        ("dv1", pp.Derivate(1, 1)),
        ("dv2", pp.Derivate(2, 1)),
        ("dv3", pp.Derivate(2, 2)),
        ("baseline", pp.StandardNormalVariate()),
        ("baseline*savgol", Pipeline([("_sg1", pp.StandardNormalVariate()), ("_sg2", pp.SavitzkyGolay())])),
        ("baseline*gaussian1", Pipeline([("_sg1", pp.StandardNormalVariate()), ("g2", pp.Gaussian(order=1, sigma=2))])),
        ("baseline*gaussian2", Pipeline([("_sg1", pp.StandardNormalVariate()), ("g2", pp.Gaussian(order=2, sigma=1))])),
        ("baseline*haar", Pipeline([("_sg1", pp.StandardNormalVariate()), ("_sg2", pp.Wavelet("haar"))])),
        ("savgol", pp.SavitzkyGolay()),
        ("savgol*gaussian1", Pipeline([("_sg1", pp.SavitzkyGolay()), ("g2", pp.Gaussian(order=1, sigma=2))])),
        ("savgol*gaussian2", Pipeline([("_sg1", pp.SavitzkyGolay()), ("g2", pp.Gaussian(order=2, sigma=1))])),
        ("savgol*dv2", Pipeline([("_sg1", pp.SavitzkyGolay()), ("g2", pp.Derivate(2, 1))])),
        ("savgol*dv3", Pipeline([("_sg1", pp.SavitzkyGolay()), ("g2", pp.Derivate(1, 2))])),
        ("gaussian1", pp.Gaussian(order=1, sigma=2)),
        ("gaussian2*savgol", Pipeline([("_g2", pp.Gaussian(order=1, sigma=2)), ("_sg4", pp.SavitzkyGolay())])),
        ("gaussian2", pp.Gaussian(order=2, sigma=1)),
        ("haar", pp.Wavelet("haar")),
        ("haar*gaussian2", Pipeline([("_haar2", pp.Wavelet("haar")), ("g2", pp.Gaussian(order=2, sigma=1))])),
        ("coif3", pp.Wavelet("coif3")),
    ]
    return preprocessing


def transf_set():
    preprocessing = [
        ("id", pp.IdentityTransformer()),
        ("baseline", pp.StandardNormalVariate()),
        ("savgol", pp.SavitzkyGolay()),
        ("gaussian1", pp.Gaussian(order=1, sigma=2)),
        ("gaussian2", pp.Gaussian(order=2, sigma=1)),
        ("haar", pp.Wavelet("haar")),
        ("coif3", pp.Wavelet("coif3")),
        ("detrend", pp.Detrend()),
        ("msc", pp.MultiplicativeScatterCorrection(scale=False)),
        ("dv1", pp.Derivate(1, 1)),
        ("dv2", pp.Derivate(2, 1)),
        ("dv3", pp.Derivate(2, 2)),
    ]
    return preprocessing


def small_set():
    preprocessing = [
        ("id", pp.IdentityTransformer()),
        ("baseline", pp.StandardNormalVariate()),
        ("savgol", pp.SavitzkyGolay()),
        ("haar", pp.Wavelet("haar")),
        ("detrend", pp.Detrend()),
    ]
    return preprocessing


def dumb_set():
    pp_list = transf_set()

    preprocessings = []
    for i in pp_list:
        for j in pp_list:
            new_pp = (i[0] + "_" + j[0], Pipeline([(i[0] + "_0", i[1]), (j[0] + "_1", j[1])]))
            preprocessings.append(new_pp)
    for i in pp_list:
        preprocessings.append(i)

    return preprocessings


def dumb_and_dumber_set():
    pp_list = [
        ("id", pp.IdentityTransformer()),
        ("baseline", pp.StandardNormalVariate()),
        ("savgol", pp.SavitzkyGolay()),
        ("gaussian1", pp.Gaussian(order=1, sigma=2)),
        ("gaussian2", pp.Gaussian(order=2, sigma=1)),
        ("haar", pp.Wavelet("haar")),
        ("coif3", pp.Wavelet("coif3")),
        # ("detrend", pp.Detrend()),
        ("msc", pp.MultiplicativeScatterCorrection(scale=False)),
        ("dv1", pp.Derivate(1, 1)),
        ("dv2", pp.Derivate(2, 1)),
        ("dv3", pp.Derivate(2, 2)),
    ]

    preprocessings = []
    for i in pp_list:
        for j in pp_list:
            for k in pp_list:
                new_pp = (i[0] + "_" + j[0]+ "_" + k[0], Pipeline([(i[0] + "_0", i[1]), (j[0] + "_1", j[1]), (k[0] + "_2", k[1])]))
                preprocessings.append(new_pp)
    for i in pp_list:
        preprocessings.append(i)

    return preprocessings


def optimal_set_2D():
    optimal = ['dv3', 'dv2', 'dv1', 'msc', 'detrend', 'coif3', 'haar', 'gaussian2', 'gaussian1', 'savgol', 'baseline', 'id', 'dv3_baseline', 'dv3_id', 'dv2_dv3', 'dv2_dv2', 'dv2_dv1', 'dv2_msc', 'dv2_detrend', 'dv2_coif3', 'dv2_haar', 'dv2_gaussian2', 'dv2_gaussian1', 'dv2_savgol', 'dv2_baseline', 'dv2_id', 'dv1_dv3', 'dv1_dv2', 'dv1_dv1', 'dv1_msc', 'dv1_detrend', 'msc_gaussian1', 'msc_savgol', 'msc_baseline', 'msc_id', 'detrend_dv2', 'detrend_dv1', 'detrend_msc', 'detrend_detrend', 'detrend_coif3', 'detrend_haar', 'detrend_gaussian2', 'detrend_gaussian1', 'detrend_savgol', 'detrend_baseline', 'detrend_id', 'coif3_dv3', 'coif3_dv2', 'coif3_dv1', 'coif3_msc', 'coif3_detrend', 'coif3_coif3', 'coif3_haar', 'coif3_gaussian2', 'coif3_gaussian1', 'coif3_savgol', 'coif3_baseline', 'coif3_id', 'haar_dv3', 'haar_dv2', 'haar_dv1', 'haar_msc', 'haar_detrend', 'haar_coif3', 'haar_haar', 'haar_gaussian2', 'haar_gaussian1', 'haar_savgol', 'haar_baseline', 'haar_id', 'gaussian2_dv3', 'gaussian2_dv2', 'gaussian2_dv1', 'gaussian2_msc', 'gaussian2_detrend', 'gaussian2_coif3', 'gaussian2_haar', 'gaussian2_gaussian2', 'gaussian2_gaussian1', 'gaussian2_savgol', 'gaussian2_baseline', 'gaussian2_id', 'gaussian1_dv3', 'gaussian1_dv2', 'gaussian1_dv1', 'gaussian1_msc', 'gaussian1_detrend', 'gaussian1_coif3', 'baseline_dv3', 'baseline_dv2', 'baseline_dv1', 'baseline_detrend', 'baseline_coif3', 'baseline_haar', 'baseline_gaussian2', 'baseline_gaussian1', 'baseline_savgol', 'baseline_baseline', 'baseline_id', 'id_dv3', 'id_dv2', 'id_dv1', 'id_msc', 'id_detrend', 'id_coif3', 'id_haar', 'id_gaussian2', 'id_gaussian1', 'id_savgol', 'id_baseline', 'id']
    src_set = dumb_set()
    optimal_set = []
    for it in src_set:
        if it[0] in optimal:
            optimal_set.append(it)
    return optimal_set


def transform_test_data(preprocessing, X_train, y_train, X_test, y_test, type="augmentation", classification_mode=False):
    if classification_mode:
        y_scaler = None
        y_valid = y_test
    else:
        y_scaler = MinMaxScaler()
        y_scaler.fit(y_train.reshape((-1, 1)))
        y_valid = y_scaler.transform(y_test.reshape((-1, 1)))

    if preprocessing is None:
        transformer_pipeline = Pipeline([
                ("scaler", MinMaxScaler()),
        ])
    elif type == "augmentation":
        transformer_pipeline = Pipeline([
                ("scaler", MinMaxScaler()),
                ("preprocessing", FeatureAugmentation(preprocessing)),
        ])
    else:
        transformer_pipeline = Pipeline([
                ("scaler", MinMaxScaler()),
                ("preprocessing", FeatureUnion(preprocessing)),
        ])

    transformer_pipeline.fit(X_train)
    X_valid = transformer_pipeline.transform(X_test)

    return X_valid, y_valid, transformer_pipeline, y_scaler


def preprocessing_list():
    # return [id_preprocessing, savgol, haar, bacon_set, decon_set]
    # return [id_preprocessing, transf_set, decon_set]
    # return [decon_set, dumb_set]
    # return [transf_set]
    return [decon_set]
    # return [id_preprocessing]


def fat_set():
    fat_list = ['detrend', 'coif3', 'haar', 'gaussian2', 'gaussian1', 'savgol', 'baseline', 'id', 'dv3_dv3_dv3', 'dv3_dv3_dv2', 'dv3_dv3_dv1', 'dv3_dv3_msc', 'dv3_dv3_detrend', 'dv3_dv3_coif3', 'dv3_dv3_haar', 'dv3_dv3_gaussian2', 'dv3_dv3_gaussian1', 'dv3_dv3_savgol', 'dv3_dv3_baseline', 'dv3_dv3_id', 'dv3_dv2_dv3', 'dv3_dv2_dv2', 'dv3_dv2_dv1', 'dv3_dv2_msc', 'dv3_dv2_detrend', 'dv3_dv2_coif3', 'dv3_dv2_haar', 'dv3_dv2_gaussian2', 'dv2_dv2_dv1', 'dv2_dv2_msc', 'dv2_dv2_detrend', 'dv2_dv2_coif3', 'dv2_dv2_haar', 'dv2_dv2_gaussian2', 'dv2_dv2_gaussian1', 'dv2_dv2_savgol', 'dv2_dv2_baseline', 'dv2_dv2_id', 'dv2_dv1_dv3', 'dv2_dv1_dv2', 'dv2_dv1_dv1', 'dv2_dv1_msc', 'dv2_dv1_detrend', 'dv2_dv1_coif3', 'dv2_dv1_haar', 'dv2_dv1_gaussian2', 'dv2_dv1_gaussian1', 'dv2_dv1_savgol', 'dv2_dv1_baseline', 'dv2_dv1_id', 'dv2_msc_dv3', 'dv2_msc_dv2', 'dv2_msc_dv1', 'dv2_msc_msc', 'dv2_msc_detrend', 'dv2_msc_coif3', 'dv2_msc_haar', 'dv2_msc_gaussian2', 'dv2_msc_gaussian1', 'dv2_msc_savgol', 'dv2_msc_baseline', 'dv2_msc_id', 'dv2_detrend_dv3', 'dv2_detrend_dv2', 'dv2_detrend_dv1', 'dv2_detrend_msc', 'dv2_detrend_detrend', 'dv2_detrend_coif3', 'dv2_detrend_haar', 'dv2_detrend_gaussian2', 'dv2_detrend_gaussian1', 'dv2_detrend_savgol', 'dv2_detrend_baseline', 'dv2_detrend_id', 'dv2_coif3_dv3', 'dv2_coif3_dv2', 'dv2_coif3_dv1', 'dv2_coif3_msc', 'dv2_coif3_detrend', 'dv2_coif3_coif3', 'dv2_coif3_haar', 'dv2_coif3_gaussian2', 'dv2_coif3_gaussian1', 'dv2_coif3_savgol', 'dv2_coif3_baseline', 'dv2_coif3_id', 'dv2_haar_dv3', 'dv2_haar_dv2', 'dv2_haar_dv1', 'dv2_haar_msc', 'dv2_haar_coif3', 'dv2_haar_id', 'dv2_gaussian2_dv2', 'dv2_gaussian2_msc', 'dv2_gaussian2_coif3', 'dv2_gaussian2_gaussian2', 'dv2_gaussian2_savgol', 'dv2_gaussian2_id', 'dv2_gaussian1_dv3', 'dv2_gaussian1_dv2', 'dv2_gaussian1_dv1', 'dv2_gaussian1_msc', 'dv2_gaussian1_detrend', 'dv2_gaussian1_coif3', 'dv2_gaussian1_haar', 'dv2_gaussian1_gaussian2', 'dv2_gaussian1_gaussian1', 'dv2_gaussian1_savgol', 'dv2_gaussian1_baseline', 'dv2_gaussian1_id', 'dv2_savgol_dv3', 'dv2_savgol_dv2', 'dv2_savgol_dv1', 'dv2_savgol_msc', 'dv2_savgol_coif3', 'dv2_savgol_gaussian2', 'dv2_savgol_gaussian1', 'dv2_savgol_savgol', 'dv2_savgol_id', 'dv2_baseline_dv2', 'dv2_baseline_savgol', 'dv2_id_msc', 'dv2_id_detrend', 'dv2_id_coif3', 'dv2_id_haar', 'dv2_id_baseline', 'dv2_id_id', 'dv1_dv3_dv3', 'dv1_dv3_dv2', 'dv1_dv3_dv1', 'dv1_dv3_msc', 'dv1_dv3_detrend', 'dv1_dv3_coif3', 'dv1_dv3_haar', 'dv1_dv3_gaussian2', 'dv1_dv3_gaussian1', 'dv1_dv3_savgol', 'dv1_dv3_baseline', 'dv1_dv3_id', 'dv1_dv2_dv3', 'dv1_dv2_dv2', 'dv1_dv2_dv1', 'dv1_dv2_msc', 'dv1_dv2_detrend', 'dv1_dv2_coif3', 'dv1_dv2_haar', 'dv1_dv2_gaussian2', 'dv1_dv2_gaussian1', 'dv1_dv2_savgol', 'dv1_dv2_baseline', 'dv1_dv2_id', 'dv1_dv1_dv3', 'dv1_dv1_detrend', 'dv1_msc_detrend', 'dv1_msc_coif3', 'dv1_msc_gaussian2', 'dv1_detrend_haar', 'dv1_detrend_gaussian2', 'dv1_detrend_gaussian1', 'dv1_detrend_savgol', 'dv1_detrend_baseline', 'dv1_detrend_id', 'dv1_coif3_dv3', 'dv1_coif3_dv2', 'dv1_coif3_dv1', 'dv1_coif3_msc', 'dv1_coif3_detrend', 'dv1_coif3_gaussian2', 'dv1_haar_detrend', 'dv1_haar_haar', 'dv1_haar_gaussian2', 'dv1_haar_savgol', 'dv1_haar_baseline', 'dv1_haar_id', 'dv1_gaussian2_dv3', 'dv1_gaussian2_dv2', 'dv1_gaussian2_dv1', 'dv1_gaussian2_msc', 'dv1_gaussian2_detrend', 'dv1_gaussian2_coif3', 'dv1_gaussian2_haar', 'dv1_gaussian2_gaussian2', 'dv1_gaussian2_gaussian1', 'dv1_gaussian2_savgol', 'dv1_gaussian2_baseline', 'dv1_gaussian2_id', 'dv1_gaussian1_dv3', 'dv1_gaussian1_dv2', 'dv1_gaussian1_dv1', 'dv1_gaussian1_msc', 'dv1_gaussian1_detrend', 'dv1_gaussian1_coif3', 'dv1_gaussian1_haar', 'dv1_gaussian1_gaussian2', 'dv1_gaussian1_gaussian1', 'dv1_gaussian1_savgol', 'dv1_gaussian1_baseline', 'dv1_gaussian1_id', 'dv1_savgol_dv3', 'dv1_savgol_dv2', 'dv1_savgol_dv1', 'dv1_savgol_msc', 'dv1_savgol_detrend', 'dv1_savgol_coif3', 'dv1_savgol_haar', 'dv1_savgol_gaussian2', 'dv1_savgol_gaussian1', 'dv1_savgol_id', 'dv1_baseline_dv3', 'dv1_baseline_dv2', 'dv1_baseline_dv1', 'dv1_baseline_msc', 'dv1_baseline_detrend', 'dv1_baseline_coif3', 'dv1_baseline_haar', 'dv1_baseline_gaussian2', 'dv1_baseline_gaussian1', 'dv1_baseline_savgol', 'dv1_baseline_baseline', 'dv1_baseline_id', 'dv1_id_dv3', 'dv1_id_dv2', 'dv1_id_dv1', 'dv1_id_msc', 'dv1_id_detrend', 'dv1_id_haar', 'msc_dv3_dv1', 'msc_dv3_msc', 'msc_dv3_detrend', 'msc_dv3_coif3', 'msc_dv3_haar', 'msc_dv3_gaussian2', 'msc_dv3_gaussian1', 'msc_dv3_savgol', 'msc_dv3_baseline', 'msc_dv3_id', 'msc_dv2_dv3', 'msc_dv2_dv2', 'msc_dv2_dv1', 'msc_dv2_msc', 'msc_dv2_detrend', 'msc_dv2_coif3', 'msc_dv2_haar', 'msc_dv2_gaussian2', 'msc_dv2_gaussian1', 'msc_dv2_savgol', 'msc_dv2_baseline', 'msc_dv2_id', 'msc_dv1_dv3', 'msc_dv1_dv2', 'msc_dv1_dv1', 'msc_dv1_msc', 'msc_dv1_detrend', 'msc_dv1_coif3', 'msc_dv1_haar', 'msc_dv1_gaussian2', 'msc_dv1_gaussian1', 'msc_dv1_savgol', 'msc_dv1_baseline', 'msc_dv1_id', 'msc_msc_dv3', 'msc_msc_dv2', 'msc_msc_dv1', 'msc_msc_msc', 'msc_msc_detrend', 'msc_msc_coif3', 'msc_msc_haar', 'msc_msc_gaussian2', 'msc_msc_gaussian1', 'msc_msc_savgol', 'msc_msc_baseline', 'msc_msc_id', 'msc_detrend_dv3', 'msc_detrend_dv2', 'msc_detrend_dv1', 'msc_detrend_msc', 'msc_detrend_detrend', 'msc_detrend_coif3', 'msc_detrend_haar', 'msc_detrend_gaussian2', 'msc_detrend_gaussian1', 'msc_detrend_savgol', 'msc_detrend_baseline', 'msc_detrend_id', 'msc_coif3_dv3', 'msc_coif3_dv1', 'msc_coif3_msc', 'msc_coif3_coif3', 'msc_coif3_haar', 'msc_coif3_gaussian2', 'msc_coif3_gaussian1', 'msc_coif3_savgol', 'msc_coif3_baseline', 'msc_coif3_id', 'msc_haar_dv3', 'msc_haar_dv2', 'msc_haar_dv1', 'msc_haar_msc', 'msc_haar_detrend', 'msc_haar_coif3', 'msc_haar_haar', 'msc_haar_gaussian2', 'msc_haar_gaussian1', 'msc_haar_savgol', 'msc_haar_baseline', 'msc_haar_id', 'msc_gaussian2_dv3', 'msc_gaussian2_dv2', 'msc_id_id', 'detrend_dv3_dv3', 'detrend_dv3_dv2', 'detrend_dv3_msc', 'detrend_dv3_detrend', 'detrend_dv3_coif3', 'detrend_dv3_haar', 'detrend_dv3_gaussian2', 'detrend_dv3_gaussian1', 'detrend_dv3_savgol', 'detrend_dv3_baseline', 'detrend_dv3_id', 'detrend_dv2_dv3', 'detrend_dv2_dv2', 'detrend_dv2_dv1', 'detrend_dv2_msc', 'detrend_dv2_detrend', 'detrend_dv2_coif3', 'detrend_dv2_haar', 'detrend_dv2_gaussian2', 'detrend_dv2_gaussian1', 'detrend_dv2_savgol', 'detrend_dv2_baseline', 'detrend_msc_dv1', 'detrend_msc_msc', 'detrend_msc_detrend', 'detrend_msc_coif3', 'detrend_msc_haar', 'detrend_msc_gaussian2', 'detrend_msc_gaussian1', 'detrend_msc_savgol', 'detrend_msc_baseline', 'detrend_msc_id', 'detrend_detrend_dv3', 'detrend_detrend_dv2', 'detrend_detrend_dv1', 'detrend_detrend_msc', 'detrend_detrend_detrend', 'detrend_detrend_coif3', 'detrend_detrend_haar', 'detrend_detrend_gaussian2', 'detrend_detrend_gaussian1', 'detrend_detrend_savgol', 'detrend_detrend_baseline', 'detrend_detrend_id', 'detrend_coif3_dv3', 'detrend_coif3_dv2', 'detrend_coif3_dv1', 'detrend_coif3_msc', 'detrend_coif3_detrend', 'detrend_coif3_coif3', 'detrend_coif3_haar', 'detrend_coif3_gaussian2', 'detrend_coif3_gaussian1', 'detrend_coif3_savgol', 'detrend_coif3_baseline', 'detrend_coif3_id', 'detrend_haar_dv3', 'detrend_haar_dv2', 'detrend_haar_dv1', 'detrend_haar_msc', 'detrend_haar_detrend', 'detrend_haar_coif3', 'detrend_haar_haar', 'detrend_haar_gaussian2', 'detrend_haar_gaussian1', 'detrend_haar_savgol', 'detrend_haar_baseline', 'detrend_haar_id', 'detrend_gaussian2_dv3', 'detrend_gaussian2_dv2', 'detrend_gaussian2_dv1', 'detrend_gaussian2_msc', 'detrend_gaussian2_detrend', 'detrend_gaussian2_coif3', 'detrend_gaussian2_haar', 'detrend_gaussian2_gaussian2', 'detrend_gaussian2_gaussian1', 'detrend_gaussian2_savgol', 'detrend_gaussian2_baseline', 'detrend_gaussian2_id', 'detrend_gaussian1_dv3', 'detrend_gaussian1_dv2', 'detrend_gaussian1_dv1', 'detrend_gaussian1_msc', 'detrend_gaussian1_detrend', 'detrend_gaussian1_coif3', 'detrend_gaussian1_haar', 'detrend_gaussian1_gaussian2', 'detrend_gaussian1_gaussian1', 'detrend_gaussian1_savgol', 'detrend_gaussian1_baseline', 'detrend_gaussian1_id', 'detrend_savgol_dv3', 'detrend_savgol_dv2', 'detrend_savgol_dv1', 'detrend_savgol_msc', 'detrend_savgol_detrend', 'detrend_savgol_coif3', 'haar_dv3_gaussian2', 'haar_dv3_gaussian1', 'haar_dv3_savgol', 'haar_dv3_baseline', 'haar_dv3_id', 'haar_dv2_dv3', 'haar_dv2_dv2', 'haar_dv2_dv1', 'haar_dv2_msc', 'haar_dv2_detrend', 'haar_dv2_coif3', 'haar_dv2_haar', 'haar_dv2_gaussian2', 'haar_dv2_gaussian1', 'haar_dv2_savgol', 'haar_dv2_baseline', 'haar_dv2_id', 'haar_dv1_dv3', 'haar_dv1_dv2', 'haar_dv1_dv1', 'haar_dv1_msc', 'haar_dv1_detrend', 'haar_dv1_coif3', 'haar_dv1_haar', 'haar_dv1_gaussian2', 'haar_dv1_gaussian1', 'haar_dv1_savgol', 'haar_dv1_baseline', 'haar_dv1_id', 'haar_msc_dv3', 'haar_msc_dv2', 'haar_msc_dv1', 'haar_msc_msc', 'haar_msc_detrend', 'haar_msc_coif3', 'haar_msc_haar', 'haar_msc_gaussian2', 'haar_msc_gaussian1', 'haar_msc_savgol', 'haar_msc_baseline', 'haar_msc_id', 'haar_detrend_dv2', 'haar_detrend_dv1', 'haar_detrend_msc', 'haar_detrend_gaussian1', 'haar_coif3_dv1', 'haar_coif3_gaussian1', 'haar_coif3_savgol', 'haar_coif3_id', 'haar_haar_dv3', 'haar_haar_dv2', 'haar_haar_dv1', 'haar_haar_msc', 'haar_haar_detrend', 'haar_haar_coif3', 'haar_haar_haar', 'haar_haar_gaussian2', 'haar_haar_gaussian1', 'haar_haar_savgol', 'haar_haar_baseline', 'haar_haar_id', 'haar_gaussian2_dv3', 'haar_gaussian2_dv2', 'haar_gaussian2_dv1', 'haar_gaussian2_msc', 'haar_gaussian2_detrend', 'haar_gaussian2_coif3', 'haar_gaussian2_haar', 'haar_gaussian2_gaussian2', 'haar_gaussian2_gaussian1', 'haar_gaussian2_savgol', 'haar_gaussian2_baseline', 'haar_gaussian2_id', 'haar_gaussian1_dv3', 'haar_gaussian1_dv2', 'haar_gaussian1_dv1', 'haar_gaussian1_msc', 'haar_gaussian1_detrend', 'haar_gaussian1_coif3', 'haar_gaussian1_haar', 'haar_gaussian1_gaussian2', 'haar_gaussian1_gaussian1', 'haar_gaussian1_savgol', 'haar_gaussian1_baseline', 'haar_gaussian1_id', 'haar_savgol_dv3', 'haar_savgol_dv2', 'haar_savgol_dv1', 'haar_savgol_msc', 'haar_savgol_detrend', 'haar_savgol_coif3', 'haar_savgol_haar', 'haar_savgol_gaussian2', 'haar_savgol_gaussian1', 'haar_savgol_savgol', 'haar_savgol_baseline', 'haar_savgol_id', 'haar_baseline_dv3', 'haar_baseline_dv2', 'haar_baseline_dv1', 'haar_baseline_msc', 'haar_baseline_detrend', 'haar_baseline_coif3', 'haar_baseline_haar', 'haar_baseline_gaussian2', 'haar_baseline_gaussian1', 'haar_baseline_savgol', 'haar_baseline_baseline', 'haar_baseline_id', 'haar_id_dv3', 'haar_id_dv2', 'haar_id_dv1', 'haar_id_msc', 'haar_id_detrend', 'haar_id_coif3', 'haar_id_haar', 'haar_id_gaussian2', 'haar_id_gaussian1', 'haar_id_savgol', 'haar_id_baseline', 'haar_id_id', 'gaussian2_dv3_dv3', 'gaussian2_dv3_dv2', 'gaussian2_dv3_dv1', 'gaussian2_dv3_msc', 'gaussian2_dv3_detrend', 'gaussian2_dv3_coif3', 'gaussian2_dv3_haar', 'gaussian2_dv3_gaussian2', 'gaussian2_dv3_gaussian1', 'gaussian2_dv3_savgol', 'gaussian2_dv3_baseline', 'gaussian2_dv3_id', 'gaussian2_dv2_dv3', 'gaussian2_dv2_dv2', 'gaussian2_dv2_dv1', 'gaussian2_dv2_msc', 'gaussian2_dv2_detrend', 'gaussian2_dv2_coif3', 'gaussian2_dv2_haar', 'gaussian2_dv2_gaussian2', 'gaussian2_dv2_gaussian1', 'gaussian2_dv2_savgol', 'gaussian2_dv2_baseline', 'gaussian2_dv2_id', 'gaussian2_dv1_dv3', 'gaussian2_dv1_dv1', 'gaussian2_dv1_msc', 'gaussian2_dv1_detrend', 'gaussian2_dv1_coif3', 'gaussian2_dv1_haar', 'gaussian2_dv1_gaussian2', 'gaussian2_dv1_gaussian1', 'gaussian2_dv1_savgol', 'gaussian2_dv1_baseline', 'gaussian2_dv1_id', 'gaussian2_msc_dv3', 'gaussian2_msc_dv2', 'gaussian2_msc_dv1', 'gaussian2_msc_msc', 'gaussian2_msc_detrend', 'gaussian2_msc_coif3', 'gaussian2_msc_haar', 'gaussian2_msc_gaussian2', 'gaussian2_msc_gaussian1', 'gaussian2_msc_savgol', 'gaussian2_msc_baseline', 'gaussian2_msc_id', 'gaussian2_detrend_dv3', 'gaussian2_detrend_dv2', 'gaussian2_detrend_dv1', 'gaussian2_detrend_msc', 'gaussian2_detrend_detrend', 'gaussian2_detrend_coif3', 'gaussian2_detrend_haar', 'gaussian2_detrend_gaussian2', 'gaussian2_detrend_gaussian1', 'gaussian2_detrend_savgol', 'gaussian2_detrend_baseline', 'gaussian2_detrend_id', 'gaussian2_coif3_dv3', 'gaussian2_coif3_dv2', 'gaussian2_coif3_dv1', 'gaussian2_coif3_msc', 'gaussian2_coif3_detrend', 'gaussian2_coif3_coif3', 'gaussian2_coif3_haar', 'gaussian2_coif3_gaussian2', 'gaussian2_coif3_gaussian1', 'gaussian2_coif3_savgol', 'gaussian2_coif3_baseline', 'gaussian2_coif3_id', 'gaussian2_haar_dv3', 'gaussian2_haar_dv2', 'gaussian2_haar_dv1', 'gaussian2_haar_msc', 'gaussian2_haar_detrend', 'gaussian2_haar_coif3', 'gaussian2_haar_haar', 'gaussian2_haar_gaussian2', 'gaussian2_haar_gaussian1', 'gaussian2_haar_savgol', 'gaussian2_haar_baseline', 'gaussian2_haar_id', 'gaussian2_gaussian2_dv3', 'gaussian2_gaussian2_dv2', 'gaussian2_gaussian2_dv1', 'gaussian2_gaussian2_msc', 'gaussian2_gaussian2_detrend', 'gaussian2_gaussian2_coif3', 'gaussian2_gaussian2_haar', 'gaussian2_gaussian2_gaussian2', 'gaussian2_gaussian2_gaussian1', 'gaussian2_gaussian2_savgol', 'gaussian2_gaussian2_baseline', 'gaussian2_gaussian2_id', 'gaussian2_gaussian1_dv3', 'gaussian2_gaussian1_dv2', 'gaussian2_gaussian1_dv1', 'gaussian2_gaussian1_msc', 'gaussian2_gaussian1_detrend', 'gaussian2_gaussian1_gaussian2', 'gaussian2_gaussian1_gaussian1', 'gaussian2_gaussian1_savgol', 'gaussian2_gaussian1_baseline', 'gaussian2_gaussian1_id', 'gaussian2_savgol_dv3', 'gaussian2_savgol_dv2', 'gaussian2_savgol_dv1', 'gaussian2_savgol_msc', 'gaussian2_savgol_detrend', 'gaussian2_savgol_coif3', 'gaussian2_savgol_haar', 'gaussian2_savgol_gaussian2', 'gaussian2_savgol_gaussian1', 'gaussian2_savgol_baseline', 'gaussian2_savgol_id', 'gaussian2_baseline_dv2', 'gaussian2_baseline_dv1', 'gaussian2_baseline_msc', 'gaussian2_baseline_coif3', 'gaussian2_baseline_haar', 'gaussian2_baseline_gaussian2', 'gaussian2_baseline_gaussian1', 'gaussian2_baseline_savgol', 'gaussian2_baseline_baseline', 'gaussian2_baseline_id', 'gaussian2_id_dv3', 'gaussian2_id_dv2', 'gaussian2_id_dv1', 'gaussian2_id_msc', 'gaussian2_id_detrend', 'gaussian2_id_coif3', 'gaussian2_id_haar', 'gaussian2_id_gaussian2', 'gaussian2_id_gaussian1', 'gaussian2_id_savgol', 'gaussian2_id_baseline', 'gaussian2_id_id', 'gaussian1_dv3_dv3', 'gaussian1_dv3_dv2', 'gaussian1_dv3_dv1', 'gaussian1_dv3_msc', 'gaussian1_dv3_detrend', 'gaussian1_dv3_coif3', 'gaussian1_dv3_haar', 'gaussian1_dv3_gaussian2', 'gaussian1_dv3_gaussian1', 'gaussian1_dv3_savgol', 'gaussian1_dv3_baseline', 'gaussian1_dv3_id', 'gaussian1_dv2_dv3', 'gaussian1_dv2_dv2', 'gaussian1_dv2_dv1', 'gaussian1_dv2_msc', 'gaussian1_dv2_detrend', 'gaussian1_dv2_haar', 'gaussian1_dv2_gaussian2', 'gaussian1_dv2_gaussian1', 'gaussian1_dv2_savgol', 'gaussian1_dv2_baseline', 'gaussian1_dv2_id', 'gaussian1_dv1_dv3', 'gaussian1_dv1_dv2', 'gaussian1_dv1_dv1', 'gaussian1_dv1_msc', 'gaussian1_dv1_coif3', 'gaussian1_dv1_haar', 'gaussian1_dv1_gaussian2', 'gaussian1_dv1_gaussian1', 'gaussian1_dv1_savgol', 'gaussian1_dv1_baseline', 'gaussian1_msc_dv2', 'gaussian1_msc_msc', 'gaussian1_msc_detrend', 'gaussian1_msc_haar', 'gaussian1_msc_gaussian2', 'gaussian1_detrend_savgol', 'gaussian1_coif3_dv3', 'gaussian1_haar_dv3', 'gaussian1_id_savgol', 'gaussian1_id_baseline', 'gaussian1_id_id', 'savgol_dv3_dv3', 'savgol_dv3_dv2', 'savgol_dv3_dv1', 'savgol_dv3_msc', 'savgol_dv3_detrend', 'savgol_dv3_coif3', 'savgol_dv3_haar', 'savgol_dv3_gaussian2', 'savgol_dv3_gaussian1', 'savgol_dv3_savgol', 'savgol_dv3_baseline', 'savgol_dv3_id', 'savgol_dv2_dv3', 'savgol_dv2_dv2', 'savgol_dv2_dv1', 'savgol_dv2_msc', 'savgol_dv2_detrend', 'savgol_dv2_coif3', 'savgol_dv2_haar', 'savgol_dv2_gaussian2', 'savgol_dv2_gaussian1', 'savgol_dv2_savgol', 'savgol_dv2_baseline', 'savgol_dv2_id', 'savgol_dv1_dv3', 'savgol_dv1_dv2', 'savgol_dv1_dv1', 'savgol_dv1_msc', 'savgol_dv1_detrend', 'savgol_dv1_coif3', 'savgol_dv1_haar', 'savgol_dv1_gaussian2', 'savgol_dv1_gaussian1', 'savgol_dv1_savgol', 'savgol_dv1_baseline', 'savgol_dv1_id', 'savgol_msc_dv3', 'savgol_msc_dv2', 'savgol_msc_dv1', 'savgol_msc_msc', 'savgol_msc_detrend', 'savgol_msc_coif3', 'savgol_msc_haar', 'savgol_msc_gaussian2', 'savgol_msc_gaussian1', 'savgol_msc_savgol', 'savgol_msc_baseline', 'savgol_msc_id', 'savgol_detrend_dv3', 'savgol_detrend_dv2', 'savgol_detrend_dv1', 'savgol_detrend_msc', 'savgol_detrend_detrend', 'savgol_detrend_coif3', 'savgol_detrend_haar', 'savgol_detrend_gaussian2', 'savgol_detrend_gaussian1', 'savgol_detrend_savgol', 'savgol_detrend_baseline', 'savgol_detrend_id', 'savgol_coif3_dv3', 'savgol_coif3_dv2', 'savgol_coif3_dv1', 'savgol_coif3_msc', 'savgol_coif3_detrend', 'savgol_coif3_coif3', 'savgol_coif3_haar', 'savgol_coif3_gaussian2', 'savgol_coif3_gaussian1', 'savgol_coif3_savgol', 'savgol_coif3_baseline', 'savgol_coif3_id', 'savgol_haar_dv3', 'savgol_haar_dv2', 'savgol_haar_dv1', 'savgol_haar_msc', 'savgol_haar_detrend', 'savgol_haar_coif3', 'savgol_haar_haar', 'savgol_haar_gaussian2', 'savgol_haar_gaussian1', 'savgol_haar_savgol', 'savgol_haar_baseline', 'savgol_haar_id', 'savgol_gaussian2_dv3', 'savgol_gaussian2_dv2', 'savgol_gaussian2_dv1', 'savgol_gaussian2_msc', 'savgol_gaussian2_detrend', 'savgol_gaussian2_coif3', 'savgol_gaussian2_haar', 'savgol_gaussian2_gaussian2', 'savgol_gaussian2_gaussian1', 'savgol_gaussian2_savgol', 'savgol_gaussian2_baseline', 'savgol_gaussian2_id', 'savgol_gaussian1_dv3', 'savgol_gaussian1_dv2', 'savgol_gaussian1_dv1', 'savgol_gaussian1_msc', 'savgol_gaussian1_detrend', 'savgol_gaussian1_coif3', 'savgol_gaussian1_haar', 'savgol_gaussian1_gaussian2', 'savgol_gaussian1_gaussian1', 'savgol_gaussian1_savgol', 'savgol_gaussian1_baseline', 'savgol_gaussian1_id', 'savgol_savgol_dv3', 'savgol_savgol_dv2', 'savgol_savgol_dv1', 'savgol_savgol_msc', 'savgol_savgol_detrend', 'savgol_savgol_coif3', 'savgol_savgol_haar', 'savgol_savgol_gaussian2', 'savgol_savgol_gaussian1', 'savgol_savgol_savgol', 'savgol_savgol_baseline', 'savgol_savgol_id', 'savgol_baseline_dv3', 'savgol_baseline_dv2', 'savgol_baseline_dv1', 'savgol_baseline_msc', 'savgol_baseline_detrend', 'savgol_baseline_coif3', 'savgol_baseline_haar', 'savgol_baseline_gaussian2', 'savgol_baseline_gaussian1', 'savgol_baseline_savgol', 'savgol_baseline_baseline', 'savgol_baseline_id', 'savgol_id_dv3', 'savgol_id_dv2', 'savgol_id_dv1', 'savgol_id_msc', 'savgol_id_detrend', 'savgol_id_coif3', 'savgol_id_haar', 'savgol_id_gaussian2', 'savgol_id_gaussian1', 'savgol_id_savgol', 'savgol_id_baseline', 'savgol_id_id', 'baseline_dv3_dv3', 'baseline_dv3_dv2', 'baseline_dv3_dv1', 'baseline_dv3_msc', 'baseline_dv3_detrend', 'baseline_dv3_coif3', 'baseline_dv3_haar', 'baseline_dv3_gaussian2', 'baseline_dv3_gaussian1', 'baseline_dv3_savgol', 'baseline_dv3_baseline', 'baseline_dv3_id', 'baseline_dv2_dv3', 'baseline_dv2_dv2', 'baseline_dv2_dv1', 'baseline_dv2_msc', 'baseline_dv2_detrend', 'baseline_dv2_coif3', 'baseline_dv2_haar', 'baseline_dv2_gaussian2', 'baseline_dv2_gaussian1', 'baseline_dv2_savgol', 'baseline_dv2_baseline', 'baseline_dv2_id', 'baseline_dv1_dv3', 'baseline_dv1_dv2', 'baseline_dv1_dv1', 'baseline_dv1_msc', 'baseline_dv1_detrend', 'baseline_dv1_coif3', 'baseline_dv1_haar', 'baseline_dv1_gaussian2', 'baseline_dv1_gaussian1', 'baseline_dv1_savgol', 'baseline_dv1_baseline', 'baseline_dv1_id', 'baseline_msc_dv3', 'baseline_msc_dv2', 'baseline_msc_dv1', 'baseline_msc_msc', 'baseline_msc_detrend', 'baseline_msc_coif3', 'baseline_msc_haar', 'baseline_msc_gaussian2', 'baseline_msc_gaussian1', 'baseline_msc_savgol', 'baseline_msc_baseline', 'baseline_msc_id', 'baseline_detrend_dv3', 'baseline_detrend_dv2', 'baseline_detrend_dv1', 'baseline_detrend_msc', 'baseline_detrend_detrend', 'baseline_detrend_coif3', 'baseline_detrend_haar', 'baseline_detrend_gaussian2', 'baseline_detrend_gaussian1', 'baseline_detrend_savgol', 'baseline_detrend_baseline', 'baseline_detrend_id', 'baseline_coif3_dv3', 'baseline_coif3_dv2', 'baseline_coif3_dv1', 'baseline_coif3_msc', 'baseline_coif3_detrend', 'baseline_coif3_coif3', 'baseline_coif3_haar', 'baseline_coif3_gaussian2', 'baseline_coif3_gaussian1', 'baseline_coif3_savgol', 'baseline_coif3_baseline', 'baseline_coif3_id', 'baseline_haar_dv3', 'baseline_haar_dv2', 'baseline_haar_dv1', 'baseline_haar_msc', 'baseline_haar_detrend', 'baseline_haar_coif3', 'baseline_haar_haar', 'baseline_haar_gaussian2', 'baseline_haar_gaussian1', 'baseline_haar_savgol', 'baseline_haar_baseline', 'baseline_haar_id', 'baseline_gaussian2_dv3', 'baseline_gaussian2_dv2', 'baseline_gaussian2_dv1', 'baseline_gaussian2_msc', 'baseline_gaussian2_detrend', 'baseline_gaussian2_coif3', 'baseline_gaussian2_haar', 'baseline_gaussian2_gaussian2', 'baseline_gaussian2_gaussian1', 'baseline_gaussian2_savgol', 'baseline_gaussian2_baseline', 'baseline_gaussian2_id', 'baseline_gaussian1_dv3', 'baseline_gaussian1_dv2', 'baseline_gaussian1_dv1', 'baseline_gaussian1_msc', 'baseline_gaussian1_detrend', 'baseline_gaussian1_coif3', 'baseline_gaussian1_haar', 'baseline_gaussian1_gaussian2', 'baseline_gaussian1_gaussian1', 'baseline_gaussian1_savgol', 'baseline_gaussian1_baseline', 'baseline_gaussian1_id', 'baseline_savgol_dv3', 'baseline_savgol_dv2', 'baseline_savgol_dv1', 'baseline_savgol_msc', 'baseline_savgol_detrend', 'baseline_savgol_coif3', 'baseline_savgol_haar', 'baseline_savgol_gaussian2', 'baseline_savgol_gaussian1', 'baseline_savgol_savgol', 'baseline_savgol_baseline', 'baseline_savgol_id', 'baseline_baseline_dv3', 'baseline_baseline_dv2', 'baseline_baseline_dv1', 'baseline_baseline_msc', 'baseline_baseline_detrend', 'baseline_baseline_coif3', 'baseline_baseline_haar', 'baseline_baseline_gaussian2', 'baseline_baseline_gaussian1', 'baseline_baseline_savgol', 'baseline_baseline_baseline', 'baseline_baseline_id', 'baseline_id_dv3', 'baseline_id_dv2', 'baseline_id_dv1', 'baseline_id_msc', 'baseline_id_detrend', 'baseline_id_coif3', 'baseline_id_haar', 'baseline_id_gaussian2', 'baseline_id_gaussian1', 'baseline_id_savgol', 'baseline_id_baseline', 'baseline_id_id', 'id_dv3_dv3', 'id_dv3_dv2', 'id_dv3_dv1', 'id_dv3_msc', 'id_dv3_detrend', 'id_dv3_coif3', 'id_dv3_haar', 'id_dv3_gaussian2', 'id_dv3_gaussian1', 'id_dv3_savgol', 'id_dv3_baseline', 'id_dv3_id', 'id_dv2_dv3', 'id_dv2_dv2', 'id_dv2_dv1', 'id_dv2_msc', 'id_dv2_detrend', 'id_dv2_coif3', 'id_dv2_haar', 'id_dv2_gaussian2', 'id_dv2_gaussian1', 'id_dv2_savgol', 'id_dv2_baseline', 'id_dv2_id', 'id_dv1_dv3', 'id_dv1_dv2', 'id_dv1_dv1', 'id_dv1_msc', 'id_dv1_detrend', 'id_dv1_coif3', 'id_dv1_haar', 'id_dv1_gaussian2', 'id_dv1_gaussian1', 'id_dv1_savgol', 'id_dv1_baseline', 'id_dv1_id', 'id_msc_dv3', 'id_msc_dv2', 'id_msc_dv1', 'id_msc_msc', 'id_msc_detrend', 'id_msc_coif3', 'id_msc_haar', 'id_msc_gaussian2', 'id_msc_gaussian1', 'id_msc_savgol', 'id_msc_baseline', 'id_msc_id', 'id_detrend_dv3', 'id_detrend_dv2', 'id_detrend_dv1', 'id_detrend_msc', 'id_detrend_detrend', 'id_detrend_coif3', 'id_detrend_haar', 'id_detrend_gaussian2', 'id_detrend_gaussian1', 'id_detrend_savgol', 'id_detrend_baseline', 'id_detrend_id', 'id_coif3_dv3', 'id_coif3_dv2', 'id_coif3_dv1', 'id_coif3_msc', 'id_coif3_detrend', 'id_coif3_coif3', 'id_coif3_haar', 'id_coif3_gaussian2', 'id_coif3_gaussian1', 'id_coif3_savgol', 'id_coif3_baseline', 'id_coif3_id', 'id_haar_dv3', 'id_haar_dv2', 'id_haar_dv1', 'id_haar_msc', 'id_haar_detrend', 'id_haar_coif3', 'id_haar_haar', 'id_haar_gaussian2', 'id_haar_gaussian1', 'id_haar_savgol', 'id_haar_baseline', 'id_haar_id', 'id_gaussian2_dv3', 'id_gaussian2_dv2', 'id_gaussian2_dv1', 'id_gaussian2_msc', 'id_gaussian2_detrend', 'id_gaussian2_coif3', 'id_gaussian2_haar', 'id_gaussian2_gaussian2', 'id_gaussian2_gaussian1', 'id_gaussian2_savgol', 'id_gaussian2_baseline', 'id_gaussian2_id', 'id_gaussian1_dv3', 'id_gaussian1_dv2', 'id_gaussian1_dv1', 'id_gaussian1_msc', 'id_gaussian1_detrend', 'id_gaussian1_coif3', 'id_gaussian1_haar', 'id_gaussian1_gaussian2', 'id_gaussian1_gaussian1', 'id_gaussian1_savgol', 'id_gaussian1_baseline', 'id_gaussian1_id', 'id_savgol_dv3', 'id_savgol_dv2', 'id_savgol_dv1', 'id_savgol_msc', 'id_savgol_detrend', 'id_savgol_coif3', 'id_savgol_haar', 'id_savgol_gaussian2', 'id_savgol_gaussian1', 'id_savgol_savgol', 'id_savgol_baseline', 'id_savgol_id', 'id_baseline_dv3', 'id_baseline_dv2', 'id_baseline_dv1', 'id_baseline_msc', 'id_baseline_detrend', 'id_baseline_coif3', 'id_baseline_haar', 'id_baseline_gaussian2', 'id_baseline_gaussian1', 'id_baseline_savgol', 'id_baseline_baseline', 'id_baseline_id', 'id_id_dv3', 'id_id_dv2', 'id_id_dv1', 'id_id_msc', 'id_id_detrend', 'id_id_coif3', 'id_id_haar', 'id_id_gaussian2', 'id_id_gaussian1', 'id_id_savgol', 'id_id_baseline', 'id_id_id']
    src_set = dumb_and_dumber_set()
    optimal_set = []
    for it in src_set:
        if it[0] in fat_list:
            optimal_set.append(it)
    return optimal_set
