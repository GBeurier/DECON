from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler

from pinard import preprocessing as pp
from pinard.sklearn import FeatureAugmentation


def id_preprocessing():
    return [('id', pp.IdentityTransformer())]


def savgol():
    return [('savgol', pp.SavitzkyGolay())]


def haar():
    return [('haar', pp.Wavelet('haar'))]


def bacon_set():
    preprocessing = [   
        ('id', pp.IdentityTransformer()),
        ('savgol1', pp.SavitzkyGolay(window_length=17, polyorder=2, deriv=2)),
        ('savgol2', pp.SavitzkyGolay(window_length=5, polyorder=2)),
        ('gaussian1', pp.Gaussian(order=1, sigma=2)),
        ('gaussian2', pp.Gaussian(order=2, sigma=1)),
        ('gaussian3', pp.Gaussian(order=0, sigma=2)),
        ('baseline', pp.StandardNormalVariate()),
        ('msc', pp.MultiplicativeScatterCorrection(scale=False)),
        ('detrend', pp.Detrend()),
        ('derivate', pp.Derivate(2, 1)),
        ('dv2', pp.Derivate(2, 1)),
        ('haar', pp.Wavelet('haar')),
        ]
    return preprocessing


def decon_set():
    preprocessing = [   
        ('id', pp.IdentityTransformer()),
        ('baseline', pp.StandardNormalVariate()),
        ('savgol', pp.SavitzkyGolay()),
        ('gaussian1', pp.Gaussian(order=1, sigma=2)),
        ('gaussian2', pp.Gaussian(order=2, sigma=1)),
        ('haar', pp.Wavelet('haar')),
        ('coif3', pp.Wavelet('coif3')),
        ('detrend', pp.Detrend()),
        ('msc', pp.MultiplicativeScatterCorrection(scale=False)),
        ('dv1', pp.Derivate(1, 1)),
        ('dv2', pp.Derivate(2, 1)),
        ('dv3', pp.Derivate(2, 2)),
        ('baseline*savgol', Pipeline([('_sg1', pp.StandardNormalVariate()), ('_sg2', pp.SavitzkyGolay())])),
        ('baseline*gaussian1', Pipeline([('_sg1', pp.StandardNormalVariate()), ('g2', pp.Gaussian(order=1, sigma=2))])),
        ('baseline*gaussian2', Pipeline([('_sg1', pp.StandardNormalVariate()), ('g2', pp.Gaussian(order=2, sigma=1))])),
        ('baseline*haar', Pipeline([('_sg1', pp.StandardNormalVariate()), ('_sg2', pp.Wavelet('haar'))])),
        ('savgol*gaussian1', Pipeline([('_sg1', pp.SavitzkyGolay()), ('g2', pp.Gaussian(order=1, sigma=2))])),
        ('savgol*gaussian2', Pipeline([('_sg1', pp.SavitzkyGolay()), ('g2', pp.Gaussian(order=2, sigma=1))])),
        ('gaussian2*savgol', Pipeline([('_g2', pp.Gaussian(order=1, sigma=2)), ('_sg4', pp.SavitzkyGolay())])),
        ('haar*gaussian2', Pipeline([('_haar2', pp.Wavelet('haar')), ('g2', pp.Gaussian(order=2, sigma=1))])),
    ]
    return preprocessing


def transform_test_data(preprocessing, X_train, y_train, X_test, y_test, type="augmentation"):
    y_scaler = MinMaxScaler()
    y_scaler.fit(y_train.reshape((-1, 1)))
    y_valid = y_scaler.transform(y_test.reshape((-1, 1)))

    if type == "augmentation":
        transformer_pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('preprocessing', FeatureAugmentation(preprocessing())),
        ])
    else:
        transformer_pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('preprocessing', FeatureUnion(preprocessing())),
        ])

    transformer_pipeline.fit(X_train)
    X_valid = transformer_pipeline.transform(X_test)

    return X_valid, y_valid, transformer_pipeline, y_scaler


def preprocessing_list():
    return [id_preprocessing, savgol, haar, bacon_set, decon_set]