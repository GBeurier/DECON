from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor
from lwpls import LWPLS


def ml_list(SEED, X_test, y_test):
    # ml_models = [(PLSRegression(nc, max_iter=5000), "PLS" + str(nc)) for nc in range(4, 12, 4)] # test
    ml_models = [(PLSRegression(nc, max_iter=5000), "PLS" + str(nc))
                 for nc in range(4, 100, 4)]
    ml_models.append((XGBRegressor(seed=SEED), 'XGBoost_100_None'))
    ml_models.append(
        (XGBRegressor(n_estimators=200, max_depth=50, seed=SEED), 'XGBoost_200_10'))
    ml_models.append(
        (XGBRegressor(n_estimators=50, max_depth=100, seed=SEED), 'XGBoost_50_100'))
    ml_models.append(
        (XGBRegressor(n_estimators=200, seed=SEED), 'XGBoost_200_'))
    ml_models.append(
        (XGBRegressor(n_estimators=400, max_depth=100, seed=SEED), 'XGBoost_400_100'))

    ml_models.append((LWPLS(2, 2 ** -2, X_test, y_test), "LWPLS_2_0.25"))
    ml_models.append((LWPLS(16, 2 ** -2, X_test, y_test), "LWPLS_16_0.25"))
    ml_models.append((LWPLS(30, 2 ** -2, X_test, y_test), "LWPLS_30_0.25"))

    return ml_models
