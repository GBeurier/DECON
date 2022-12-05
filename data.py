from pathlib import Path

from pinard.utils import load_csv


def load_data(path):
    # Load data
    projdir = Path(path)
    files = tuple(next(projdir.glob(n)) for n in ["*Xcal*", "*Ycal*", "*Xval*", "*Yval*"])
    X_train, y_train = load_csv(files[0], files[1], x_hdr=0, y_hdr=0, sep=';')
    X_valid, y_valid = load_csv(files[2], files[3], x_hdr=0, y_hdr=0, sep=';')
    return X_train, y_train, X_valid, y_valid



# def data_cv_config(XCal, yCal, XVal, yVal):
#     fold = RepeatedKFold(5, 2, random_state=SEED)
#     datasets = []
#     for train_index, test_index in fold.split(XCal):
#         datasets.append((XCal[train_index], yCal[train_index], XCal[test_index], yCal[test_index]))

#     return (datasets, (XVal, yVal))

# def data_augmented_config(XCal, yCal, XVal, yVal):
#     #TODO
#     pass