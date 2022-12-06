from pathlib import Path

from pinard.utils import load_csv


def load_data(path):
    # Load data
    projdir = Path(path)
    files = tuple(next(projdir.glob(n)) for n in ["*Xcal*", "*Ycal*", "*Xval*", "*Yval*"])
    X_train, y_train = load_csv(files[0], files[1], x_hdr=0, y_hdr=0, sep=';')
    X_valid, y_valid = load_csv(files[2], files[3], x_hdr=0, y_hdr=0, sep=';')
    return X_train, y_train, X_valid, y_valid
