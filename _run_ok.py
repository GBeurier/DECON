import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input


def main():
    X_train = np.random.rand(256, 1024)
    expanded_array = np.expand_dims(X_train, axis=2)
    X_train = np.repeat(expanded_array, 5, axis=2)

    y_train = np.random.rand(256, 1)

    model = Sequential()
    model.add(Input(shape=X_train.shape[1:]))
    model.add(Flatten())
    model.add(Dense(units=64, activation="relu"))
    model.add(Dense(units=16, activation="sigmoid"))
    model.add(Dense(units=1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="mse", metrics=["mse", "mae"])
    model.summary()

    model.fit(X_train, y_train, batch_size=50, epochs=10, verbose=2)
    print("Training done")


if __name__ == "__main__":
    main()
