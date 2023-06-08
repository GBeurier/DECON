import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Conv1D, SpatialDropout1D, DepthwiseConv1D, MaxPooling1D, BatchNormalization, SeparableConv1D, Dropout


def main():
    X_train = np.random.rand(2500, 2048)
    expanded_array = np.expand_dims(X_train, axis=2)
    X_train = np.repeat(expanded_array, 12, axis=2)
    y_train = np.random.rand(2500, 1)

    model = Sequential()
    model.add(Input(shape=X_train.shape[1:]))
    model.add(SpatialDropout1D(0.2))
    model.add(DepthwiseConv1D(kernel_size=7, padding="same", depth_multiplier=2, activation="relu"))
    model.add(DepthwiseConv1D(kernel_size=7, padding="same", depth_multiplier=2, activation="relu"))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(DepthwiseConv1D(kernel_size=5, padding="same", depth_multiplier=2, activation="relu"))
    model.add(DepthwiseConv1D(kernel_size=5, padding="same", depth_multiplier=2, activation="relu"))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(DepthwiseConv1D(kernel_size=9, padding="same", depth_multiplier=2, activation="relu"))
    model.add(DepthwiseConv1D(kernel_size=9, padding="same", depth_multiplier=2, activation="relu"))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=1, padding="same", activation="relu"))
    model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=5, strides=3))
    model.add(SpatialDropout1D(0.1))
    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=32, activation="relu"))
    model.add(Dropout(0.2))
    # model.add(Dense(units=16, activation="sigmoid"))
    model.add(Dense(units=1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="mse", metrics=["mse", "mae"])
    model.summary()

    model.fit(X_train, y_train, batch_size=50, epochs=10, verbose=2)
    print("Training done")


if __name__ == "__main__":
    main()
