import datetime
import inspect
import math
import numpy as np
import re
import os

from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
import numpy as np


import tensorflow as tf
# import tensorflow_addons as tfa

from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    Activation,
    SpatialDropout1D,
    BatchNormalization,
    Flatten,
    Dropout,
    Input,
    GRU,
    LSTM,
    Bidirectional,
    MaxPool1D,
    AveragePooling1D,
    SeparableConv1D,
    Add,
    GlobalAveragePooling1D,
    GlobalMaxPool1D,
    Concatenate, concatenate,
    DepthwiseConv1D,
    Permute,
    MaxPooling1D,
    LayerNormalization,
    MultiHeadAttention,
    SeparableConv1D,
    ConvLSTM1D,
    LocallyConnected1D,
    Multiply,
    UpSampling1D,
    Lambda,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from scikeras.wrappers import KerasRegressor

from keras_self_attention import SeqSelfAttention

# from transformers import Transformer
from lwpls import LWPLS as LWPLS2
from contextlib import redirect_stdout


class Auto_Save_Multiple(Callback):
    best_weights = []

    def __init__(self, model_name, shape, cb_func=None):
        super(Auto_Save_Multiple, self).__init__()
        self.model_name = model_name
        self.shape = shape
        self.best = np.Inf
        self.best_unscaled = np.Inf
        self.cb = cb_func

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss")
        lr = self.model.optimizer.learning_rate
        
        print("epoch", str(epoch).zfill(5), "lr", lr.numpy(), " - ", "{:.6f}".format(current_loss), "{:.6f}".format(self.best), " "*10, end="\r")

        if np.less(current_loss, self.best):
            self.best = current_loss
            Auto_Save_Multiple.best_weights = self.model.get_weights()
            self.best_epoch = epoch

            if self.cb is not None:
                res = self.cb(epoch, self.best)
                RMSE = float(res['RMSE'])
                if RMSE < self.best_unscaled:
                    self.best_unscaled = RMSE
                    
            print("Best so far >", self.best_unscaled, self.model_name)


    def on_train_end(self, logs=None):
        # if self.params['verbose'] == 2:
        print("Saved best {0:6.4f} at epoch".format(self.best_unscaled), self.best_epoch)
        self.model.set_weights(Auto_Save_Multiple.best_weights)
        self.model.save_weights(self.model_name + ".hdf5")
        with open(self.model_name + "_sum.txt", 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

class Auto_Save(Callback):
    best_weights = []

    def __init__(self, model_name, shape, cb_func=None):
        super(Auto_Save, self).__init__()
        self.model_name = model_name
        self.shape = shape
        self.best = np.Inf
        self.best_unscaled = np.Inf
        self.cb = cb_func

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss")
        lr = self.model.optimizer.learning_rate

        print("epoch", str(epoch).zfill(5), "lr", lr.numpy(), " - ", "{:.6f}".format(current_loss), "{:.6f}".format(self.best), " " * 10, end="\r")

        if np.less(current_loss, self.best):
            if self.cb is not None:
                res = self.cb(epoch, self.best)
                RMSE = float(res["RMSE"])
                self.best = current_loss
                if RMSE < self.best_unscaled:
                    self.best_unscaled = RMSE
                    Auto_Save.best_weights = self.model.get_weights()
                    self.best_epoch = epoch
            print("Best so far >", self.best_unscaled, self.model_name)

    def on_train_end(self, logs=None):
        # if self.params['verbose'] == 2:
        print("Saved best {0:6.4f} at epoch".format(self.best_unscaled), self.best_epoch)
        self.model.set_weights(Auto_Save.best_weights)
        self.model.save_weights(self.model_name + ".hdf5")
        self.model.save(self.model_name + ".h5")
        with open(self.model_name + "_sum.txt", "w") as f:
            with redirect_stdout(f):
                self.model.summary()


class Print_LR(Callback):
    def on_epoch_end(self, epoch, logs=None):
        iteration = self.model.optimizer.iterations.numpy()
        # lr = clr(iteration).numpy()
        lr = self.model.optimizer.learning_rate
        if self.params["verbose"] == 2:
            print("Iteration {} - Learning rate: {}".format(iteration, lr))


def rmse_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean((y_true - y_pred) ** 2))


# def coeff_determination(y_true, y_pred):
#     # SS_res = tf.sum(tf.square(y_true - y_pred))
#     # SS_tot = tf.sum(tf.square(y_true - tf.mean(y_true)))
#     # return (1 - SS_res/(SS_tot + tf.epsilon()))


def scale_fn(x):
    # return 1. ** x
    return 1 / (2.0 ** (x - 1))

    # def calc_lr(step, warmup_steps=200):
    #     return 2**(-0.5) * min(step**(-0.5), (step+1) * warmup_steps**(-1.5))


def clr(epoch):
    # return 0.05
    cycle_params = {
        "MIN_LR": 0.0001,
        "MAX_LR": 0.05,
        "CYCLE_LENGTH": 256,
    }
    MIN_LR, MAX_LR, CYCLE_LENGTH = (
        cycle_params["MIN_LR"],
        cycle_params["MAX_LR"],
        cycle_params["CYCLE_LENGTH"],
    )
    initial_learning_rate = MIN_LR
    maximal_learning_rate = MAX_LR
    step_size = CYCLE_LENGTH
    step_as_dtype = float(epoch)
    cycle = math.floor(1 + step_as_dtype / (2 * step_size))
    x = abs(step_as_dtype / step_size - 2 * cycle + 1)
    mode_step = cycle  # if scale_mode == "cycle" else step
    return initial_learning_rate + (maximal_learning_rate - initial_learning_rate) * max(0, (1 - x)) * scale_fn(mode_step)


class NIRS_Regressor:
    def __init__(self, *, params={}, name=""):
        self.params = params
        self.name_ = name

    def model(self, X_train, y_train=None, X_test=None, y_test=None, *, run_name="", cb=None, params=None, desc=None, discretizer=None):
        pass

    def name(self):
        if len(self.name_) > 0:
            return self.name_
        return re.search(".*'(.*)'.*", str(self.__class__)).group(1).split(".")[-1]


class NN_NIRS_Regressor(NIRS_Regressor):
    def build_model(self, input_shape, params):
        return None

    def model(self, X_train, y_train=None, X_test=None, y_test=None, *, run_name="default_run", cb=None, params=None, desc=None, discretizer=None):

        early_stop = EarlyStopping(monitor="val_loss", patience=params["patience"], verbose=1, mode="min", min_delta=0)
        # log_dir = os.path.join('logs','fit','run_name', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=int(params["patience"] / 2), min_lr=0.001)
        lrScheduler = tf.keras.callbacks.LearningRateScheduler(clr)
        weights_path = os.path.join("results", desc[0], run_name)
        auto_save = Auto_Save(weights_path, X_test.shape, cb)
        callbacks = [auto_save, early_stop, lrScheduler]  #  reduce_lr tensorboard_callback
        model_inst = self.build_model(X_test.shape[1:], params)

        if discretizer is not None:
            x = model_inst.layers[-2].output
            x = Dense(discretizer.n_bins, activation="softmax")(x)
            model_inst = Model(inputs=model_inst.inputs, outputs=x)

        # model_inst.summary()

        dot_img_file = os.path.join("results", desc[0], run_name + "_model.jpg")
        tf.keras.utils.plot_model(model_inst, to_file=dot_img_file, show_shapes=True)

        trainableParams = np.sum([np.prod(v.get_shape()) for v in model_inst.trainable_weights])
        nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model_inst.non_trainable_weights])
        totalParams = trainableParams + nonTrainableParams
        print(
            "--- Trainable:",
            trainableParams,
            "- untrainable:",
            nonTrainableParams,
            ">",
            totalParams,
        )

        if discretizer is None:
            rmse = tf.keras.metrics.RootMeanSquaredError()
            k_regressor = KerasRegressor(
                model=model_inst,
                loss="mse",
                metrics=[rmse],
                optimizer=params["optimizer"],
                callbacks=callbacks,
                epochs=params["epoch"],
                batch_size=params["batch_size"],
                fit__validation_data=(X_test, y_test),
                fit__shuffle=True,
                verbose=params["verbose"],
            )
        else:
            # scc = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            k_regressor = KerasRegressor(
                model=model_inst,
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=["accuracy"],
                optimizer=params["optimizer"],
                callbacks=callbacks,
                epochs=params["epoch"],
                batch_size=params["batch_size"],
                fit__validation_data=(X_test, y_test),
                fit__shuffle=True,
                verbose=params["verbose"],
            )

        return k_regressor


class Custom_NN(NN_NIRS_Regressor):
    def __init__(self, model, *, params={}, name=""):
        self.model_ = model
        NN_NIRS_Regressor.__init__(params=params, name=name)

    def build_model(self, input_shape, params):
        return self.model_


from models.VGG_1DCNN import VGG


class UNet_NIRS(NN_NIRS_Regressor):
    def build_model(self, input_shape, params):
        length = input_shape[0]  # Length of each Segment of a 1D Signal
        num_channel = input_shape[1]  # Number of Input Channels in the Model
        # length = 1024  # Length of each Segment
        # model_name = 'VGG19'  # DenseNet Models
        model_width = 8  # Width of the Initial Layer, subsequent layers start from here
        problem_type = "Regression"  # Classification or Regression
        output_nums = 1  # Number of Class for Classification Problems, always '1' for Regression Problems
        #
        Model = VGG(length, num_channel, model_width, problem_type=problem_type, output_nums=output_nums, dropout_rate=False).VGG11()
        return Model


class VGG_1D(NN_NIRS_Regressor):
    def vgg_block(self, layer_in, n_filters, n_conv):
        # add convolutional layers
        for _ in range(n_conv):
            layer_in = Conv1D(filters=n_filters, kernel_size=3, padding="same", activation="relu")(layer_in)
        layer_in = MaxPooling1D(2, strides=2)(layer_in)
        return layer_in

    def build_model(self, input_shape, params):
        visible = Input(shape=input_shape)
        layer = self.vgg_block(visible, 64, 2)
        layer = self.vgg_block(layer, 128, 2)
        layer = self.vgg_block(layer, 256, 2)
        layer = Flatten()(layer)
        layer = Dense(units=16, activation="sigmoid")(layer)
        layer = Dense(units=1, activation="linear")(layer)
        model = Model(inputs=visible, outputs=layer)
        return model


class CONV_LSTM(NN_NIRS_Regressor):
    def build_model(self, input_shape, params):
        inputs = Input(shape=input_shape)
        x = inputs
        x1 = SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu")(x)
        x1 = MaxPooling1D()(x1)
        x1 = BatchNormalization()(x1)
        x1 = SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu")(x1)
        x1 = MaxPooling1D()(x1)
        x1 = BatchNormalization()(x1)
        x1 = SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu")(x1)
        x1 = MaxPooling1D()(x1)
        x1 = BatchNormalization()(x1)
        x1 = Flatten()(x1)

        x2 = MultiHeadAttention(key_dim=64, num_heads=8, dropout=0.1)(x, x)
        x2 = MultiHeadAttention(key_dim=64, num_heads=8, dropout=0.1)(x, x)
        x2 = MaxPooling1D()(x2)
        x2 = BatchNormalization()(x2)
        x2 = Conv1D(32, 3, strides=2, padding="same")(x2)
        x2 = Flatten()(x2)

        x3 = Bidirectional(GRU(128))(x)
        x3 = BatchNormalization()(x3)

        x4 = Bidirectional(LSTM(128))(x)
        x4 = BatchNormalization()(x4)

        x = Concatenate()([x1, x2, x3, x4])
        # x = Conv1D(filters=32, kernel_size=3, strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)

        x = Dense(units=64, activation="relu")(x)
        x = Dense(units=16, activation="relu")(x)
        x = Dropout(0.2)(x)
        outputs = Dense(units=1, activation="sigmoid")(x)
        return Model(inputs, outputs)


class UNET(NN_NIRS_Regressor):
    def cbr(self, x, out_layer, kernel, stride, dilation):
        x = Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    def se_block(self, x_in, layer_n):
        x = GlobalAveragePooling1D()(x_in)
        x = Dense(layer_n // 8, activation="relu")(x)
        x = Dense(layer_n, activation="sigmoid")(x)
        x_out = Multiply()([x_in, x])
        return x_out

    def resblock(self, x_in, layer_n, kernel, dilation, use_se=True):
        x = self.cbr(x_in, layer_n, kernel, 1, dilation)
        x = self.cbr(x, layer_n, kernel, 1, dilation)
        if use_se:
            x = self.se_block(x, layer_n)
        x = Add()([x_in, x])
        return x

    def build_model(self, input_shape, params):
        layer_n = 64
        kernel_size = 7
        depth = 2

        input_layer = Input(input_shape)
        input_layer_1 = AveragePooling1D(5)(input_layer)
        input_layer_2 = AveragePooling1D(25)(input_layer)

        ########## Encoder
        x = self.cbr(input_layer, layer_n, kernel_size, 1, 1)  # 1000
        for i in range(depth):
            x = self.resblock(x, layer_n, kernel_size, 1)
        out_0 = x

        x = self.cbr(x, layer_n * 2, kernel_size, 5, 1)
        for i in range(depth):
            x = self.resblock(x, layer_n * 2, kernel_size, 1)
        out_1 = x

        x = Concatenate()([x, input_layer_1])
        x = self.cbr(x, layer_n * 3, kernel_size, 5, 1)
        for i in range(depth):
            x = self.resblock(x, layer_n * 3, kernel_size, 1)
        out_2 = x

        x = Concatenate()([x, input_layer_2])
        x = self.cbr(x, layer_n * 4, kernel_size, 5, 1)
        for i in range(depth):
            x = self.resblock(x, layer_n * 4, kernel_size, 1)

        # ########## Decoder
        # x = UpSampling1D(5)(x)
        # x = Concatenate()([x, out_2])
        # x = self.cbr(x, layer_n*3, kernel_size, 1, 1)

        # x = UpSampling1D(5)(x)
        # x = Concatenate()([x, out_1])
        # x = self.cbr(x, layer_n*2, kernel_size, 1, 1)

        # x = UpSampling1D(5)(x)
        # x = Concatenate()([x, out_0])
        # x = self.cbr(x, layer_n, kernel_size, 1, 1)

        # # regressor
        x = Conv1D(1, kernel_size=kernel_size, strides=1, padding="same")(x)
        out = Activation("relu")(x)
        out = Lambda(lambda x: 12 * x)(out)
        ##
        out = Flatten()(x)
        out = Dense(1, activation="sigmoid")(out)
        ##

        # classifier
        # x = Conv1D(11, kernel_size=kernel_size, strides=1, padding="same")(x)
        # out = Activation("softmax")(x)

        model = Model(input_layer, out)

        return model


from SE_ResNet_1DCNN import SEResNet
from ResNet_v2_1DCNN import ResNetv2


class SEResNet(NN_NIRS_Regressor):
    def build_model(self, input_shape, params):
        # Configurations
        length = input_shape[0]
        num_channel = input_shape[-1]  # 1  # Number of Channels in the Model
        model_width = 16  # Width of the Initial Layer, subsequent layers start from here
        problem_type = "Regression"  # Classification or Regression
        output_nums = 1  # Number of Class for Classification Problems, always '1' for Regression Problems
        reduction_ratio = 4
        # Build, Compile and Print Summary
        # model_name = 'SEResNet152'  # Modified DenseNet
        model = SEResNet(
            length,
            num_channel,
            model_width,
            ratio=reduction_ratio,
            problem_type=problem_type,
            output_nums=output_nums,
            pooling="avg",
            dropout_rate=False,
        ).SEResNet101()

        return model


class ResNetV2(NN_NIRS_Regressor):
    def build_model(self, input_shape, params):
        # Configurations
        length = input_shape[0]  # Length of each Segment
        num_channel = input_shape[-1]  # 1  # Number of Channels in the Model
        model_width = 16  # Width of the Initial Layer, subsequent layers start from here
        problem_type = "Regression"  # Classification or Regression
        output_nums = 1  # Number of Class for Classification Problems, always '1' for Regression Problems
        #
        # model_name = 'ResNet152'  # DenseNet Models
        model = ResNetv2(
            length, num_channel, model_width, problem_type=problem_type, output_nums=output_nums, pooling="avg", dropout_rate=False
        ).ResNet34()

        return model


class XCeption1D(NN_NIRS_Regressor):
    def xception_entry_flow(self, inputs):
        x = DepthwiseConv1D(kernel_size=3, strides=2, depth_multiplier=2)(inputs)
        # x = Conv1D(32, 3, strides=2, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = DepthwiseConv1D(kernel_size=3, strides=2, depth_multiplier=2)(x)
        # x = Conv1D(64, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        previous_block_activation = x

        for size in [128, 256, 728]:

            x = Activation("relu")(x)
            x = SeparableConv1D(size, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = Activation("relu")(x)
            x = SeparableConv1D(size, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = MaxPool1D(3, strides=2, padding="same")(x)

            residual = Conv1D(size, 1, strides=2, padding="same")(previous_block_activation)

            x = Add()([x, residual])
            previous_block_activation = x

        return x

    def xception_middle_flow(self, x, num_blocks=8):
        previous_block_activation = x
        for _ in range(num_blocks):

            x = Activation("relu")(x)
            x = SeparableConv1D(728, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = Activation("relu")(x)
            x = SeparableConv1D(728, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = Activation("relu")(x)
            x = SeparableConv1D(728, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = Add()([x, previous_block_activation])
            previous_block_activation = x

        return x

    def xception_exit_flow(self, x):
        previous_block_activation = x

        x = Activation("relu")(x)
        x = SeparableConv1D(728, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = SeparableConv1D(1024, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = MaxPool1D(3, strides=2, padding="same")(x)

        residual = Conv1D(1024, 1, strides=2, padding="same")(previous_block_activation)
        x = Add()([x, residual])

        x = Activation("relu")(x)
        x = SeparableConv1D(728, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = Activation("relu")(x)
        x = SeparableConv1D(1024, 3, padding="same")(x)
        x = BatchNormalization()(x)

        x = GlobalAveragePooling1D()(x)
        x = Dense(1, activation="sigmoid")(x)

        return x

    def build_model(self, input_shape, params):
        inputs = Input(shape=input_shape)
        outputs = self.xception_exit_flow(self.xception_middle_flow(self.xception_entry_flow(inputs)))
        return Model(inputs, outputs)


class FFT_Conv(NN_NIRS_Regressor):
    def build_model(self, input_shape, params):

        inputs = Input(shape=input_shape)
        x = SpatialDropout1D(0.2)(inputs)
        x = Permute((2, 1))(x)
        x = Lambda(lambda v: tf.cast(tf.signal.fft(tf.cast(v, dtype=tf.complex64)), tf.float32))(x)
        x = Permute((2, 1))(x)
        x = SeparableConv1D(64, kernel_size=3, strides=2, depth_multiplier=32, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = SeparableConv1D(64, kernel_size=3, strides=2, depth_multiplier=32, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        # x = SeparableConv1D(64, kernel_size=3, depth_multiplier=32, padding="same")(x)
        # x = BatchNormalization()(x)
        # x = Activation("relu")(x)
        # x = SeparableConv1D(64, kernel_size=3, depth_multiplier=32, padding="same")(x)
        # x = BatchNormalization()(x)
        # x = Activation("relu")(x)

        x = Conv1D(filters=32, kernel_size=5, strides=2, padding="same")(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Activation("relu")(x)

        # x = BatchNormalization()(x)
        # x = GlobalMaxPool1D()(x)

        x = Dense(128, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(1, activation="sigmoid")(x)

        outputs = x
        model = Model(inputs, outputs)
        model.summary()
        return model


class Custom_Residuals(NN_NIRS_Regressor):
    def build_model(self, input_shape, params):
        inputs = Input(shape=input_shape)
        x = SpatialDropout1D(0.2)(inputs)
        x = DepthwiseConv1D(kernel_size=3, strides=3, depth_multiplier=2, activation="relu")(x)
        x = DepthwiseConv1D(kernel_size=5, strides=3, activation="relu")(x)
        x = DepthwiseConv1D(kernel_size=5, strides=3, activation="relu")(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = MultiHeadAttention(key_dim=11, num_heads=4, dropout=0.1)(x, x)
        x = GlobalAveragePooling1D(data_format="channels_first")(x)
        x = Flatten()(x)
        x = Dense(16, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(1, activation="sigmoid")(x)

        outputs = x
        model = Model(inputs, outputs)
        model.summary()
        return model


class Custom_VG_Residuals(NN_NIRS_Regressor):
    def build_model(self, input_shape, params):

        inputs = Input(shape=input_shape)
        x = SpatialDropout1D(0.2)(inputs)
        x = DepthwiseConv1D(kernel_size=3, strides=3, depth_multiplier=2, activation="relu")(x)

        def block(x, strides):
            x = DepthwiseConv1D(kernel_size=3, strides=strides, activation="relu")(x)
            x = DepthwiseConv1D(kernel_size=5, strides=1, activation="relu")(x)
            x = DepthwiseConv1D(kernel_size=5, strides=1, activation="relu")(x)
            x = LayerNormalization(epsilon=1e-6)(x)
            x = MultiHeadAttention(key_dim=11, num_heads=4, dropout=0.1)(x, x)
            return x

        x = block(x, 2)
        x = block(x, 2)
        x = block(x, 2)
        x = block(x, 1)
        x = block(x, 1)
        x = block(x, 1)

        x = Conv1D(filters=16, kernel_size=1, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Conv1D(8, strides=8, kernel_size=8)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = GlobalAveragePooling1D(data_format="channels_first")(x)

        x = Flatten()(x)
        x = Dense(16, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(1, activation="sigmoid")(x)

        outputs = x
        model = Model(inputs, outputs)
        # model.summary()
        return model


class Custom_VG_Residuals2(NN_NIRS_Regressor):
    def build_model(self, input_shape, params):

        inputs = Input(shape=input_shape)

        x = SpatialDropout1D(0.2)(inputs)
        x1 = DepthwiseConv1D(kernel_size=3, padding="same", depth_multiplier=8, activation="relu")(x)
        x1 = DepthwiseConv1D(kernel_size=3, padding="same", depth_multiplier=2, activation="sigmoid")(x1)
        # x1 = MaxPool1D(pool_size=7, strides=7)(x1)
        # x1 = LayerNormalization()(x1)
        x2 = DepthwiseConv1D(kernel_size=7, padding="same", depth_multiplier=8, activation="relu")(x)
        x2 = DepthwiseConv1D(kernel_size=7, padding="same", depth_multiplier=2, activation="sigmoid")(x2)
        # x2 = MaxPool1D(pool_size=7, strides=7)(x2)
        # x2 = LayerNormalization()(x2)
        x3 = DepthwiseConv1D(kernel_size=15, padding="same", depth_multiplier=8, activation="relu")(x)
        x3 = DepthwiseConv1D(kernel_size=15, padding="same", depth_multiplier=2, activation="sigmoid")(x3)
        # x3 = MaxPool1D(pool_size=7, strides=7)(x3)
        # x3 = LayerNormalization()(x3)
        x = Concatenate(axis=2)([x1, x2, x3])
        x = BatchNormalization()(x)

        x = Conv1D(filters=64, kernel_size=7, strides=5, activation="relu")(x)
        # x = Conv1D(filters=128, kernel_size=7, strides=5, activation='relu')(x)
        x = Conv1D(filters=16, kernel_size=3, strides=3, activation="sigmoid")(x)
        # x = MaxPool1D(pool_size=3, strides=3)(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation="sigmoid")(x)
        x = Dense(1, activation="linear")(x)

        outputs = x
        model = Model(inputs, outputs)
        # model.summary()
        return model


class Bacon_VG(NN_NIRS_Regressor):
    def build_model(self, input_shape, params):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(SpatialDropout1D(0.2))
        model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="swish"))
        model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="swish"))
        model.add(MaxPool1D(pool_size=5, strides=3))
        model.add(SpatialDropout1D(0.2))
        model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation="swish"))
        model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation="swish"))
        model.add(MaxPool1D(pool_size=5, strides=3))
        model.add(SpatialDropout1D(0.2))
        model.add(Flatten())
        model.add(Dense(units=1024, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=1024, activation="relu"))
        model.add(Dense(units=1, activation="sigmoid"))
        # we compile the model with the custom Adam optimizer
        # model.compile(loss='mean_squared_error', metrics=['mse'], optimizer="adam")
        return model


class MLP(NN_NIRS_Regressor):
    def build_model(self, input_shape, params):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(Dropout(0.2))
        model.add(Dense(units=1024, activation="relu"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(units=1, activation="sigmoid"))
        # we compile the model with the custom Adam optimizer
        # model.compile(loss='mean_squared_error', metrics=['mse'], optimizer="adam")
        return model


class Bacon(NN_NIRS_Regressor):
    def build_model(self, input_shape, params):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(SpatialDropout1D(0.08))
        # model.add(SeqSelfAttention(attention_width=7, attention_activation='relu'))
        model.add(Conv1D(filters=8, kernel_size=15, strides=5, activation="selu"))
        model.add(Dropout(0.2))
        model.add(Conv1D(filters=64, kernel_size=21, strides=3, activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=32, kernel_size=5, strides=3, activation="elu"))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(16, activation="sigmoid"))
        model.add(Dense(1, activation="sigmoid"))
        # model.compile(loss='mean_squared_error', metrics=['mse'], optimizer="adam")
        return model


from Inception_1DCNN import Inception


class Inception1D(NN_NIRS_Regressor):
    def build_model(self, input_shape, params):
        length = input_shape[1]  # Number of Features (or length of the signal)
        model_width = 16  # Number of Filter or Kernel in the Input Layer (Power of 2 to avoid error)
        num_channel = 1  # Number of Input Channels
        problem_type = "Regression"  # Regression or Classification
        output_number = 1  # Number of Outputs in the Regression Mode - 1 input is mapped to a single output
        Regression_Model = Inception(length, num_channel, model_width, problem_type=problem_type, output_nums=output_number).Inception_v3()
        return Regression_Model


class Decon_SepDep(NN_NIRS_Regressor):
    def build_model(self, input_shape, params):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(SpatialDropout1D(0.2))

        model.add(DepthwiseConv1D(kernel_size=3, padding="same", depth_multiplier=64, activation="relu"))
        model.add(BatchNormalization())
        # model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu"))
        model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu"))
        model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu"))
        model.add(BatchNormalization())
        # model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu"))
        # model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu"))
        # model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu"))
        # model.add(BatchNormalization())
        model.add(Conv1D(filters=32, kernel_size=9, strides=6, padding="same", activation="relu"))
        model.add(Flatten())
        # model.add(BatchNormalization())
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation="sigmoid"))
        return model


# class Decon_Sep_Multiple(NN_NIRS_Regressor_Multiple):
#     def build_model(self, input_shape_global, params):
#         print("input_shape_global", input_shape_global)
#         input_shape = input_shape_global[2:]
#         start_models = []
#         for i in range(input_shape_global[0]):
#             input = Input(shape=input_shape)
#             x = SpatialDropout1D(0.2)(input)
#             x = SeparableConv1D(64, kernel_size=3, strides=2, depth_multiplier=32, padding="same", activation="relu")(x)
#             x = BatchNormalization()(x)
#             x = SeparableConv1D(64, kernel_size=3, depth_multiplier=32, strides=2, padding="same", activation="relu")(x)
#             x = BatchNormalization()(x)
#             x = SeparableConv1D(64, kernel_size=3, depth_multiplier=32, padding="same", activation="relu")(x)
#             x = BatchNormalization()(x)
#             x = SeparableConv1D(64, kernel_size=3, depth_multiplier=32, padding="same", activation="relu")(x)
#             x = BatchNormalization()(x)
#             model = Model(inputs=input, outputs=x)
#             start_models.append(model)
        
#         combined = concatenate([model.output for model in start_models])
#         x = SeparableConv1D(64, kernel_size=3, depth_multiplier=32, padding="same", activation="relu")(combined)
#         x = BatchNormalization()(x)
#         x = Conv1D(filters=32, kernel_size=5, strides=2, padding="same", activation="relu")(x)
#         x = Flatten()(x)
#         x = BatchNormalization()(x)
#         # model.add(Dense(units=128, activation="relu"))
#         x = Dense(units=32, activation="relu")(x)
#         x = Dropout(0.2)(x)
#         z = Dense(units=1, activation="sigmoid")(x)

#         model = Model(inputs=[model.input for model in start_models], outputs=z)
#         return model


class Decon_Sep(NN_NIRS_Regressor):
    def build_model(self, input_shape, params):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(SpatialDropout1D(0.2))        
        model.add(SeparableConv1D(64, kernel_size=3, strides=2, depth_multiplier=32, padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(SeparableConv1D(64, kernel_size=3, strides=2, depth_multiplier=32, padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=32, padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=32, padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=32, kernel_size=5, strides=2, padding="same", activation="relu"))
        model.add(Flatten())
        model.add(BatchNormalization())
        # model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation="sigmoid"))
        return model


class Decon_Sep_VG(NN_NIRS_Regressor):
    def build_model(self, input_shape, params):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(BatchNormalization())
        model.add(SpatialDropout1D(0.2))
        model.add(SeparableConv1D(64, kernel_size=3, strides=2, depth_multiplier=64, padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(SeparableConv1D(64, kernel_size=3, strides=2, depth_multiplier=64, padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=32, kernel_size=5, strides=2, padding="same", activation="relu"))
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation="sigmoid"))
        return model


class Decon_SepPo(NN_NIRS_Regressor):
    def build_model(self, input_shape, params):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(SpatialDropout1D(0.3))
        model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        # model.add(SeparableConv1D(128, kernel_size=3, depth_multiplier=128, padding="same", activation="relu"))
        # model.add(MaxPooling1D(pool_size=3, strides=2, padding="same"))
        # model.add(BatchNormalization())
        # model.add(SeparableConv1D(32, kernel_size=3, depth_multiplier=32, padding="same", activation="relu"))
        # model.add(MaxPooling1D(pool_size=3, strides=2, padding="same"))
        # model.add(BatchNormalization())
        model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=32, kernel_size=3, strides=2, padding="same", activation="relu"))
        # model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation="sigmoid"))
        return model


class Decon_SepRes(NN_NIRS_Regressor):
    def build_model(self, input_shape, params):
        inputs = Input(shape=input_shape)
        x = inputs
        x1 = SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu")(x)
        x1 = BatchNormalization()(x1)
        x1 = SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu")(x1)
        x1 = BatchNormalization()(x1)

        x3 = SeparableConv1D(64, kernel_size=7, depth_multiplier=64, padding="same", activation="relu")(x)
        x3 = BatchNormalization()(x3)
        x3 = SeparableConv1D(64, kernel_size=7, depth_multiplier=64, padding="same", activation="relu")(x3)
        x3 = BatchNormalization()(x3)

        x5 = SeparableConv1D(64, kernel_size=15, depth_multiplier=64, padding="same", activation="relu")(x)
        x5 = BatchNormalization()(x5)
        x5 = SeparableConv1D(64, kernel_size=15, depth_multiplier=64, padding="same", activation="relu")(x5)
        x5 = BatchNormalization()(x5)

        x = Concatenate(axis=2)([x1, x3, x5])
        x = SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu")(x)
        # x = SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Conv1D(filters=32, kernel_size=3, strides=2, padding="same", activation="relu")(x)

        x = Flatten()(x)
        x = Dense(units=128, activation="relu")(x)
        x = Dense(units=32, activation="relu")(x)
        x = Dropout(0.2)(x)
        outputs = Dense(units=1, activation="sigmoid")(x)
        return Model(inputs, outputs)


class Decon(NN_NIRS_Regressor):
    def build_model(self, input_shape, params):
        model = Sequential()
        model.add(Input(shape=input_shape))
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
        # model.compile(loss='mean_squared_error', metrics=['mse'], optimizer="adam")
        return model


class Decon_Layer(NN_NIRS_Regressor):
    def build_model(self, input_shape, params):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(SpatialDropout1D(0.2))
        model.add(DepthwiseConv1D(kernel_size=7, padding="same", depth_multiplier=2, activation="relu"))
        model.add(DepthwiseConv1D(kernel_size=7, padding="same", depth_multiplier=2, activation="relu"))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(LayerNormalization())
        model.add(DepthwiseConv1D(kernel_size=5, padding="same", depth_multiplier=2, activation="relu"))
        model.add(DepthwiseConv1D(kernel_size=5, padding="same", depth_multiplier=2, activation="relu"))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(LayerNormalization())
        model.add(DepthwiseConv1D(kernel_size=9, padding="same", depth_multiplier=2, activation="relu"))
        model.add(DepthwiseConv1D(kernel_size=9, padding="same", depth_multiplier=2, activation="relu"))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(LayerNormalization())
        model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=1, padding="same", activation="relu"))
        model.add(Conv1D(filters=32, kernel_size=3, padding="same"))
        model.add(MaxPooling1D(pool_size=5, strides=3))
        model.add(SpatialDropout1D(0.1))
        model.add(Flatten())
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation="sigmoid"))
        # model.compile(loss='mean_squared_error', metrics=['mse'], optimizer="adam")
        return model


class Transformer_NIRS(NN_NIRS_Regressor):
    def transformer_encoder_nirs(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # x = Conv1D(filters=64, kernel_size=15, strides=15, activation='relu')(x)
        # x = Conv1D(filters=8, kernel_size=15, strides=15, activation='relu')(x)
        # Attention and Normalization
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        inputs = tf.cast(inputs, tf.float16)
        res = x + inputs

        # Feed Forward Part
        x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = Dropout(dropout)(x)
        x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        res = tf.cast(res, tf.float16)
        return x + res

    def build_model(self, input_shape, params):
        head_size = 32
        num_heads = 2
        ff_dim = 8
        num_transformer_blocks = 4
        mlp_units = [16]
        dropout = 0.05
        mlp_dropout = 0.1

        inputs = Input(shape=input_shape)
        x = SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu")(inputs)
        x = BatchNormalization()(x)
        x = SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu")(x)
        # x = SeparableConv1D(64, kernel_size=3, depth_multiplier=64, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        # x = inputs
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder_nirs(x, head_size, num_heads, ff_dim, dropout)

        x = GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(mlp_dropout)(x)
        outputs = Dense(1, activation="sigmoid")(x)
        return Model(inputs, outputs)


class Abstract_Transformer(NN_NIRS_Regressor):
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # x = Conv1D(filters=64, kernel_size=15, strides=15, activation='relu')(x)
        # x = Conv1D(filters=8, kernel_size=15, strides=15, activation='relu')(x)
        # Attention and Normalization
        inputs = tf.cast(inputs, tf.float16)
        x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = Dropout(dropout)(x)
        x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        res = tf.cast(res, tf.float16)
        return x + res

    def transformer_model(
        self,
        input_shape,
        head_size=16,
        num_heads=2,
        ff_dim=8,
        num_transformer_blocks=1,
        mlp_units=[32, 8],
        dropout=0.05,
        mlp_dropout=0.1,
    ):
        inputs = Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = Dense(dim, activation="relu")(x)
            x = Dropout(mlp_dropout)(x)
        outputs = Dense(1, activation="sigmoid")(x)
        return Model(inputs, outputs)


class Transformer_VG(Abstract_Transformer):
    def build_model(self, input_shape, params):
        return super().transformer_model(
            input_shape,
            head_size=16,
            num_heads=32,
            ff_dim=8,
            num_transformer_blocks=1,
            mlp_units=[32, 8],
            dropout=0.05,
            mlp_dropout=0.1,
        )


# class Transformer_VG_Multiple(NN_NIRS_Regressor_Multiple):

#     def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
#         # x = Conv1D(filters=64, kernel_size=15, strides=15, activation='relu')(x)
#         # x = Conv1D(filters=8, kernel_size=15, strides=15, activation='relu')(x)
#         # Attention and Normalization
#         inputs = tf.cast(inputs, tf.float16)
#         x = MultiHeadAttention(
#             key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
#         x = LayerNormalization(epsilon=1e-6)(x)
#         x = Dropout(dropout)(x)
#         res = x + inputs

#         # Feed Forward Part
#         x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
#         x = Dropout(dropout)(x)
#         x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
#         x = LayerNormalization(epsilon=1e-6)(x)
#         res = tf.cast(res, tf.float16)
#         return x + res

#     def transformer_model(
#         self,
#         input_shape,
#         head_size=16,
#         num_heads=2,
#         ff_dim=4,
#         num_transformer_blocks=2,
#         mlp_units=[32, 8],
#         dropout=0.05,
#         mlp_dropout=0.1,
#     ):
#         inputs = Input(shape=input_shape)
#         x = inputs
#         for _ in range(num_transformer_blocks):
#             x = self.transformer_encoder(
#                 x, head_size, num_heads, ff_dim, dropout)

#         x = GlobalAveragePooling1D(data_format="channels_first")(x)
#         output = BatchNormalization()(x)
#         return Model(inputs, output)

#     def build_model(self, input_shape_global, params):
#         print("input_shape_global", input_shape_global)
#         input_shape = input_shape_global[2:]
#         start_models = []
#         for i in range(input_shape_global[0]):
#             model = self.transformer_model(
#                 input_shape,
#                 head_size=16,
#                 num_heads=32,
#                 ff_dim=8,
#                 num_transformer_blocks=1,
#                 mlp_units=[32, 8],
#                 dropout=0.05,
#                 mlp_dropout=0.1,
#             )
#             start_models.append(model)

#         combined = concatenate([model.output for model in start_models])
#         # x = SeparableConv1D(64, kernel_size=3, depth_multiplier=32, padding="same", activation="relu")(combined)
#         # x = BatchNormalization()(x)
#         # x = Conv1D(filters=32, kernel_size=5, strides=2, padding="same", activation="relu")(x)
#         # x = Flatten()(x)
#         # x = BatchNormalization()(x)
#         # model.add(Dense(units=128, activation="relu"))
#         x = Dense(units=32, activation="relu")(combined)
#         x = Dropout(0.2)(x)
#         z = Dense(units=1, activation="sigmoid")(x)

#         model = Model(inputs=[model.input for model in start_models], outputs=z)
#         return model


class Transformer_LongRange(Abstract_Transformer):
    def build_model(self, input_shape, params):
        return super().transformer_model(
            input_shape,
            head_size=512,
            num_heads=8,
            ff_dim=8,
            num_transformer_blocks=2,
            mlp_units=[8],
            dropout=0.05,
            mlp_dropout=0.1,
        )


class Transformer(Abstract_Transformer):
    def build_model(self, input_shape, params):
        return super().transformer_model(
            input_shape,
            head_size=8,
            num_heads=2,
            ff_dim=4,
            num_transformer_blocks=1,
            mlp_units=[8],
            dropout=0.05,
            mlp_dropout=0.1,
        )


class Transformer_Max(Abstract_Transformer):
    def build_model(self, input_shape, params):
        return super().transformer_model(
            input_shape,
            head_size=16,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=6,
            mlp_units=[16],
            dropout=0.05,
            mlp_dropout=0.1,
        )


# def nn_list():
#     # return [custom_residuals]
#     # return [vgg1D]
#     # return [transformer_vg]
#     return [transformer_nirs]
#     # return [bacon, decon, transformer]
#     # return [bacon, bacon_vg, decon, decon_layer, transformer, xception1D]


class ML_Regressor(NIRS_Regressor):
    def __init__(self, model_class, *, params={}, name=""):
        self.model_class = model_class
        NIRS_Regressor.__init__(self, params=params, name=name)

    def model(self, X_train, y_train=None, X_test=None, y_test=None, *, run_name="", cb=None, params=None, desc=None, discretizer=None):
        signature = inspect.signature(self.model_class.__init__)
        if "X_train" in signature.parameters:
            params["X_train"] = X_train
        if "y_train" in signature.parameters:
            params["y_train"] = y_train
        if "X_test" in signature.parameters:
            params["X_test"] = X_test
        if "y_test" in signature.parameters:
            params["y_test"] = y_test
            
        return self.model_class(**params)

    def name(self):
        # if self.name == "":
        return re.search(".*'(.*)'.*", str(self.model_class)).group(1)
        # else:
            # return self.name




from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import RBFSampler

class NonlinearPLSRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=5, poly_degree=2, gamma=0.1):
        self.n_components = n_components
        self.poly_degree = poly_degree
        self.gamma = gamma
        self.pipeline = None

    def fit(self, X, y):
        self.pipeline = make_pipeline(
            # PolynomialFeatures(degree=self.poly_degree),
            RBFSampler(gamma=self.gamma, n_components=8),
            PLSRegression(n_components=self.n_components)
        )
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)
    
    

# from sklearn.base import BaseEstimator, RegressorMixin
# from sklearn.utils.validation import check_X_y, check_array
# import numpy as np

# class LWPLS(BaseEstimator, RegressorMixin):
#     def __init__(self, n_components=2, weighting='bisquare'):
#         self.n_components = n_components
#         self.weighting = weighting
    
#     def fit(self, X, y):
#         print(">", y.shape)
#         if isinstance(y, np.ndarray):
#             y = y.reshape(-1, 1)
#         else:
#             y = np.array(y).reshape(-1, 1)
#         print(">", y.shape, X.shape)
        
#         if len(X.shape) == 1:
#             X = X.reshape(-1, 1)
#         # Check that X and y have correct shape
#         X, y = check_X_y(X, y)
#         print(">", y.shape, X.shape)
#         # y = y.reshape(-1, 1)
        
#         # Store the number of samples and variables
#         self.n_samples_, self.n_variables_ = X.shape
        
#         # Initialize the scores and loading vectors
#         T = np.zeros((self.n_samples_, self.n_components))
#         W = np.zeros((self.n_variables_, self.n_components))
        
#         # Center the data
#         self.x_mean_ = np.mean(X, axis=0)
#         self.y_mean_ = np.mean(y)
#         print(">>", self.x_mean_.shape, self.y_mean_)
#         Xc = X - self.x_mean_
#         yc = y - self.y_mean_
        
#         # Iterate over the components
#         for i in range(self.n_components):
#             # Initialize the weights for the first component
#             if i == 0:
#                 w = Xc.T @ yc
#             else:
#                 w = Xc.T @ (yc - T[:, :i] @ np.linalg.pinv(T[:, :i].T @ T[:, :i]) @ T[:, :i].T @ yc)
            
#             # Normalize the weights
#             w /= np.linalg.norm(w)
            
#             # Calculate the scores
#             t = Xc @ w
            
#             # Update the loading vectors
#             W[:, i] = w.ravel()
            
#             # Deflate the data
#             p = (Xc.T @ t) / (t.T @ t)
#             Xc -= t.reshape(-1, 1) @ p.reshape(1, -1)
        
#         # Calculate the regression coefficients
#         B = W @ np.linalg.pinv(T.T @ T) @ T.T @ y
        
#         # Store the coefficients
#         self.coef_ = B.ravel()
        
#         return self
    
#     def predict(self, X):
#         # Check that X has correct shape
#         X = check_array(X)
        
#         # Apply a weighting scheme to the test samples
#         D = np.sum((X - self.x_mean_)**2, axis=-1)
        
#         if self.weighting == 'bisquare':
#             w = (1 - (D / np.max(D))**2)**2
#         elif self.weighting == 'tricube':
#             w = (1 - np.abs(D / np.max(D))**3)**3
#         else:
#             w = 1 / D
        
#         # Make predictions for each test sample
#         y_pred = []
        
#         for i in range(X.shape[0]):
#             # Fit a weighted PLS model to the training data using NIPALS algorithm
#             lwpls = LWPLS(n_components=self.n_components)
#             lwpls.fit(self.x_mean_, self.y_mean_)
            
#             # Update the regression coefficients using sample weights
#             lwpls.coef_ *= w[i]
            
#             # Make a prediction for the current test sample
#             y_pred.append(lwpls.predict(X[i].reshape(1, -1)))
        
#         return np.array(y_pred).ravel()





# from sklearn.base import BaseEstimator, RegressorMixin
# from sklearn.linear_model import Ridge
# import numpy as np

# class RIDGE(BaseEstimator, RegressorMixin):
#     def __init__(self, alpha=1.0, weighting='bisquare'):
#         self.alpha = alpha
#         self.weighting = weighting
#         self.ridge = Ridge(alpha=alpha)

#     def fit(self, X_train, y_train):
#         self.X_train = X_train
#         self.y_train = y_train
#         self.ridge.fit(X_train, y_train)
#         return self

#     def predict(self, X_test):
#         # Calculate the distance between each test sample and the training samples
#         D = np.sum((self.X_train - X_test[:, np.newaxis])**2, axis=-1)

#         # Apply a weighting scheme
#         if self.weighting == 'bisquare':
#             w = (1 - (D / np.max(D))**2)**2
#         elif self.weighting == 'tricube':
#             w = (1 - np.abs(D / np.max(D))**3)**3
#         else:
#             w = 1 / D

#         # Build local models for each test sample
#         y_pred = []
#         for i in range(X_test.shape[0]):
#             self.ridge.fit(self.X_train, self.y_train, sample_weight=w[:, i])
#             y_pred.append(self.ridge.predict(X_test[i].reshape(1, -1)))

#         return np.array(y_pred).ravel()


# def pls_generator(start, end, step):
#     funcs = []
#     for nc in range(start, end, step):

#         def pls(X_test, y_test, seed, nc=nc):
#             return (PLSRegression(nc, max_iter=5000), "PLS_" + str(nc))

#         funcs.append(pls)
#     return funcs


# def xgboot_generator(n_estimators=100, max_depth=None):
#     def xgboost(X_test, y_test, seed, n_estimators=n_estimators, max_depth=max_depth):
#         return (
#             XGBRegressor(n_estimators=200, max_depth=50, seed=seed),
#             "XGBoost_" + str(nc) + "_" + str(max_depth),
#         )

#     return xgboost


# def lwpls_generator(n_components=100):
#     def lwpls(X_test, y_test, seed, n_components=n_components):
#         return (
#             LWPLS(n_components, 2**-2, X_test, y_test),
#             "LWPLS_" + str(n_components),
#         )

#     return lwpls


# def get_ml_model(model_func, X_test, y_test, seed=0):
#     return model_func(X_test, y_test, seed)


# def ml_list(SEED, X_test, y_test):
#     # ml_models = [(PLSRegression(nc, max_iter=5000), "PLS" + str(nc)) for nc in range(4, 12, 4)] # test
#     # ml_models = [(PLSRegression(nc, max_iter=5000), "PLS" + str(nc)) for nc in range(4, 100, 4)]
#     # ml_models.append((XGBRegressor(seed=SEED), 'XGBoost_100_None'))
#     # ml_models.append((XGBRegressor(n_estimators=200, max_depth=50, seed=SEED), 'XGBoost_200_10'))
#     # ml_models.append((XGBRegressor(n_estimators=50, max_depth=100, seed=SEED), 'XGBoost_50_100'))
#     # ml_models.append((XGBRegressor(n_estimators=200, seed=SEED), 'XGBoost_200_'))
#     # ml_models.append((XGBRegressor(n_estimators=400, max_depth=100, seed=SEED), 'XGBoost_400_100'))

#     # ml_models.append((LWPLS(2, 2 ** -2, X_test, y_test), "LWPLS_2_0.25"))
#     # ml_models.append((LWPLS(16, 2 ** -2, X_test, y_test), "LWPLS_16_0.25"))
#     # ml_models.append((LWPLS(30, 2 ** -2, X_test, y_test), "LWPLS_30_0.25"))

#     return ml_models
