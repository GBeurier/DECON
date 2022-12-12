import datetime
import math
import numpy as np

from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense, Conv1D, Activation, SpatialDropout1D, BatchNormalization, Flatten, Dropout, \
                                    Input, MaxPool1D, SeparableConv1D, Add, GlobalAveragePooling1D, Concatenate, \
                                    DepthwiseConv1D, MaxPooling1D, LayerNormalization, MultiHeadAttention, SeparableConv1D, \
                                    LocallyConnected1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import Callback, EarlyStopping
from scikeras.wrappers import KerasRegressor

from keras_self_attention import SeqSelfAttention
# from transformers import Transformer
from lwpls import LWPLS


def vgg_block(layer_in, n_filters, n_conv):
    # add convolutional layers
    for _ in range(n_conv):
        layer_in = Conv1D(filters=n_filters, kernel_size=3, padding='same', activation="relu")(layer_in)
    layer_in = MaxPooling1D(2, strides=2)(layer_in)
    return layer_in


def vgg1D(meta):
    input_shape = meta["X_shape_"][1:]
    visible = Input(shape=input_shape)
    layer = vgg_block(visible, 64, 2)
    layer = vgg_block(layer, 128, 2)
    layer = vgg_block(layer, 256, 2)
    layer = Flatten()(layer)
    layer = Dense(units=16, activation="sigmoid")(layer)
    layer = Dense(units=1, activation="linear")(layer)
    model = Model(inputs=visible, outputs=layer)
    # model.compile(loss='root_mean_squared_error', metrics=['root_mean_squared_error'], optimizer="adam")
    return model


def xception_entry_flow(inputs):
    x = Conv1D(32, 3, strides=2, padding='same')(inputs)
    x = SpatialDropout1D(0.3)(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    previous_block_activation = x

    for size in [128, 256, 728]:

        x = Activation('relu')(x)
        x = SeparableConv1D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv1D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = MaxPool1D(3, strides=2, padding='same')(x)

        residual = Conv1D(size, 1, strides=2, padding='same')(previous_block_activation)

        x = Add()([x, residual])
        previous_block_activation = x

    return x


def xception_middle_flow(x, num_blocks=8):
    previous_block_activation = x
    for _ in range(num_blocks):

        x = Activation('relu')(x)
        x = SeparableConv1D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv1D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv1D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Add()([x, previous_block_activation])
        previous_block_activation = x

    return x


def xception_exit_flow(x):
    previous_block_activation = x

    x = Activation('relu')(x)
    x = SeparableConv1D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv1D(1024, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPool1D(3, strides=2, padding='same')(x)

    residual = Conv1D(1024, 1, strides=2, padding='same')(previous_block_activation)
    x = Add()([x, residual])

    x = Activation('relu')(x)
    x = SeparableConv1D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv1D(1024, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(1, activation='linear')(x)

    return x


def xception1D(meta):
    input_shape = meta["X_shape_"][1:]
    inputs = Input(shape=input_shape)
    outputs = xception_exit_flow(xception_middle_flow(xception_entry_flow(inputs)))
    return Model(inputs, outputs)


def transformer_depthwise_encoder_custom(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = Conv1D(filters=32, kernel_size=7, strides=3, activation='relu')
    x = Conv1D(filters=8, kernel_size=7, strides=3, activation='relu')
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


def custom_VG_residuals(meta):
    input_shape = meta["X_shape_"][1:]

    inputs = Input(shape=input_shape)
    x = SpatialDropout1D(0.2)(inputs)
    x = DepthwiseConv1D(kernel_size=3, strides=3, depth_multiplier=2, activation='relu')(x)
    x = DepthwiseConv1D(kernel_size=5, strides=3, activation='relu')(x)
    x = DepthwiseConv1D(kernel_size=5, strides=3, activation='relu')(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = MultiHeadAttention(key_dim=11, num_heads=4, dropout=0.1)(x, x)

    # x = SpatialDropout1D(0.2)(inputs)
    # x1 = DepthwiseConv1D(kernel_size=3, padding="same", depth_multiplier=8, activation='relu')(x)
    # x1 = DepthwiseConv1D(kernel_size=3, padding="same", depth_multiplier=2, activation='softmax')(x1)
    # # x1 = LayerNormalization()(x1)
    # x2 = DepthwiseConv1D(kernel_size=7, padding="same", depth_multiplier=8, activation='relu')(x)
    # x2 = DepthwiseConv1D(kernel_size=7, padding="same", depth_multiplier=2, activation='softmax')(x2)
    # # x2 = LayerNormalization()(x2)
    # x3 = DepthwiseConv1D(kernel_size=15, padding="same", depth_multiplier=8, activation='relu')(x)
    # x3 = DepthwiseConv1D(kernel_size=15, padding="same", depth_multiplier=2, activation='softmax')(x3)
    # # x3 = LayerNormalization()(x3)
    # x = Concatenate(axis=2)([x1, x2, x3])
    # x = BatchNormalization()(x)



    # # Feed Forward Part
    # x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    # x = Dropout(dropout)(x)
    # x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    # x = LayerNormalization(epsilon=1e-6)(x)
    # res = tf.cast(res, tf.float16)
    # return x + res


    x = GlobalAveragePooling1D(data_format="channels_first")(x)

    # x = LayerNormalization(epsilon=1e-6)(x)
    # inputs = tf.cast(inputs, tf.float16)
    # res = x + inputs

    # x = Conv1D(filters=64, kernel_size=7, strides=5, activation='relu')(x)
    # x = Conv1D(filters=128, kernel_size=7, strides=5, activation='relu')(x)
    # x = Conv1D(filters=16, kernel_size=7, strides=5, activation='relu')(x)
    # # x = MaxPool1D(pool_size=3, strides=3)(x)
    # x = BatchNormalization()(x)
    # res = x + inputs

    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation='sigmoid')(x)

    outputs = x
    model = Model(inputs, outputs)
    model.summary()
    return model


def custom_VG_residuals2(meta):
    input_shape = meta["X_shape_"][1:]
    inputs = Input(shape=input_shape)

    x = SpatialDropout1D(0.2)(inputs)
    x1 = DepthwiseConv1D(kernel_size=3, padding="same", depth_multiplier=8, activation='relu')(x)
    x1 = DepthwiseConv1D(kernel_size=3, padding="same", depth_multiplier=2, activation='sigmoid')(x1)
    # x1 = MaxPool1D(pool_size=7, strides=7)(x1)
    # x1 = LayerNormalization()(x1)
    x2 = DepthwiseConv1D(kernel_size=7, padding="same", depth_multiplier=8, activation='relu')(x)
    x2 = DepthwiseConv1D(kernel_size=7, padding="same", depth_multiplier=2, activation='sigmoid')(x2)
    # x2 = MaxPool1D(pool_size=7, strides=7)(x2)
    # x2 = LayerNormalization()(x2)
    x3 = DepthwiseConv1D(kernel_size=15, padding="same", depth_multiplier=8, activation='relu')(x)
    x3 = DepthwiseConv1D(kernel_size=15, padding="same", depth_multiplier=2, activation='sigmoid')(x3)
    # x3 = MaxPool1D(pool_size=7, strides=7)(x3)
    # x3 = LayerNormalization()(x3)
    x = Concatenate(axis=2)([x1, x2, x3])
    x = BatchNormalization()(x)

    x = Conv1D(filters=64, kernel_size=7, strides=5, activation='relu')(x)
    # x = Conv1D(filters=128, kernel_size=7, strides=5, activation='relu')(x)
    x = Conv1D(filters=16, kernel_size=3, strides=3, activation='sigmoid')(x)
    # x = MaxPool1D(pool_size=3, strides=3)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='sigmoid')(x)
    x = Dense(1, activation='linear')(x)

    outputs = x
    model = Model(inputs, outputs)
    model.summary()
    return model


def bacon_vg(meta):
    input_shape = meta["X_shape_"][1:]
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(0.2))
    model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation='swish'))
    model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation='swish'))
    model.add(MaxPool1D(pool_size=5,strides=3))
    model.add(SpatialDropout1D(0.2))
    model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation='swish'))
    model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation='swish'))
    model.add(MaxPool1D(pool_size=5,strides=3))
    model.add(SpatialDropout1D(0.2))
    model.add(Flatten())
    model.add(Dense(units=1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units=1024, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))
    # we compile the model with the custom Adam optimizer
    # model.compile(loss='mean_squared_error', metrics=['mse'], optimizer="adam")
    return model


def bacon(meta):
    input_shape = meta["X_shape_"][1:]
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(0.08))
    # model.add(SeqSelfAttention(attention_width=7, attention_activation='relu'))
    model.add(Conv1D(filters=8, kernel_size=15, strides=5, activation='selu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=64, kernel_size=21, strides=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=32, kernel_size=5, strides=3, activation='elu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='mean_squared_error', metrics=['mse'], optimizer="adam")
    return model


def decon(meta):
    input_shape = meta["X_shape_"][1:]
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(0.2))
    model.add(DepthwiseConv1D(kernel_size=7, padding="same", depth_multiplier=2, activation='relu'))
    model.add(DepthwiseConv1D(kernel_size=7, padding="same", depth_multiplier=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(DepthwiseConv1D(kernel_size=5, padding="same", depth_multiplier=2, activation='relu'))
    model.add(DepthwiseConv1D(kernel_size=5, padding="same", depth_multiplier=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(DepthwiseConv1D(kernel_size=9, padding="same", depth_multiplier=2, activation='relu'))
    model.add(DepthwiseConv1D(kernel_size=9, padding="same", depth_multiplier=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(BatchNormalization())
    model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=1, padding="same", activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=3, padding="same"))
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


def decon_layer(meta):
    input_shape = meta["X_shape_"][1:]
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(SpatialDropout1D(0.2))
    model.add(DepthwiseConv1D(kernel_size=7, padding="same", depth_multiplier=2, activation='relu'))
    model.add(DepthwiseConv1D(kernel_size=7, padding="same", depth_multiplier=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(LayerNormalization())
    model.add(DepthwiseConv1D(kernel_size=5, padding="same", depth_multiplier=2, activation='relu'))
    model.add(DepthwiseConv1D(kernel_size=5, padding="same", depth_multiplier=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(LayerNormalization())
    model.add(DepthwiseConv1D(kernel_size=9, padding="same", depth_multiplier=2, activation='relu'))
    model.add(DepthwiseConv1D(kernel_size=9, padding="same", depth_multiplier=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(LayerNormalization())
    model.add(SeparableConv1D(64, kernel_size=3, depth_multiplier=1, padding="same", activation='relu'))
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


def transformer_depthwise_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = Conv1D(filters=32, kernel_size=7, strides=3, activation='relu')
    x = Conv1D(filters=8, kernel_size=7, strides=3, activation='relu')
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


def transformer_nirs(
    meta,
    head_size=8,
    num_heads=2,
    ff_dim=8,
    num_transformer_blocks=1,
    mlp_units=[8],
    dropout=0.2,
    mlp_dropout=0.2,
):
    input_shape = meta["X_shape_"][1:]
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_depthwise_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    # model.summary()
    return model





def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
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


def transformer_vg(
    meta,
    head_size=16,
    num_heads=2,
    ff_dim=8,
    num_transformer_blocks=1,
    mlp_units=[8],
    dropout=0.05,
    mlp_dropout=0.1,
):
    input_shape = meta["X_shape_"][1:]
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs)


def transformer(
    meta,
    head_size=24,
    num_heads=2,
    ff_dim=4,
    num_transformer_blocks=2,
    mlp_units=[32],
    dropout=0.05,
    mlp_dropout=0.1,
):
    input_shape = meta["X_shape_"][1:]
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs)


def transformer_max(
    meta,
    head_size=16,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=6,
    mlp_units=[16],
    dropout=0.05,
    mlp_dropout=0.1,
):
    input_shape = meta["X_shape_"][1:]
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs)


class Auto_Save(Callback):
    best_weights = []

    def __init__(self, model_func, shape, cb_func=None):
        super(Auto_Save, self).__init__()
        self.model_name = model_func.__name__
        self.shape = shape
        self.best = np.Inf
        self.cb = cb_func

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        print('epoch', str(epoch).zfill(5), end='\r')
        if np.less(current_loss, self.best):
            self.best = current_loss            
            Auto_Save.best_weights = self.model.get_weights()
            self.best_epoch = epoch
            if self.cb is not None:
                self.cb(epoch, self.best)
            # print("Best so far >", self.best)

    def on_train_end(self, logs=None):
        # if self.params['verbose'] == 2:
        print('Saved best {0:6.4f} at epoch'.format(self.best), self.best_epoch)
        self.model.set_weights(Auto_Save.best_weights)
        # self.model.save_weights(self.model_name + "-".join(self.shape[1:] + ".hdf5")


class Print_LR(Callback):
    def on_epoch_end(self, epoch, logs=None):
        iteration = self.model.optimizer.iterations.numpy()
        # lr = clr(iteration).numpy()
        lr = self.model.optimizer.learning_rate
        if self.params['verbose'] == 2:
            print("Iteration {} - Learning rate: {}".format(iteration, lr))


def scale_fn(x):
    # return 1. ** x
    return 1 / (2.0 ** (x - 1))


# def calc_lr(step, warmup_steps=200):
#     return 2**(-0.5) * min(step**(-0.5), (step+1) * warmup_steps**(-1.5))

def clr(epoch):
    cycle_params = {
        'MIN_LR': 1e-5,
        'MAX_LR': 1e-2,
        'CYCLE_LENGTH': 64,
    }
    MIN_LR, MAX_LR, CYCLE_LENGTH = cycle_params['MIN_LR'], cycle_params['MAX_LR'], cycle_params['CYCLE_LENGTH']
    initial_learning_rate = MIN_LR
    maximal_learning_rate = MAX_LR
    step_size = CYCLE_LENGTH
    step_as_dtype = float(epoch)
    cycle = math.floor(1 + step_as_dtype / (2 * step_size))
    x = abs(step_as_dtype / step_size - 2 * cycle + 1)
    mode_step = cycle  # if scale_mode == "cycle" else step
    return initial_learning_rate + (maximal_learning_rate - initial_learning_rate) * max(0, (1 - x)) * scale_fn(mode_step)


def nn_list():
    return [custom_VG_residuals]
    # return [vgg1D]
    # return [transformer_vg]
    # return [bacon, decon, transformer]
    # return [bacon, bacon_vg, decon, decon_layer, transformer, xception1D]


def rmse_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean((y_true - y_pred)**2))


# def coeff_determination(y_true, y_pred):
#     # SS_res = tf.sum(tf.square(y_true - y_pred))
#     # SS_tot = tf.sum(tf.square(y_true - tf.mean(y_true)))
#     # return (1 - SS_res/(SS_tot + tf.epsilon()))


def get_keras_model(run_name, model_func, epochs, batch_size, X_test, y_test, *, callback_func=None, transfer=False, verbose=2, seed=0):

    early_stop = EarlyStopping(monitor='val_loss', patience=256, verbose=0, mode='min') 
    log_dir = "logs/fit/"+run_name+"/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    lrScheduler = tf.keras.callbacks.LearningRateScheduler(clr)

    auto_save = Auto_Save(model_func, X_test.shape, callback_func)
    callbacks = [auto_save, early_stop, tensorboard_callback, lrScheduler]
    if model_func == transformer_vg:
        batch_size = 15
        epochs = 512
        # callbacks = callbacks[:-1]

    if model_func == transformer_nirs:
        batch_size = 50
        # callbacks = callbacks[:-1]

    if model_func == transformer_max:
        batch_size = 25
        callbacks = callbacks[:-1]

    if transfer:
        meta = {
            "X_shape_": X_test.shape
        }
        model_instance = model_func(meta)

        # weights_name = self.model_name + "-".join(self.shape[1:]) + ".hdf5"
        rmse = tf.keras.metrics.RootMeanSquaredError()
        k_regressor = KerasRegressor(
            model=model_instance,
            loss=rmse_loss,
            metrics=[rmse],
            optimizer="adam",
            callbacks=callbacks,
            epochs=epochs,
            batch_size=batch_size,
            fit__validation_data=(X_test, y_test),
            verbose=verbose,
            )

        return k_regressor

    k_regressor = KerasRegressor(
        model=model_func,
        loss='mean_squared_error', metrics=['root_mean_squared_error'],
        optimizer="adam",
        callbacks=callbacks,
        epochs=epochs,
        batch_size=batch_size,
        fit__validation_data=(X_test, y_test),
        verbose=verbose,
        )

    return k_regressor


def pls_generator(start, end, step):
    funcs = []
    for nc in range(start, end, step):
        def pls(X_test, y_test, seed, nc=nc):
            return (PLSRegression(nc, max_iter=5000), 'PLS_' + str(nc))
        funcs.append(pls)
    return funcs


def xgboot_generator(n_estimators=100, max_depth=None):
    def xgboost(X_test, y_test, seed, n_estimators=n_estimators, max_depth=max_depth):
        return (XGBRegressor(n_estimators=200, max_depth=50, seed=seed), 'XGBoost_' + str(nc) + '_' + str(max_depth))
    return xgboost


def lwpls_generator(n_components=100):
    def lwpls(X_test, y_test, seed, n_components=n_components):
        return (LWPLS(n_components, 2 ** -2, X_test, y_test), 'LWPLS_' + str(n_components))
    return lwpls


def get_ml_model(model_func, X_test, y_test, seed=0):
    return model_func(X_test, y_test, seed)


def ml_list(SEED, X_test, y_test):
    # ml_models = [(PLSRegression(nc, max_iter=5000), "PLS" + str(nc)) for nc in range(4, 12, 4)] # test
    # ml_models = [(PLSRegression(nc, max_iter=5000), "PLS" + str(nc)) for nc in range(4, 100, 4)]
    # ml_models.append((XGBRegressor(seed=SEED), 'XGBoost_100_None'))
    # ml_models.append((XGBRegressor(n_estimators=200, max_depth=50, seed=SEED), 'XGBoost_200_10'))
    # ml_models.append((XGBRegressor(n_estimators=50, max_depth=100, seed=SEED), 'XGBoost_50_100'))
    # ml_models.append((XGBRegressor(n_estimators=200, seed=SEED), 'XGBoost_200_'))
    # ml_models.append((XGBRegressor(n_estimators=400, max_depth=100, seed=SEED), 'XGBoost_400_100'))

    # ml_models.append((LWPLS(2, 2 ** -2, X_test, y_test), "LWPLS_2_0.25"))
    # ml_models.append((LWPLS(16, 2 ** -2, X_test, y_test), "LWPLS_16_0.25"))
    # ml_models.append((LWPLS(30, 2 ** -2, X_test, y_test), "LWPLS_30_0.25"))

    return ml_models
