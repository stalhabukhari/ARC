"""
cnn_model.py
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Dense, Dropout, Activation, Flatten, Conv2D,
                          MaxPooling2D, BatchNormalization, ReLU, LeakyReLU, PReLU, Softmax)
from tensorflow.keras.initializers import he_normal, Constant
from tensorflow.keras.regularizers import l2


def cnn_model(configuration={}):
    bn_flag = configuration.get('BatchNorm')
    acti_func = configuration.get('Activation')
    reg_scale = configuration.get('Regularization')
    dropout_rate = configuration.get('Dropout')

    k_init = he_normal()
    k_reg = l2(reg_scale)
    b_init = Constant(0.1)
    b_reg = l2(reg_scale)

    model = Sequential()

    # Conv 1
    model.add(Conv2D(filters=8, kernel_size=(3, 3), padding='valid', kernel_initializer=k_init,
                     kernel_regularizer=k_reg, bias_initializer=b_init, bias_regularizer=b_reg,
                     input_shape=(150, 150, 3)))
    if bn_flag:
        model.add(BatchNormalization())
    model.add(get_acti(name=acti_func))

    # Downsample 1
    model.add(MaxPooling2D(pool_size=(4, 4)))

    # Conv 2
    model.add(Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_initializer=k_init,
                     kernel_regularizer=k_reg, bias_initializer=b_init, bias_regularizer=b_reg))
    if bn_flag:
        model.add(BatchNormalization())
    model.add(get_acti(name=acti_func))

    # Downsample 2
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # FC 1
    model.add(Dense(512, kernel_initializer=k_init, kernel_regularizer=k_reg, bias_initializer=b_init,
                    bias_regularizer=b_reg))
    model.add(get_acti(name=acti_func))
    if dropout_rate:
        model.add(Dropout(dropout_rate))

    # FC 2
    model.add(Dense(256, kernel_initializer=k_init, kernel_regularizer=k_reg, bias_initializer=b_init,
                    bias_regularizer=b_reg))
    model.add(get_acti(name=acti_func))
    if dropout_rate:
        model.add(Dropout(dropout_rate))

    # FC 3
    model.add(Dense(100, kernel_initializer=k_init, kernel_regularizer=k_reg, bias_initializer=b_init,
                    bias_regularizer=b_reg))
    model.add(Softmax())
    return model


def get_acti(name=None):
    if name == 'relu':
        return ReLU()
    elif name == 'leakyrelu':
        return LeakyReLU()
    elif name == 'prelu':
        return PReLU()
    else:
        print('[Warning] No Activation selected!')
        return None


if __name__ == '__main__':
    net_configuration = {
        'BatchNorm': True,
        'Activation': 'prelu',
        'Regularization': 0.01,
        'Dropout': 0.1,
    }
    net = cnn_model(net_configuration)
    net.summary()
