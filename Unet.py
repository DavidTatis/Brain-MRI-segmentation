from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K



def dice_coef(y_true, y_pred):
    smooth=100
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    smooth=100
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def unet_model(input_shape=(256,256,3),n_layers=1,activation_function='relu',learning_rate=0.0001):
    inputs = Input(input_shape)
    n_layers-=1
    conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
    bn1 = Activation(activation_function)(conv1)
    for i in range(n_layers):
        conv1 = Conv2D(64, (3, 3), padding='same')(bn1)
        bn1 = BatchNormalization(axis=3)(conv1)
        bn1 = Activation(activation_function)(bn1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    bn2 = Activation(activation_function)(conv2)
    for i in range(n_layers):
        conv2 = Conv2D(128, (3, 3), padding='same')(bn2)
        bn2 = BatchNormalization(axis=3)(conv2)
        bn2 = Activation(activation_function)(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
    bn3 = Activation(activation_function)(conv3)
    for i in range(n_layers):
        conv3 = Conv2D(256, (3, 3), padding='same')(bn3)
        bn3 = BatchNormalization(axis=3)(conv3)
        bn3 = Activation(activation_function)(bn3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
    bn4 = Activation(activation_function)(conv4)
    for i in range(n_layers):
        conv4 = Conv2D(512, (3, 3), padding='same')(bn4)
        bn4 = BatchNormalization(axis=3)(conv4)
        bn4 = Activation(activation_function)(bn4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(1024, (3, 3), padding='same')(pool4)
    bn5 = Activation(activation_function)(conv5)
    for i in range(n_layers):
        conv5 = Conv2D(1024, (3, 3), padding='same')(bn5)
        bn5 = BatchNormalization(axis=3)(conv5)
        bn5 = Activation(activation_function)(bn5)

    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bn5), conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), padding='same')(up6)
    bn6 = Activation(activation_function)(conv6)
    for i in range(n_layers):
        conv6 = Conv2D(512, (3, 3), padding='same')(bn6)
        bn6 = BatchNormalization(axis=3)(conv6)
        bn6 = Activation(activation_function)(bn6)

    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bn6), conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), padding='same')(up7)
    bn7 = Activation(activation_function)(conv7)
    for i in range(n_layers):
        conv7 = Conv2D(256, (3, 3), padding='same')(bn7)
        bn7 = BatchNormalization(axis=3)(conv7)
        bn7 = Activation(activation_function)(bn7)

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(bn7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), padding='same')(up8)
    bn8 = Activation(activation_function)(conv8)
    for i in range(n_layers):
        conv8 = Conv2D(128, (3, 3), padding='same')(bn8)
        bn8 = BatchNormalization(axis=3)(conv8)
        bn8 = Activation(activation_function)(bn8)

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(bn8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), padding='same')(up9)
    bn9 = Activation(activation_function)(conv9)
    for i in range(n_layers):
        conv9 = Conv2D(64, (3, 3), padding='same')(bn9)
        bn9 = BatchNormalization(axis=3)(conv9)
        bn9 = Activation(activation_function)(bn9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(bn9)
    model=Model(inputs,conv10)
    opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
    model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[iou, dice_coef])

    return model

