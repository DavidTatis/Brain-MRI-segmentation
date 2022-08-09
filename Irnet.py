from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K



smooth=100

def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


def conv_irnet_model(input_shape=(256,256,3),n_layers=1,activation_function='relu',learning_rate=0.0001):   
    input_vec = Input(shape=input_shape)
    x01=Conv2D(32, (3, 3), padding='same')(input_vec)
    x02=BatchNormalization()(x01)
    x03=Activation(activation_function)(x02)
    pool1 = MaxPooling2D(pool_size=(2, 2))(x03)
    m1=pool1
    for i in range(n_layers):
        x01=Conv2D(32, (3, 3), padding='same')(m1)
        x02=BatchNormalization()(x01)
        x1=Activation(activation_function)(x02)
        m1=concatenate([m1, x1], axis=-1)

    for i in range(n_layers):
        x01=Conv2D(32, (3, 3), padding='same')(m1)
        x02=BatchNormalization()(x01)
        x1=Activation(activation_function)(x02)
        m1=concatenate([m1, x1], axis=-1)

    for i in range(n_layers):
        x01=Conv2D(32, (3, 3), padding='same')(m1)
        x02=BatchNormalization()(x01)
        x1=Activation(activation_function)(x02)
        m1=concatenate([m1, x1], axis=-1)

    for i in range(n_layers):
        x01=Conv2D(32, (3, 3), padding='same')(m1)
        x02=BatchNormalization()(x01)
        x1=Activation(activation_function)(x02)
        m1=concatenate([m1, x1], axis=-1)
        
    for i in range(n_layers):
        x01=Conv2D(32, (3, 3), padding='same')(m1)
        x02=BatchNormalization()(x01)
        x1=Activation(activation_function)(x02)
        m1=concatenate([m1, x1], axis=-1)

    for i in range(n_layers):
        x01=Conv2D(32, (3, 3), padding='same')(m1)
        x02=BatchNormalization()(x01)
        x1=Activation(activation_function)(x02)
        m1=concatenate([m1, x1], axis=-1)
    for i in range(n_layers):
        x01=Conv2D(32, (3, 3), padding='same')(m1)
        x02=BatchNormalization()(x01)
        x1=Activation(activation_function)(x02)
        m1=concatenate([m1, x1], axis=-1)

    convT=Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(m1) 
    xf = Conv2D(1, (1, 1), activation='sigmoid')(convT)

    model=Model(input_vec,xf)
    decay_rate = learning_rate / 200
    opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
    
    model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[iou, dice_coef])

    

    return model