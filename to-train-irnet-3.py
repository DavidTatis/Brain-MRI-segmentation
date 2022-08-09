# %%
# Reference https://www.kaggle.com/code/quang7doan/unet-doi-train-test-them-metric
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
# %matplotlib inline

# import cv2
from tqdm import tqdm_notebook, tnrange
from glob import glob
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import *
import tensorflow as tf
from skimage.color import rgb2gray
from tensorflow.keras.models import Model, load_model, save_model

os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras import models, layers, regularizers
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR) #to hide the autigraph WARNING at model.fit


from Unet import unet_model
from Irnet import conv_irnet_model

# %%
#Set Parameters
im_width = 256
im_height = 256

# %%
train_files = []
mask_files = glob('./data/input/lgg-mri-segmentation/kaggle_3m/*/*_mask*')

for i in mask_files:
    train_files.append(i.replace('_mask',''))

# print(train_files[:10])
# print(mask_files[:10])

# %%
rows,cols=3,3
fig=plt.figure(figsize=(10,10))
for i in range(1,rows*cols+1):
    fig.add_subplot(rows,cols,i)
    img_path=train_files[i]
    msk_path=mask_files[i]
    img=plt.imread(img_path)
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    msk=plt.imread(msk_path)
    plt.imshow(img)
    plt.imshow(msk,alpha=0.7)
plt.show()

# %%
df = pd.DataFrame(data={"filename": train_files, 'mask' : mask_files})
df_train, df_test = train_test_split(df,test_size = 0.15)
df_train, df_val = train_test_split(df_train,test_size = 0.1765)
print(df_train.values.shape)
print(df_val.values.shape)
print(df_test.values.shape)

# %%
def train_generator(data_frame, batch_size, aug_dict,
        image_color_mode="rgb",
        mask_color_mode="grayscale",
        image_save_prefix="image",
        mask_save_prefix="mask",
        save_to_dir=None,
        target_size=(256,256),
        seed=1):
    '''
    can generate image and mask at the same time use the same seed for
    image_datagen and mask_datagen to ensure the transformation for image
    and mask is the same if you want to visualize the results of generator,
    set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        x_col = "filename",
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col = "mask",
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_gen = zip(image_generator, mask_generator)
    
    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img,mask)

def adjust_data(img,mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    return (img, mask)

# %%
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


# %%


# %%
model_irnet=conv_irnet_model((256,256,3),3,'relu',1e-4)
# model_unet=unet_model((256,256,3),3,'relu',0.0)
model_name="irnet-3-0.00001"

# %%
EPOCHS = 200
BATCH_SIZE = 32
learning_rate = 1e-4
len(df_val)/BATCH_SIZE

# %%
train_generator_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest')
train_gen = train_generator(df_train, BATCH_SIZE,
                                train_generator_args,
                                target_size=(im_height, im_width))
    
test_gener = train_generator(df_val, BATCH_SIZE,
                                dict(),
                                target_size=(im_height, im_width))
    
# model = Attention_UNet(input_shape=(im_height, im_width, 3))
#callbacks
earlystopping = EarlyStopping(monitor='val_loss',
                              mode='min', 
                              verbose=1, 
                              patience=20
                             )
# save the best model with lower validation loss

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              mode='min',
                              verbose=1,
                              patience=10,
                              min_delta=0.0001,
                              factor=0.2
                             )
                             
callbacks = [ModelCheckpoint(model_name+'.hdf5', verbose=1, save_best_only=True), earlystopping,reduce_lr]

history = model_irnet.fit(train_gen,
                    steps_per_epoch=len(df_train) / BATCH_SIZE, 
                    epochs=EPOCHS, 
                    callbacks=callbacks,
                    validation_data = test_gener,
                    validation_steps=len(df_val) / BATCH_SIZE)

# %%
# TEST MODEL

ALPHA = 0.5
BETA = 0.5

def Tversky(y_true, y_pred, alpha=ALPHA, beta=BETA, smooth=100):
        
        #flatten label and prediction tensors
        y_pred = K.flatten(y_pred)
        y_true = K.flatten(y_true)
        
        #True Positives, False Positives & False Negatives
        TP = K.sum((y_pred * y_true))
        FP = K.sum(((1-y_true) * y_pred))
        FN = K.sum((y_true * (1-y_pred)))
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return Tversky
model = load_model(model_name+'.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef,'Tversky':Tversky})
model.summary()
test_gen = train_generator(df_test, 32,
                                dict(),
                                target_size=(im_height, im_width))
results = model.evaluate(test_gen, steps=len(df_test) / 32)
print("Test IOU: ",results[2])
# print("Test lost: ",results[0])
# print("Test Dice Coefficent: ",results[3])




# %%
