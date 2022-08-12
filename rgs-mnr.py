# %%
# Reference https://www.kaggle.com/code/quang7doan/unet-doi-train-test-them-metric
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
# %matplotlib inline
import csv
# import visualkeras
# import cv2
from tqdm import tqdm_notebook, tnrange
from glob import glob
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from skimage.color import rgb2gray
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping, ModelCheckpoint

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.backend import clear_session
from tensorflow.keras import models, layers, regularizers
import logging
from tensorflow.keras.utils import plot_model
logging.getLogger("tensorflow").setLevel(logging.ERROR) #to hide the autigraph WARNING at model.fit
from random import random,randrange
from operator import itemgetter
import timeit
from sklearn.model_selection import train_test_split, ParameterGrid, ParameterSampler
# from Unet import unet_model
# from Irnet import conv_irnet_model
from MnR import mnr_model

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


# %% [markdown]
# ## Data is ready for training/testing, Random Grid Search:

# %% [markdown]
# #### DATA/TASK INFORMATION:

# %%
#DATA/TASK INFORMATION:
architecture_name="mnr"
problem_type="segmentation"
num_features=df_train.shape
input_shape =(im_height,im_width,3)
model_file_name=architecture_name

# %% [markdown]
# #### RGS evaluate_fitness

# %%
def evaluate_fitness(input_shape,n_layers,activation_function,learning_rate,batch_size,hp_dataset_name,weights_name,max_epochs,patience_epochs):
    clear_session()
    EPOCHS = max_epochs
    len(df_val)/batch_size
    #callbacks
    earlystopping = EarlyStopping(monitor='val_loss',
                              mode='min', 
                                verbose=1, 
                                patience=patience_epochs
                                )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                mode='min',
                                verbose=1,
                                patience=10,
                                min_delta=0.0001,
                                factor=0.2
                                )
    #augmentate training data
    train_generator_args = dict(rotation_range=0.2,
                                width_shift_range=0.05,
                                height_shift_range=0.05,
                                shear_range=0.05,
                                zoom_range=0.05,
                                horizontal_flip=True,
                                fill_mode='nearest')

    train_gen = train_generator(df_train, batch_size,
                                    train_generator_args,
                                    target_size=(im_height, im_width))
    #augmentate valid data
    val_gener = train_generator(df_val, batch_size,
                                    dict(),
                                    target_size=(im_height, im_width))
    #create the model
    model=mnr_model(input_shape,n_layers,activation_function,learning_rate)
    plot_model(model,to_file='output.png')
    callbacks = [ModelCheckpoint('data/checkpoints/'+model_file_name+'.hdf5', verbose=1, save_best_only=True), earlystopping,reduce_lr]
    start_time = timeit.default_timer()
    history = model.fit(train_gen,
                    steps_per_epoch=len(df_train) / batch_size, 
                    epochs=EPOCHS, 
                    callbacks=callbacks,
                    validation_data = val_gener,
                    validation_steps=len(df_val) / batch_size)
    
    end_time = timeit.default_timer()
    training_and_validation_samples=len(df_train)+len(df_val)
    print("==== len train valid data",len(df_train),len(df_val))
    #EVALUATE MODEL
    test_gen = train_generator(df_test, batch_size,
                                    dict(),
                                    target_size=(im_height, im_width))
    results = model.evaluate(test_gen, steps=len(df_test) / batch_size)
    print("Test IOU: ",results[2])
    iou_test=results[2]

    #SAVE THE WEIGHTS
    model.save("data/weights/"+architecture_name+"/"+weights_name+".h5")
    #SAVE THE HYPERPARAMS AND THE METRIC
    with open('data/'+hp_dataset_name, mode='a+') as hp_dataset:
        hp_dataset_writer=csv.writer(hp_dataset,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        hp_dataset_writer.writerow([architecture_name,
                                problem_type,
                                num_features,
                                training_and_validation_samples,
                                n_layers,
                                input_shape,
                                activation_function,
                                learning_rate,
                                batch_size,
                                str(len(history.history['loss'])),
                                end_time-start_time,
                                iou_test
                                ])
    return iou_test

# %% [markdown]
# #### RGS main

# %%
def random_gridsearch(population_size,input_shape,n_layers,activation_function,learning_rate,batch_size,hp_dataset_name,max_epochs,patience_epochs):
        dict_all_hyperparams=dict(n_layers=n_layers,
                                learning_rate=learning_rate,
                                activation_function=activation_function,
                                batch_size=batch_size,
                                )
        r_grid_search_population=list(ParameterSampler(dict_all_hyperparams,population_size))
        
        RGS_evaluated_hparams=[]
        with open("data/logs/"+architecture_name+"_RandomGridSearch.csv", mode='a+') as logs_dataset:
                logs_dataset_writer=csv.writer(logs_dataset,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                logs_dataset_writer.writerow(["population: "+str(population_size)])
                logs_dataset_writer.writerows(dict(x=r_grid_search_population).values())
        print(r_grid_search_population)

        
        for i in range(len(r_grid_search_population)):
                weights_name='{}-{}-{}-{}'.format(r_grid_search_population[i]['n_layers'],r_grid_search_population[i]['activation_function'],r_grid_search_population[i]['learning_rate'],r_grid_search_population[i]['batch_size'])
                model_file_name=architecture_name+str(i)
                metric=evaluate_fitness(input_shape,
                                r_grid_search_population[i]['n_layers'],
                                r_grid_search_population[i]['activation_function'],
                                r_grid_search_population[i]['learning_rate'],
                                r_grid_search_population[i]['batch_size'],
                                hp_dataset_name,
                                weights_name,
                                max_epochs,
                                patience_epochs
                                )
                
                with open("data/logs/"+architecture_name+"_RandomGridSearch.csv", mode='a+') as logs_dataset:
                        logs_dataset_writer=csv.writer(logs_dataset,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        logs_dataset_writer.writerow(["i:"+str(i)+"Metric:"+str(metric)])
                print("i",i,"Mae:",metric)

                RGS_evaluated_hparams.insert(len(RGS_evaluated_hparams),{"hparam":i,"metric":metric})
        rgs_top_hparam=sorted(RGS_evaluated_hparams,key=itemgetter('metric'),reverse=True)[0]['hparam']
        
        with open("data/logs/"+architecture_name+"_RandomGridSearch.csv", mode='a+') as logs_dataset:
                        logs_dataset_writer=csv.writer(logs_dataset,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        logs_dataset_writer.writerow("END")
                        logs_dataset_writer.writerows(sorted(RGS_evaluated_hparams,key=itemgetter('metric'),reverse=True)[0]['metric'],r_grid_search_population[rgs_top_hparam])
        
        return sorted(RGS_evaluated_hparams,key=itemgetter('metric'),reverse=True)[0]['metric'],r_grid_search_population[rgs_top_hparam]

# %% [markdown]
# #### RGS Definitions and invocation

# %%
#HYPERPARAMETERS DEFINITION
n_layers = [1,2,3]
activation_function=['relu','tanh','sigmoid','elu']
learning_rate=[0.01,0.001,0.0001,0.00001]
batch_size=[8,16,32,64]
max_epochs=200
patience_epochs=20

#FILES NAME
hp_dataset_name="mnr_hyperparams_with_metric.csv"

#ALGORITHM PARAMS
population_size=30

random_gridsearch(population_size,input_shape,n_layers,activation_function,learning_rate,batch_size,hp_dataset_name,max_epochs,patience_epochs)


# %%
# TEST MODEL

# model = load_model(model_name+'.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})
# # model.summary()
# test_gen = train_generator(df_test, 32,
#                                 dict(),
#                                 target_size=(im_height, im_width))
# results = model.evaluate(test_gen, steps=len(df_test) / 32)
# print("Test IOU: ",results[2])
# print("Test lost: ",results[0])
# print("Test Dice Coefficent: ",results[3])



