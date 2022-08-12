# Brain-MRI-segmentation
## Dataset
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
Brain MRI images together with manual FLAIR abnormality segmentation masks.

## Data preprocessing utils 
https://www.kaggle.com/code/quang7doan/unet-doi-train-test-them-metric

## Models
Unet, Conv2d Irnet, Conv2d MnR.
Each model is created in a script file ready to be imported in the main program. Each models receive as input a (256,256,3) img the output is (256,256,1) which is the binary segmentation.

## Training
Each model is trained individually using Random Grid Search.
The hyperparameters adjusted are Batch size, Learning rate, Activation function and Num of layers. Each model is trained with 30 different hyperparam combination.
