# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 21:43:47 2018

*py. File for Fixed Learning Rate.
*Need to install Necessary libraries

#!pip install -q keras
#!pip3 install torch torchvision
#!apt-get -qq install -y libsm6 libxext6 && pip install -q -U opencv-python
#import cv2
#import tensorflow as tf
#!pip install -q scikit-plot

**Need to set the name of the models you want to save.

@author: Fakrul-IslamTUSHAR
"""
# =============================================================================
# Fixed learing Training
# =============================================================================
# Fixed learling Rate

from __future__ import print_function

###Initialixzing Training Values
import datetime
train_size  = 64000 ##skew=1/36
actual_num_pos_train = 3459
actual_num_neg_train = train_size - actual_num_pos_train

#initially, the rising edge covers the complete experiment
lr_fixed=0.002     
epochs = 12
batch_size = 32 
print('confirmed at {}'.format(datetime.datetime.now().time()))

##Initializing weights and models name to save
model_name = "kerasModel_fixed_002_epoch12_sk17.json"
weights_name = "modelWeights_fixed_002_epoch12_sk17.h5"
normalizer_name = "normalizer_fixed_002_epoch12_sk17.pkl"
save = True; #True and False to save the model ad weights

import numpy as np
import matplotlib.pyplot as plt
#from CLR import CyclicLR
### keras tools
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


## sklearn tools
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support
import scikitplot as skplt

import scikitplot as skplt

import h5py
#################################################################################################################

# POSITIVE - TRAIN
fpos = h5py.File('Positive_train.h5', 'r')
first_key=list(fpos.keys())[0]
dpos=fpos[first_key]
size_image=dpos.shape[1]
num_pos_train = dpos.shape[0]
dposar = np.zeros(dpos.shape)
dpos.read_direct(dposar)
train_pos=dposar.reshape([-1,size_image,size_image,1])

print("The positive patches for training are: ", num_pos_train)

# NEGATIVE - TRAIN
fneg = h5py.File('300kNegative_train.h5', 'r')
first_key=list(fneg.keys())[0]
dneg=fneg[first_key]
size_image=dneg.shape[1]
num_neg_train = dneg.shape[0]
dnegar = np.zeros(dneg.shape)
dneg.read_direct(dnegar)
train_neg=dnegar.reshape([-1,size_image,size_image,1])

print("The egative patches for training are: ", num_neg_train)

   

# Create the target vectors
train_pos_lab = np.ones((actual_num_pos_train,1))
train_neg_lab = np.zeros((actual_num_neg_train,1))

#### Build the training set  (images and targets)
tpos = np.copy(train_pos[0:actual_num_pos_train,:,:,:])
tneg = np.copy(train_neg[0:actual_num_neg_train,:,:,:])

# Stack the subsets
X_Train = np.vstack((tpos,tneg))
Y_Train = np.vstack((train_pos_lab,train_neg_lab))

# Shuffle the two arrays in unison
X_Train, Y_Train = shuffle(X_Train,Y_Train)

#######################augmentation + 
#######################Normalization

### do data augmentation at the end before predicting

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=0,
    vertical_flip=False)
datagen.fit(X_Train)

#################################
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size=(3, 3), activation='relu',padding='same', name='block1_conv1', input_shape = X_Train.shape[1:]))
model.add(Conv2D(filters =32, kernel_size=(3, 3), activation='relu',padding='same', name='block1_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

model.add(Conv2D(filters =32, kernel_size=(3, 3), activation='relu',padding='same', name='block2_conv1'))
model.add(Conv2D(filters =32, kernel_size=(3, 3), activation='relu',padding='same', name='block2_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

model.add(Flatten())

model.add(Dense(units = 256, activation='relu', name='fc1'))
model.add(Dropout(rate=0.5))

model.add(Dense(units = 256, activation='relu', name='fc2'))
model.add(Dropout(rate=0.5))

model.add(Dense(units=2, activation='softmax', name='predictions'))


Y_one_hot = to_categorical(np.ravel(Y_Train),2)

#### fixed lr part
gradientDescent = SGD(lr=lr_fixed, decay=0.96)
model.compile(gradientDescent, loss= 'categorical_crossentropy', metrics = ['accuracy'])
model.fit_generator(datagen.flow(x= X_Train, y =Y_one_hot,batch_size = batch_size),steps_per_epoch=len(X_Train) / batch_size,
                     verbose=2,epochs=epochs, use_multiprocessing=True)
                    


#####################################################
#################NORMALIATION########################
normalized_Xtrain = datagen.standardize(X_Train)
####################################################
####################################################
y_pred_keras = model.predict_proba(normalized_Xtrain, verbose=2, batch_size=batch_size)
fpr_keras, tpr_keras, _ = roc_curve(Y_Train, y_pred_keras[:,1])
train_auc = roc_auc_score(Y_Train, y_pred_keras[:,1])
print("AUC: {}".format(train_auc))

skplt.metrics.plot_roc(Y_Train, y_pred_keras, plot_micro=False, plot_macro=False)
plt.figure()
plt.plot(fpr_keras, tpr_keras)
plt.title('roc curve plotted manualy')
plt.show

test = y_pred_keras[y_pred_keras[:,1]!=0, 1]
test.__len__()

############################### #serialize model to JSON and weights to HDF5
if save == True:
    model_json = model.to_json()
    with open(model_name, 'w') as json_file:
        json_file.write(model_json)

    model.save_weights(weights_name)
    joblib.dump(datagen,normalizer_name)

    print("model and weights have been saved")

###############################

