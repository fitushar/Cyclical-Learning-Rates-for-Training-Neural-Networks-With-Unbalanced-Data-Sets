# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 21:49:23 2018

*py. File for Fixed Learning Rate test.
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
# Fixed Learning rate test
# =============================================================================

from __future__ import print_function

import numpy as np

from keras.models import model_from_json
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
#from sklearn.metrics import roc_auc_score

#from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import scikitplot as skplt
import h5py

#Define the save model and weights.
# =============================================================================
##Initializing weights and models name to save
model_name = "kerasModel_fixed_002_epoch12_sk17.json"
weights_name = "modelWeights_fixed_002_epoch12_sk17.h5"
normalizer_name = "normalizer_fixed_002_epoch12_sk17.pkl"
save = True; #True and False to save the model ad weights
# =============================================================================
#################################################################################################################
# POSITIVE - Test
fpos = h5py.File('Positive_test.h5', 'r')
first_key = list(fpos.keys())[0]
dpos=fpos[first_key]
size_image=dpos.shape[1]
num_pos_test = dpos.shape[0]
dposar = np.zeros(dpos.shape)
dpos.read_direct(dposar)
test_pos=dposar.reshape([-1,size_image,size_image,1])

print("The positive patches for testing are: ", num_pos_test)

# NEGATIVE - Test
fneg = h5py.File('300kNegative_test.h5', 'r')
first_key=list(fneg.keys())[0]
dneg=fneg[first_key]
size_image=dneg.shape[1]
num_neg_test = dneg.shape[0]
dnegar = np.zeros(dneg.shape)
dneg.read_direct(dnegar)
test_neg=dnegar.reshape([-1,size_image,size_image,1])

print("The negative patches for testing are: ", num_neg_test)

# the sum of positive and negative test set ~= 20% od training set size
actual_num_pos_test = 1164
actual_num_neg_test = 110000

# Create the target vectors
test_pos_lab = np.ones((actual_num_pos_test,1))
test_neg_lab = np.zeros((actual_num_neg_test,1))

#### Build the training set  (images and targets)
tpos = np.copy(test_pos[0:actual_num_pos_test,:,:,:])
tneg = np.copy(test_neg[0:actual_num_neg_test,:,:,:])

# Stack the subsets
X_Test = np.vstack((tpos,tneg))
Y_Test = np.vstack((test_pos_lab,test_neg_lab))

json_file = open(model_name, 'r')
loaded_json_model = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_json_model)
loaded_model.load_weights(weights_name)

print("model loaded")
gradientDescent = SGD(lr= lr_fixed, decay=0.96)
loaded_model.compile(gradientDescent, loss= 'categorical_crossentropy', metrics = ['accuracy'])

####### Loading NORMALIZER #######
########                 ################################
normalizer  = joblib.load(normalizer_name)
normalized_Xtest = normalizer.standardize(X_Test)

predictions = loaded_model.predict_proba(normalized_Xtest)
Y_one_hot = to_categorical(np.ravel(Y_Test),2)
score = loaded_model.evaluate(x=X_Test,y= Y_one_hot)
print("The Scored loss is {}, accuracy{}".format(score[0], score[1]))

# print('accuracy ={}'.format(np.mean(np.argmax(predictions,1)==Y_Test)))
# print('accuracy ={}'.format(accuracy_score(Y_Test, np.argmax(predictions,1))))

fpr_keras, tpr_keras, _ = roc_curve(Y_Test, predictions[:,1])

test_auc = roc_auc_score(Y_Test, predictions[:,1])

test_precision, test_recall, test_f1score, _ = precision_recall_fscore_support(Y_Test, np.argmax(predictions, axis=1),average = 'binary')

print("AUC: %.4f PRECISION:  %.4f RECALL: %.4f F1SCORE: %.4f " %(test_auc,test_precision, test_recall, test_f1score))

skplt.metrics.plot_roc(Y_Test, predictions ,plot_micro=False, plot_macro=False, classes_to_plot=1)
skplt.metrics.plot_precision_recall(Y_Test, predictions ,plot_micro=False, classes_to_plot=1, cmap='plasma')