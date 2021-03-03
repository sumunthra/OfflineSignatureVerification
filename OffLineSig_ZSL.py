# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 21:27:20 2018

@author: sumunthra
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 17:33:57 2018

@author: sumunthra
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 17:46:01 2018

@author: sumunthra
"""

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import *
from keras.models import *
from keras.layers import Input, Dense
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
#from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
from keras.utils.np_utils import to_categorical
import glob
import os.path
from keras.models import load_model
from sklearn.datasets import load_files
from keras.utils import np_utils
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Activation, Flatten
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from tqdm import tqdm
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16 #Results in vgg weights download
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inception_v3
from keras.layers.merge import Concatenate
from keras.models import Model
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
import time
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D,MaxPool2D, SeparableConv2D,AveragePooling2D,Dense, Dropout,Activation, Flatten
from keras.applications.mobilenet import MobileNet
from keras.optimizers import SGD
# from imagenet_utils import preprocess_input
# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import datetime
import time
# baseline model for the dogs vs cats dataset
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import re 
# generate random integer values
from random import seed
from random import randint
import random

#from keras.applications.mobilenet import InceptionResNetV2
#######################################################################################################################################################################
No_of_Users = 55
Total_genuine_samples_per_user = 24
Total_forgery_samples_per_user = 24
Signature_Samples_Per_User = 24
##########################################################################################################################################################################
PerOfSeen = 1.0
PerOfUnSeen = 100-PerOfSeen
No_Of_SeenUsers = 1
No_Of_UnSeenUsers = No_of_Users-No_Of_SeenUsers
print('No_Of_SeenUsers =',No_Of_SeenUsers)
print('No_Of_UnSeenUsers =',No_Of_UnSeenUsers)
#########################################################################################################################################################################################
No_training_samples_per_User = 1 # 1, 2, 3,4,5,10,15,20 : Only changing point in the entire code 
#############################################################################################################################################################################
No_testing_samples_per_User = Total_genuine_samples_per_user - No_training_samples_per_User
##########################################################################################################################################
Y_train = []  
Y_test = []  
#####################################################################################################################################
epoch = 40
img_width, img_height = 256,256
##################################################################################################################################################
trainDataset = 'C:\\Users\\pc\\Desktop\\OfflineSignature\\OfflineSignatureVerification-master\\Datasets\\cedar1\\full_org'
#validDataset = 'Devanagari/valid'
testDataset = 'C:\\Users\\pc\\Desktop\\OfflineSignature\\OfflineSignatureVerification-master\\Datasets\\cedar1\\full_forg'
###########################################################################################################################
train_images = glob.glob(trainDataset+"/*.png")
length_images_train = len(train_images)
print(length_images_train)

test_images = glob.glob(testDataset+"/*.png")
length_images_test = len(test_images)
print(length_images_test)
##################################################################################################################################
def Get_Full_Data(writerId):
    #print(writerId)
    writer_genuine_image_list = []
    writer_forgery_image_list = []
    myRandomNumbers = []
    Writer_Specific_FileList = []
    for i in range(writerId, writerId+1):
        for j in range(1, 25):
            filename= str(i)+'_'+str(j)
            filename = trainDataset+'\original_'+filename+'.png'
            #print(filename)
            writer_genuine_image_list.append(filename)
    for i in range(writerId, writerId+1):
        for j in range(1, 25):
            filename= str(i)+'_'+str(j)
            filename = testDataset+'\\forgeries_'+filename+'.png'
            #print(filename)
            writer_forgery_image_list.append(filename)
    
    return  writer_genuine_image_list, writer_forgery_image_list
#################################################################################################################################################
Final_train_List1=[];
Final_test_List1=[];
Final_train_List2=[];
Final_test_List2=[];
Final_train_List3=[];
Final_test_List3=[];
Random_Selected_List1 = [];
Random_Selected_List2 = [];
#################################################################################################################################################
for i in range(1, No_Of_SeenUsers+1):
   Current_writer_genuine_image_list=[]
   Current_writer_forgery_image_list=[]
   Current_writer_genuine_image_list, Current_writer_forgery_image_list = Get_Full_Data(i)
   Random_Selected_List1 = [];
   Random_Selected_List2 = [];
   #print(len(Current_writer_genuine_image_list))
   #print(len(Current_writer_forgery_image_list))
   Random_Selected_List1 = random.sample(range(0, 23), No_training_samples_per_User) # rand int considers last value also. List starts from 0. hence, 0 to 23
   Random_Selected_List2 = random.sample(range(0, 23), No_training_samples_per_User) # rand int considers last value also. List starts from 0. hence, 0 to 23
   
   for j in range(0,No_training_samples_per_User):
       
       Final_train_List1.append(Current_writer_genuine_image_list[Random_Selected_List1[j]])
       Final_train_List2.append(Current_writer_forgery_image_list[Random_Selected_List2[j]])
#####################################################################################################################################################################################
for i in range(1, No_Of_UnSeenUsers+1):
   Current_writer_genuine_image_list=[]
   Current_writer_forgery_image_list=[]
   Current_writer_genuine_image_list, Current_writer_forgery_image_list = Get_Full_Data(i)

   for j in range(0,Signature_Samples_Per_User):
       
       Final_test_List1.append(Current_writer_genuine_image_list[j])
       Final_test_List2.append(Current_writer_forgery_image_list[j])
#######################################################################################################################################################################################
Final_train_List = Final_train_List1+ Final_train_List2 
Final_test_List = Final_test_List1+ Final_test_List2 

print(len(Final_train_List1))
print(len(Final_train_List2))
print(len(Final_train_List))

print(len(Final_test_List1)) 
print(len(Final_test_List2)) 
print(len(Final_test_List))

print(len(Random_Selected_List1))
print(len(Random_Selected_List2))
###################################################################################################################################################     
No_Genuine_train_Labels = No_Of_SeenUsers * No_training_samples_per_User
No_Forgery_train_Labels = No_Genuine_train_Labels


No_Genuine_test_Labels =  No_Of_UnSeenUsers * Signature_Samples_Per_User
No_Forgery_test_Labels = No_Genuine_test_Labels

print(No_Genuine_train_Labels)
print(No_Forgery_train_Labels)
print(No_Genuine_test_Labels)
print(No_Forgery_test_Labels)
#################################################################################################################################################
for photo in Final_train_List:
        #print(photo)
        img = image.load_img(photo, target_size=(img_width, img_height))
        tr_x = image.img_to_array(img)
        tr_x = preprocess_input(tr_x)
        Final_train_List3.append(tr_x)

for photo in Final_test_List:
        #print(photo)
        img = image.load_img(photo, target_size=(img_width, img_height))
        tr_x = image.img_to_array(img)
        tr_x = preprocess_input(tr_x)
        Final_test_List3.append(tr_x)
############################################################################################################
###############################################################################################################################
for i in range(0,No_Genuine_train_Labels):  
    Y_train.append(0)
for i in range(No_Forgery_train_Labels):
    Y_train.append(1)
Y_train = np.array(Y_train)

for i in range(0,No_Genuine_test_Labels):  
    Y_test.append(0)
for i in range(No_Forgery_test_Labels):
    Y_test.append(1)
Y_test = np.array(Y_test)
##################################################################################################################################
X_train = np.array(Final_train_List3)
X_test = np.array(Final_test_List3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# X_train /= 255
# X_test /= 255

print(X_train.shape)    # 7000,32,32,3 (Image size of 32*32 of 3 channels)
print(Y_train.shape)    # 7000, 10  (7000 images of 10 categories 0-9)
print(X_test.shape)    # 7000,32,32,3 (Image size of 32*32 of 3 channels)
print(Y_test.shape)    # 7000, 10  (7000 images of 10 categories 0-9)
#################################################################################################################
n_classes = 2
print("Shape before one-hot encoding: ", Y_train.shape)
Y_train = np_utils.to_categorical(Y_train, n_classes)
Y_test = np_utils.to_categorical(Y_test, n_classes)
print("Shape after one-hot encoding: ", Y_test.shape)
############################################################################################################################
model = Sequential()
model.add(SeparableConv2D(32,(3, 3),padding="same", activation="relu",kernel_initializer='he_uniform',depthwise_initializer='he_uniform', input_shape=(256,256,3)))
model.add(AveragePooling2D())
model.add(Dropout(0.2))

model.add(SeparableConv2D(64, (3, 3), padding="same",kernel_initializer='he_uniform',depthwise_initializer='he_uniform', activation="relu"))
model.add(AveragePooling2D())
model.add(Dropout(0.2))

# model.add(SeparableConv2D(128, (3, 3), padding="same",kernel_initializer='he_uniform',depthwise_initializer='he_uniform', activation="relu"))
# model.add(AveragePooling2D())
# model.add(Dropout(0.2))


model.add(GlobalAveragePooling2D())
model.add(Dense(128,activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(2,kernel_initializer='he_uniform',activation='sigmoid'))
adam = optimizers.Adam(lr=0.004,beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00)

model.summary()

model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])

t = time.time()
hist = model.fit(X_train,Y_train,batch_size=16,epochs=epoch, verbose=1,validation_data=(X_test, Y_test),)
print ('Training time: %s' % (t - time.time()))
(loss, accuracy) = model.evaluate(X_test, Y_test, batch_size=10, verbose=1)

print ('[INFO] loss={:.4f}, accuracy: {:.4f}%', loss, accuracy* 100)
print ('Accuracy = ',max(hist.history['val_accuracy'])*100)

###########################################################################################################################import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
xc=range(epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_accuracy vs val_accuracy')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

#########################################################################################################################
# Get_Full_Data(trainDataset)
# print(image_names)
# image_names1  = image_names.sort()
# print(image_names1)
      
# def load_data(path):
#     x_image = []
#     images = glob.glob(path+"/*.png")
#     for photo in images:
#         print(photo)
#         img = image.load_img(photo, target_size=(img_width, img_height))
#         tr_x = image.img_to_array(img)
#         tr_x = preprocess_input(tr_x)
#         x_image.append(tr_x)
#     return np.array(x_image)
#############################################################################################################################