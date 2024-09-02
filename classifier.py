#dense layer for classification
# IMPORT REQUIRED LIBRARIES:
# --------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
from PIL import Image
import imageio
import pandas as pd
import random
import pickle
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from skimage.io import imread_collection
import glob

#Reading .jpeg and .png images from the dataset 
CATS_folder= 'PET-IMAGES\1. Cat'
DOGS_folder= '\PET-IMAGES\2. Dog'

imdir= CATS_folder
ext= ['png','jpg']
files= []
[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
images= [cv2.imread(file) for file in files]

#Read in the training data(X) and corresponding labels(y).
#data=pickle.load(open("data.pickle","rb"))
X=pickle.load(open("X.pickle","rb"))
y=pickle.load(open("y.pickle","rb"))

#normalizing the data
X=np.array(X)
X=X/255.0
y=np.array(y)

#Building the model
model=Sequential()
model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(2,2)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

#adding sigmoid activation
model.add(Activation("sigmoid"))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

#fitting the model
model.fit(X,y,batch_size=32,epochs=50,shuffle=True, sample_weight=None, validation_split=0.1, verbose=1)

#Graphical Representation
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()





