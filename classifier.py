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
CATS_folder= 'D:\MSc_DS\Sem_3\Seminar\python-intro\input\PET-IMAGES\1. Cat'
DOGS_folder= 'D:\MSc_DS\Sem_3\Seminar\python-intro\input\PET-IMAGES\2. Dog'

imdir= CATS_folder
ext= ['png','jpg']
files= []
[files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
images= [cv2.imread(file) for file in files]

#Read in the training data(X) and corresponding labels(y).
#data=pickle.load(open("data.pickle","rb"))
X=pickle.load(open("X.pickle","rb"))
y=pickle.load(open("y.pickle","rb"))

