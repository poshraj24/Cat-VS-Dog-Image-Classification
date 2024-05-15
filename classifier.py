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
