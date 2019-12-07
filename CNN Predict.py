# PREDICTION

%tensorflow_version 1.x

import os 
os.chdir ('/content/drive/My Drive/Project_CS4347/')
working_dir = os.getcwd()

from PIL import Image
import tensorflow as tf
import tensorflow.keras
import glob
import matplotlib
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras.layers.core import Dense, Flatten
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers.convolutional import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import sklearn.metrics as metrics 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
np.random.seed(126) # numpy random seed
import random as rd # for python
rd.seed(1) # python random seed
tf.compat.v2.random.set_seed(2) # tensorflow random seed
import cv2
import pickle
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = load_model(os.path.join(working_dir, 'models', 'model.h5'))
model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
test_path = os.path.join(working_dir, 'Test_3')
images = []
for img in os.listdir(test_path):
    img = os.path.join(test_path, img)
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)

# stack up images list to pass for prediction
images = np.vstack(images)
predictions = model.predict_classes(images, batch_size=10)
