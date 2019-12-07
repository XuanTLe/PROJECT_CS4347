%tensorflow_version 1.x

import os 
os.chdir ('/content/drive/My Drive/Project_CS4347/PROJECT_CS4347/')
working_dir = os.getcwd()

from PIL import Image
import tensorflow as tf
import tensorflow.keras
import glob
import matplotlib
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

# parameters
epochs = 22
lr = 0.001
batch_size = 200
save_dir = os.path.join(working_dir, 'models')
data_augmentation = True
model_name = 'model.h5'
model_path = os.path.join(save_dir, model_name)

train_path = os.path.join('/run/shm/PROJECT_CS4347/', 'Train')
valid_path = os.path.join('/run/shm/', 'Validation')

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True, seed = 1)

test_generator = test_datagen.flow_from_directory(
        directory=valid_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')


# Initialize the model VGG16
vgg16_model = tensorflow.keras.applications.vgg16.VGG16()
# Create our empty Sequential model 
model = Sequential()
# For every layer in original add them into our model except the last one
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False

# Add our new layers and last layer (classification) for 2 categories in our model
model.add(Dense(2, activation = 'softmax'))
    
# Train our model
model.compile(Adam(lr=lr), loss="categorical_crossentropy", metrics=["accuracy"])

hist = model.fit_generator(train_generator, validation_data=test_generator, epochs=epochs, verbose=1)
model.save(model_path)

# save history of train and valid output for later graphing
f = open('history_train_val.pckl', 'wb')
pickle.dump(hist.history, f)
f.close()

# retrieve:    
#f = open('history_train_val.pckl', 'rb')
#history = pickle.load(f)
#f.close()

# plot accuracy graphs for training and test data
fig = plt.figure(figsize=(12,8))
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig.savefig((os.path.join(working_dir, 'Plots', 'accuracy.jpg')))
plt.close(fig)

# plot loss graphs for training and test data
fig = plt.figure(figsize=(12,8))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig.savefig((os.path.join(working_dir, 'Plots', 'loss.jpg')))
plt.close(fig)
