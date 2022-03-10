# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 12:33:23 2022

@author: Mohamed Farag
"""
#%% Imorting required modules
import tensorflow as tf
# import tensorflow_addons as tfa
import h5py
from tensorflow.keras import backend as K
import keras
import os
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import matplotlib.image as mpimg
import numpy as np
import tensorflow_datasets as tfds
import random as python_random

#%% Define your own model
# 1- Create your own base_model

CUDA_VISIBLE_DEVICES="" ;PYTHONHASHSEED=123

np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(123)

# Loading main Backbone
data_augmentation = tf.keras.Sequential(
  
    [   
        tf.keras.layers.experimental.preprocessing.Resizing(256,256),
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        # tf.keras.layers.experimental.preprocessing.Normalization(),  
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical",seed=123),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.5, seed=123),
        # tf.keras.layers.experimental.preprocessing.RandomHeight(factor = (0.2,0.5), seed=123),
        # tf.keras.layers.experimental.preprocessing.RandomWidth(factor = (0.2, 0.5), seed=123),
        # tf.keras.layers.experimental.preprocessing.RandomZoom(0.2, 0.3, seed=123)
    ]
)




inputs = tf.keras.Input(shape=(480, 480, 3))

x = data_augmentation(inputs)

base_model = tf.keras.applications.ResNet50V2(
  
    weights= 'imagenet',  # Load weights pre-trained on ImageNet.
    # input_shape=(480, 480, 3)
    input_shape=(256,256,3)
    , include_top = False, input_tensor = x)

# for layer in base_model.layers:
#   layer.trainable = False
base_model.summary()

#%% Load your data
# 2- Load your data.

"""It can be loaded as numpy array"""

#%% Define your activations layers

# 3- Choose the layers to visualize its activations from your base_model
"""Activations are the outputs from the Convolutional Layers"""

layer_outputs = [layer.output for layer in base_model.layers[3:-100]]
"""Note***: I selected this range based on my trials, you may face problems
with your code, however it is easy to solve by just ensuring graph consistency"""

len(layer_outputs)

#%% Create the activation model
# 4- Create an activations model that will output your activations selected from previous step.
activation_model = tf.keras.models.Model(inputs=base_model.input, outputs=layer_outputs)

# 5- Choose a sample from your data and expand the dimensions plus resizing the image.
"""The values are also chossen based on my trials:
    - you can choose different size
    - you can choose your own data"""

test = #get your data here
test = np.expand_dims(test, 0)
test = tf.keras.layers.experimental.preprocessing.Resizing(480,480)(test)
# data pre-processing step, but i include it at my network as a layer.
#test *= 1/255

#%% Get your outputs

# 6- Get your outputs
activations = activation_model.predict(test)

#%% Visualize all layers' activations

# 7- This code is developed by Francoise Chollet
"""Deep Learning with Python 1st edition"""

layer_names = []
for layer in base_model.layers[3:-100]:
  layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
  n_features = layer_activation.shape[-1]
  size = layer_activation.shape[1]
  n_cols = n_features // images_per_row
  display_grid = np.zeros((size * n_cols, images_per_row * size))
  for col in range(n_cols):
    for row in range(images_per_row):
      channel_image = layer_activation[0,
        :, :,
        col * images_per_row + row]
      channel_image -= channel_image.mean()
      channel_image /= channel_image.std()
      channel_image *= 64
      channel_image += 128
      channel_image = np.clip(channel_image, 0, 255).astype('uint8')
      display_grid[col * size : (col + 1) * size,
        row * size : (row + 1) * size] = channel_image
  scale = 1. / size
  plt.figure(figsize=(scale * display_grid.shape[1],
  scale * display_grid.shape[0]))
  plt.title(layer_name)
  plt.grid(False)
  plt.imshow(display_grid, aspect='auto', cmap='viridis')
  

  




