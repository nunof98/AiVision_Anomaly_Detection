"""
Implementation of mvtec architecture inspired by:
https://github.com/cheapthrillandwine/Improving_Unsupervised_Defect_Segmentation/blob/master/Improving_AutoEncoder_Samples.ipynb
"""

import tensorflow as tf
from tensorflow import keras

# Preprocessing parameters
RESCALE = 1.0 / 255
PREPROCESSING_FUNCTION = None
PREPROCESSING = None
VMIN = 0.0
VMAX = 1.0
DYNAMIC_RANGE = VMAX - VMIN


def build_model(shape, color_mode):
    # set channels
    if color_mode == "grayscale":
        channels = 1
    elif color_mode == "rgb":
        channels = 3
        
    # define model
    input_img = keras.layers.Input(shape=(*shape, channels))
    # Encode-----------------------------------------------------------
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decode---------------------------------------------------------------------
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    decoded = keras.layers.Conv2D(channels, (3, 3), activation='sigmoid', padding='same')(x)
	
    encoder = keras.models.Model(input_img, encoded)
    autoencoder = keras.models.Model(input_img, decoded)
    
    return autoencoder, encoder
