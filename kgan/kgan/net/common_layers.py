"""
Common layers.

"""

import numpy as np

import keras.layers as kl
import keras.activations as ka
import keras.layers.advanced_activations as kla

def batchnorm(center=True, momentum=0.8):
    return kl.BatchNormalization(center=center, momentum=momentum)

def clayer(nfeatures, kernel_size=(3, 3, 3), strides=(1,1, 1), use_bias=True, **kwargs):
    
    return kl.Conv3D(nfeatures,
                     kernel_size=kernel_size,strides=strides, 
                     use_bias=use_bias, padding='same', **kwargs)

def NonLinearity(activation='relu'):
    
    if activation == 'relu':
        return kl.Activation('relu')
    elif activation == 'lrelu':
        return kla.LeakyReLU(alpha=0.2)
