"""
Layers for 3D ResNets.

"""

import numpy as np

import keras.models as km
import keras.layers as kl
import keras.activations as ka
import keras.layers.advanced_activations as kla

from .common_layers import *

def ResBlock(x, nf, dropout=None, norm='batch', activation='relu', **kwargs):
    
    shortcut = x
    x = clayer(nf, **kwargs)(x)
    if norm=='batch':
        x = kl.BatchNormalization(momentum=0.8)(x)
    elif norm=='layer':
        x = LayerNormalization()(x)
    x = NonLinearity(activation=activation)(x)
    if dropout is not None:
        x = kl.Dropout(dropout)(x)
    x = clayer(nf, **kwargs)(x)
    x = kl.Add()([x, shortcut])
    if norm=='batch':
        x = kl.BatchNormalization(momentum=0.8)(x)
    elif norm=='layer':
        x = LayerNormalization()(x)
    x = NonLinearity(activation=activation)(x)
    if dropout is not None:
        x = kl.Dropout(dropout)(x)
    
    return x


def UpBlock(x, nf, dropout=None, **kwargs):
    """
    This increases the dimensionality using transposed convolutions.
    """
    
    shortcut = kl.Conv3DTranspose(nf, kernel_size=(1,1,1), strides=(2,2,2), **kwargs)(x)
    
    x = kl.Conv3DTranspose(nf, kernel_size=(3,3,3), strides=(2,2,2), padding='same')(x)
    x = batchnorm()(x)
    x = NonLinearity()(x)
    
    x = clayer(nf, **kwargs)(x)
    x = kl.Add()([x, shortcut])
    x = batchnorm()(x)
    x = NonLinearity()(x)
    
    return x


def DownBlock(x, nf, dropout=None, norm=None, activation='lrelu', **kwargs):
    """
    This reduces the dimensionality using convolutions.
    """
    
    shortcut = clayer(nf, kernel_size=(1,1,1), strides=(2,2,2), **kwargs)(x)
    x = clayer(nf, kernel_size=(2,2,2), strides=(2,2,2), **kwargs)(x)
    if norm=='batch':
        x = kl.BatchNormalization(momentum=0.8)(x)
    elif norm=='layer':
        x = LayerNormalization()(x)
    x = NonLinearity(activation=activation)(x)
    if dropout is not None:
        x = kl.Dropout(dropout)(x)
    x = clayer(nf, **kwargs)(x)
    x = kl.Add()([x, shortcut])
    if norm=='batch':
        x = kl.BatchNormalization(momentum=0.8)(x)
    elif norm=='layer':
        x = LayerNormalization()(x)
    x = NonLinearity(activation=activation)(x)
    if dropout is not None:
        x = kl.Dropout(dropout)(x)
        
    return x
