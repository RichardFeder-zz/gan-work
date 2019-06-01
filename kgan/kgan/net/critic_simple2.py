"""
Another simple critic model.

"""

import numpy as np

import keras.models as km
import keras.layers as kl
import keras.activations as ka
import keras.layers.advanced_activations as kla

from .common_layers import *

def get_critic2(input_shape, base_features=32, norm=None):
    """

    The critic proposed in https://arxiv.org/abs/1903.10524.

    """
    
    input_state = kl.Input(shape=input_shape)

    #Input layer
    x = kl.Conv3D(base_features, (7,7,7), padding='same', strides=2)(input_state)
    if norm=='batch':
        x = kl.BatchNormalization(momentum=0.8)(x)
    elif norm=='layer':
        x = LayerNormalization()(x)
    x = kla.LeakyReLU(alpha=0.2)(x)
    x = kl.Conv3D(base_features*2, (5,5,5), padding='same', strides=2)(x)
    if norm=='batch':
        x = kl.BatchNormalization(momentum=0.8)(x)
    elif norm=='layer':
        x = LayerNormalization()(x)
    x = kla.LeakyReLU(alpha=0.2)(x)
    x = kl.Conv3D(base_features*4, (3,3,3), padding='same', strides=2)(x)
    if norm=='batch':
        x = kl.BatchNormalization(momentum=0.8)(x)
    elif norm=='layer':
        x = LayerNormalization()(x)
    x = kla.LeakyReLU(alpha=0.2)(x)
    x = kl.Conv3D(base_features*8, (1,1,1), padding='same', strides=1)(x)
    if norm=='batch':
        x = kl.BatchNormalization(momentum=0.8)(x)
    elif norm=='layer':
        x = LayerNormalization()(x)
    x = kla.LeakyReLU(alpha=0.2)(x)
    
    x = kl.Dropout(0.25)(x)
    x = kl.Flatten()(x)
    #Linear activation for the critic
    x = kl.Dense(units=1)(x)

    model = km.Model(inputs = input_state, outputs = x)

    return model
