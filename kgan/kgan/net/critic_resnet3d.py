"""
A 3D ResNet style critic.

"""

import numpy as np

import keras.models as km
import keras.layers as kl
import keras.activations as ka
import keras.layers.advanced_activations as kla

from .common_layers import *
from .resnet3d import *

def get_critic(input_shape, 
               nlevels=3, base_features=32, nconvl=2,
               conv_kernel_size=(5, 5, 5), updown_kernel_size=(2, 2, 2),
               updown_method='block',
               norm=None, activation='lrelu', dropout=None, **kwargs):
    """
    Returns a 3D CNN critic based on the ResNet model.
    Parameters
    ---------
    input_shape : np.array((3))
        Specifies the size of the input data, usually (weight, height, 1)
    
    nlevels : int (optional)
        The number of times (-1) to downsample the resolution
        Default 4.
    base_features: int (optional)
        The number of features the input layer should learn.
        Default 32.
    Returns
    -------
    model :
        The keras model.
    """
    
    input_state = kl.Input(shape=input_shape)

    #Input layer
    x = kl.Conv3D(base_features, (7,7,7), use_bias=False, padding='same')(input_state)
    if norm=='batch':
        x = kl.BatchNormalization(momentum=0.8)(x)
    elif norm=='layer':
        x = LayerNormalization()(x)
    x = NonLinearity()(x)

    for level in range(nlevels):
        
        if level != 0:
            # This layer ups the number of features while downsampling spatially
            x = DownBlock(x, base_features*2**level, dropout=dropout, 
                          conv_kernel_size=conv_kernel_size,
                          updown_kernel_size=updown_kernel_size,
                          norm=norm, activation=activation, **kwargs)
    
        for _ in range(nconvl):
            x = ResBlock(x, base_features*2**level, dropout=dropout, 
                         conv_kernel_size=conv_kernel_size, 
                         norm=norm, activation=activation, **kwargs)
        #if level in [1, 2]:
        #    x = ResBlock(x, base_features*2**level, dropout=dropout, **kwargs)

    #x = kl.GlobalAvgPool3D()(x)
    #x = kl.Dropout(0.15)(x)
    x = kl.Flatten()(x)
    x = kl.Dense(1)(x)

    model = km.Model(inputs = input_state, outputs = x)

    return model
