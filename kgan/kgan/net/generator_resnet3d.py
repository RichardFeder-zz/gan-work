"""
A 3D ResNet generator.

"""

import numpy as np

import keras.models as km
import keras.layers as kl
import keras.activations as ka
import keras.layers.advanced_activations as kla

from .common_layers import *
from .resnet3d import *

def get_generator(input_shape, image_shape, 
                  nlevels=4, base_features=32, nconvl=2,
                  dropout=None, **kwargs):
    
    """
    Takes in noise and returns an image
    """

    # Get the input and reshape it into an image of the right shape.
    input_state = kl.Input(shape=input_shape)
    
    ii_width = image_shape[0]//2**nlevels
    ii_height = image_shape[1]//2**nlevels
    ii_depth = image_shape[2]//2**nlevels
    
    x = kl.Dense(units=ii_depth*ii_width*ii_height*base_features)(input_state)
    x = NonLinearity()(x)
    x = kl.Reshape((ii_width, ii_height, ii_depth, base_features))(x)
    
    #for _ in range(nconvl):
    #    x = ResBlock(x, base_features//2**(level+1), dropout=dropout, **kwargs)

    for level in range(nlevels):
        
        #Reduce nfeatures by half
        x = UpBlock(x, base_features//2**(level+1), dropout=dropout, **kwargs)
    
        for _ in range(nconvl):
            x = ResBlock(x, base_features//2**(level+1), dropout=dropout, **kwargs)
            
    x = clayer(1, activation='tanh')(x)
        
    # This last layer is linear
    #x = kl.Conv2D(1, kernel_size=(7,7), strides=(1,1), padding='same', activation='tanh')(x)
    #x = kl.Conv2D(1, kernel_size=(1,1), strides=(1,1), padding='same')(x)

    model = km.Model(inputs = input_state, outputs = x)

    return model
