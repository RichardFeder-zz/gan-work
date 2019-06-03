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
                  base_features=(32*2**4), nconvl=2,
                  nlevels=4, updown_method='conv',
                  conv_kernel_size=(5, 5, 5),
                  updown_kernel_size=(2, 2, 2),
                  dropout=None, **kwargs):
    
    """
    Takes in noise and returns a volume of the LSS.

    Parameters
    ---------

    input_shape : tuple (2)
        The size of the random noise vector.

    image_shape : tuple (3)
        The eventual shape of the output image.

    base_features : int
        The number of features to start with.
        Default 32*2**4

    nconvl : int
        The number of successive convolutions on each level.
        Default 2.

    nlevels : int
        The number of times the resolution is doubled.
        Default 4.

    updown_method : string
        'conv' : Use a simple transposed convolution.
        'block' : Use a residual block.
    
    Returns
    -------
    
    model : obj - `keras.models.Model`
        The keras model.
    
    
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
        if updown_method == 'conv':
            x = kl.Conv3DTranspose(base_features//2**(level+1), 
                                   kernel_size=updown_kernel_size, strides=(2,2,2),
                                   padding='same')(x)
            x = batchnorm()(x)
            x = NonLinearity()(x)
        elif updown_method == 'block':
            x = UpBlock(x, base_features//2**(level+1), 
                        conv_kernel_size=conv_kernel_size,
                        updown_kernel_size=updown_kernel_size,
                        dropout=dropout, **kwargs)
            
        # Now perform successive convolutions
        nconvli = 1 if level == (nlevels-1) else nconvl
        for _ in range(nconvli):
            x = ResBlock(x, base_features//2**(level+1), 
                         conv_kernel_size=conv_kernel_size, 
                         dropout=dropout, **kwargs)

    # This last layer is linear
    x = clayer(1, activation='tanh')(x)

    #x = kl.Conv2D(1, kernel_size=(7,7), strides=(1,1), padding='same', activation='tanh')(x)
    #x = kl.Conv2D(1, kernel_size=(1,1), strides=(1,1), padding='same')(x)

    model = km.Model(inputs = input_state, outputs = x)

    return model
