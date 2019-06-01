import keras.backend as K
import numpy as np
from keras.layers.merge import _Merge

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def _compute_gradients(tensor, var_list):
    grads = K.gradients(tensor, var_list)
    return [grad if grad is not None else K.zeros_like(var)
            for var, grad in zip(var_list, grads)]

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated 3D image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((8, 1, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    gradients = _compute_gradients(y_pred, [averaged_samples])[0]
    #gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)
