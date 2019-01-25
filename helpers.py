import argparse
import os
import sys
import numpy as np
import math
import time
import cPickle as pickle

from torch.autograd import Variable, grad
import torch.nn as nn
import torch

import matplotlib
matplotlib.use('tkAgg') 
import matplotlib.pyplot as plt

nparam_dict = dict({'2d_gaussian':5, '1d_gaussian':2, 'bernoulli':2, 'ring':2, 'grid':2})
outparam_dict = dict({'2d_gaussian':2, '1d_gaussian':1, 'bernoulli':1, 'ring':2, 'grid':2})

base_dir = '/Users/richardfeder/Documents/caltech/gan_work/results/'


def create_directories(time_string):
    new_dir = base_dir+time_string
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)
    frame_dir = new_dir+'/frames'
    if not os.path.isdir(frame_dir):
        os.makedirs(frame_dir)
    print 'New directories:'
    print new_dir
    print frame_dir
    return new_dir, frame_dir


def objective_wgan(fakeD, realD):
    return fakeD.mean() - realD.mean()

def save_nn(model, path):
    torch.save(model.state_dict(), path)

def save_params(dir, opt):
    # save parameters as dictionary, then pickle them to txt file
    param_dict = vars(opt)
    print param_dict
    param_dict['n_cond_params'] = nparam_dict[opt.sample_type]
    param_dict['n_out'] = outparam_dict[opt.sample_type]
    with open(dir+'/params.txt', 'w') as file:
        file.write(pickle.dumps(param_dict))

def init_loss(loss_func):
    # Loss functions
    if loss_func=='MSE':
        adversarial_loss = torch.nn.MSELoss()
    elif loss_func=='KL':
        adversarial_loss = torch.nn.KLDivLoss()
    elif loss_func=='SmoothL1':
        adversarial_loss = torch.nn.SmoothL1Loss()
    elif loss_func=='L1':
        adversarial_loss = torch.nn.l1_loss()
    return adversarial_loss

def plot_comp_resources(timearray, directory):

    labels = ['Sample Drawing', 'Generator', 'Discriminator + Loss', 'Gradient', 'Backprop/Update']
    timesum = np.sum(timearray, axis=1)

    plt.figure()
    plt.pie(timesum, labels=labels, autopct='%1.1f%%', shadow=True)
    plt.savefig(directory+'/comp_resources.png', bbox_inches='tight')
    plt.close()

def plot_loss_iterations(lossD_vals, lossG_vals, directory):
    n = len(lossD_vals)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(list(torch.linspace(1, n, steps=n)), lossD_vals, label='Discriminator')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Loss * -1')
    plt.subplot(1,2,2)
    plt.plot(list(torch.linspace(1, n, steps=n)), lossG_vals, label='Generator')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.savefig(directory+'/loss_GD.png', bbox_inches='tight')
    plt.close()



class Perceptron(torch.nn.Module):
    def __init__(self, sizes, final=None):
        super(Perceptron, self).__init__() # what does this line do?
        layers = []
        print 'Initializing Neural Net'
        for i in xrange(len(sizes) - 1):
            print 'Layer', i, ':', sizes[i], sizes[i+1]
            layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
            if i != (len(sizes) - 2):
                layers.append(torch.nn.ReLU())
        if final is not None:
            layers.append(final())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)