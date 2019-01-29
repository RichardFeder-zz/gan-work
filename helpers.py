import argparse
import os
import sys
import numpy as np
import math
import time
import cPickle as pickle
from torch.nn.functional import binary_cross_entropy_with_logits as bce

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

def objective_gan(fakeD, realD):
    labD = torch.cat((torch.ones(fakeD.size(0), 1) - 1e-3,
                      torch.zeros(realD.size(0), 1) + 1e-3))
    return bce(torch.cat((fakeD, realD)), Variable(labD))

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


def save_frames(fakes, reals, rhos, dir, iter):
    plt.figure(figsize=(9, 3))
    for i, rho in enumerate(rhos):
        plt.subplot(1, len(rhos), i+1)
        plt.title('Rho = '+str(rho))
        plt.scatter(fakes[i][:,0], fakes[i][:,1], label='Generator', color='b', alpha=0.5, s=1)
        plt.scatter(reals[i][:,0], reals[i][:,1], label='Truth', color='r', alpha=0.5, s=1)
        plt.legend()
    plt.savefig(dir+'/iteration_'+str(iter)+'.pdf', bbox_inches='tight')
    plt.close()




def plot_loss_iterations(lossD_vals, lossG_vals, directory):
    n = len(lossD_vals)

    lossG_vals = np.abs(lossG_vals)
    lossD_vals = np.abs(lossD_vals)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(list(torch.linspace(1, n, steps=n)), lossD_vals, label='Discriminator')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.subplot(1,2,2)
    plt.plot(list(torch.linspace(1, n, steps=n)), lossG_vals, label='Generator')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.savefig(directory+'/loss_GD.png', bbox_inches='tight')
    plt.close()

def LHS(array_list):
    d = len(array_list[0])
    LH_samples = np.zeros((d, len(array_list)))
    for i in xrange(d):
        samp = []
        for j in xrange(LH_samples.shape[1]):
            idx = np.random.randint(d-i)
            samp.append(array_list[j][idx])
            array_list[j] = np.delete(array_list[j], [idx])
        LH_samples[i,:] = np.array(samp)

    print 'Latin Hypercube sample array has shape', LH_samples.shape
    return LH_samples

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