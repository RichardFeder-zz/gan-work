import os
import numpy as np
import cPickle as pickle
from torch.nn.functional import binary_cross_entropy_with_logits as bce
import imageio
import sys
from astropy.io import fits
import torchvision.transforms.functional as TF
from torch.autograd import Variable, grad
import torch.nn as nn
import torch
from torch.utils import data

import matplotlib
if sys.platform=='darwin':
    base_dir = '/Users/richardfeder/Documents/caltech/gan_work/results/'
    matplotlib.use('tkAgg')
elif sys.platform=='linux2':
    base_dir = '/home1/06224/rfederst/gan-work/results/'
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

nparam_dict = dict({'2d_gaussian':5, '1d_gaussian':2, 'bernoulli':2, 'ring':2, 'grid':2})
outparam_dict = dict({'2d_gaussian':2, '1d_gaussian':1, 'bernoulli':1, 'ring':2, 'grid':2})

Device = 'cpu'

mu_range = np.linspace(-1.0, 1.0, 8)
sig_range = np.linspace(0.1, 2.0, 8)


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

def make_size_array(imgsize):

    n_powers = int(np.log2(imgsize)-2)
    sizes = np.zeros(n_powers)
    for i in xrange(0, n_powers):
        sizes[i] = int(2 ** i)
    sizes = np.flip(sizes, 0)

    return sizes


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
    
    with open(dir+'/params.txt', 'w') as file:
        file.write(pickle.dumps(param_dict))
    file.close()
    
    with open(dir+'/params_read.txt', 'w') as file2:
        for key in param_dict:
            file2.write(key+': '+str(param_dict[key])+'\n')
    file2.close()


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
        plt.legend(loc=1)
    plt.savefig(dir+'/iteration_'+str(iter)+'.png', bbox_inches='tight')
    plt.close()


def plot(x, y, frame_directory=None, iteration=0):
    real = x.cpu().data.numpy()
    fake = y.cpu().data.numpy()
    lims = (x.data.min() - 0.1, x.data.max() + 0.1)
    plt.figure()
    plt.plot(real[:, 0], real[:, 1], '.', label='real', color='g')
    plt.plot(fake[:, 0], fake[:, 1], '.', alpha=0.25, label='fake', color='b')
    plt.axis('off')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.xlim(*lims)
    plt.ylim(*lims)
    plt.legend(loc=2)
    if frame_directory is not None:
        plt.savefig(frame_directory+'/iteration_'+str(iteration)+'.png', bbox_inches='tight')
    plt.close()


def make_gif(dir):
    images = []
    files = os.listdir(dir)
    if len(files) > 0:
        for file in os.listdir(dir):
            images.append(imageio.imread(dir+'/'+file))
        imageio.mimsave(dir+'/movie.gif', images, fps=3)
    else:
        print('No images to save for make_gif().')

def sample_noise(bs, d):
    z = torch.randn(bs, d).float()
    return Variable(z.to(Device), requires_grad=True)


def plot_loss_iterations(lossD_vals, lossG_vals, directory):
    n = len(lossD_vals)

    lossG_vals = np.abs(lossG_vals)
    lossD_vals = np.abs(lossD_vals)


    np.savetxt(directory+'/lossG.txt', lossG_vals)
    np.savetxt(directory+'/lossD.txt', lossD_vals)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.scatter(np.arange(n), lossD_vals, label='Discriminator', s=2, alpha=0.5)
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.subplot(1,2,2)
    plt.scatter(np.arange(n), lossG_vals, label='Generator', s=2, alpha=0.5)
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


def means_circle(k=8):
    p = 3.14159265359
    t = torch.linspace(0, 2 * p - (2 * p / k), k)
    m = torch.cat((torch.sin(t).view(-1, 1),torch.cos(t).view(-1, 1)), 1)
    return m

def means_grid(k=25):
    m = torch.zeros(k, 2)
    s = int(torch.sqrt(torch.Tensor([k]))[0] / 2)
    cnt = 0
    for i in range(- s, s + 1):
        for j in range(- s, s + 1):
            m[cnt][0] = i
            m[cnt][1] = j
            cnt += 1
    return m / s


def draw_true_samples(nsamp, opt, samp_type='2d_gaussian', LH=None):

    '''For 2d_gaussian, conditional arguments are mu_0, sig_1, mu_2, sig_2, rho in that order'''
    if samp_type=='2d_gaussian':

        for i in xrange(opt.nmix):

            if LH is not None:
                samp = LH[np.random.randint(LH.shape[0])]
                mus = np.array([samp[0], samp[2]])
                sigs = np.array([samp[1], samp[3]])
                rho = np.array([samp[4]])
            else:
                mus = np.random.choice(mu_range, size=2)
                sigs = np.random.choice(sig_range, size=2)
                rho = np.random.choice(np.linspace(-1.0, 1.0, opt.ngrid))

            cov = np.array([[sigs[0]**2, rho*sigs[0]*sigs[1]],[rho*sigs[0]*sigs[1], sigs[1]**2]])

            if i > 0:
                cond_temp = np.column_stack((np.full(nsamp/opt.nmix, mus[0]),  np.full(nsamp/opt.nmix, sigs[0]), np.full(nsamp/opt.nmix, mus[1]), np.full(nsamp/opt.nmix, sigs[1]), np.full(nsamp/opt.nmix, rho)))
                conditional_params = np.vstack((conditional_params, cond_temp))
                temp = np.random.multivariate_normal(mus, cov, nsamp/opt.nmix)
                s = np.vstack((s, temp))
            else:
                conditional_params = np.column_stack((np.full(nsamp/opt.nmix, mus[0]),  np.full(nsamp/opt.nmix, sigs[0]), np.full(nsamp/opt.nmix, mus[1]), np.full(nsamp/opt.nmix, sigs[1]), np.full(nsamp/opt.nmix, rho)))
                s = np.random.multivariate_normal(mus, cov, nsamp/opt.nmix)
        
    # Conditional arguments are mu and sigma in that order '''
    elif samp_type=='1d_gaussian':
        for i in xrange(opt.nmix):
            if LH is not None:
                samp = LH[np.random.randint(LH.shape[0])]
                mu = samp[0]
                sig = samp[1]
            else:
                mu = np.random.choice(mu_range)
                sig = np.random.choice(sig_range)

            if i > 0:
                temp = np.random.normal(mu, sig, nsamp/opt.nmix)
                print 'temp:', temp.shape
                s = np.append(s, temp)
                print s.shape
                cond_temp = np.column_stack((np.full(nsamp/opt.nmix, mu), np.full(nsamp/opt.nmix, sig)))
                conditional_params = np.vstack((conditional_params, cond_temp))
            else:
                s = np.random.normal(mu, sig, nsamp/opt.nmix)
                print s.shape
                conditional_params = np.column_stack((np.full(nsamp/opt.nmix, mu), np.full(nsamp/opt.nmix, sig)))

        if opt.verbosity > 1:
            print conditional_params.shape
            print s.shape
            print conditional_params
            print s

    
    elif samp_type=='bernoulli':
        n = argv[0]
        p = argv[1]

        conditional_params = torch.cat((n,p), 0)
        s = torch.from_numpy(np.random.binomial(n, p, nsamp))

    elif samp_type == 'ring':
        m = means_circle()
        i = torch.zeros(nsamp).random_(m.size(0)).long()
        s = torch.randn(nsamp, 2) * 0.02 + m[i]
    
    elif samp_type == 'grid':
        k = 9
        std = 1.0
        m = means_grid(k)
        i = torch.zeros(nsamp).random_(m.size(0)).long()
        s = torch.randn(nsamp, 2) * 0.02 + m[i]

    if opt.n_cond_params==0:
        return s
    else:
        return s, conditional_params


def inverse_loglike_transform(s, a=4):
    im = a*(s+1)/(1-s)
    return im
    
def loglike_transform(x, a=5):
    return (2*x/(x+a)) - 1

