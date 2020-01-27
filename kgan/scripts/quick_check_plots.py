"""
This is a script generates quick check plots for a model.


"""

import numpy as np

import sys
import yaml
import ast
import h5py
import glob

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt 

plt.rcParams['font.size'] = 16.0

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)

import keras.models as km
from kgan.net.gp_loss import *

import keras.losses
keras.losses.custom_loss = wasserstein_loss

from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({"wasserstein_loss": wasserstein_loss})


from kgan.tools.delta import *

def load_config(yaml_file):

    with open(yaml_file, 'r') as tfile:
        file_str = tfile.read()
        
    return yaml.load(file_str)

def estimate_pk(kki, img, boxsize, nbin=30, normalize=True):
    
    bins = np.linspace(np.min(kki), np.max(kki), nbin+1, endpoint=True)
    ibin = [np.logical_and(kki.ravel() >= bins[i], kki.ravel() < bins[i+1]) for i in range(nbin)]

    n = kki.shape[0]
    n3 = n**3
    kmin = 2*np.pi/boxsize
    d3k = kmin**3

    if normalize:
        nimg = img / np.mean(img.ravel()) - 1.0
        ft = np.absolute(np.fft.fftn(nimg)/n3)**2
    else:
        ft = np.absolute(np.fft.fftn(img)/n3)**2
    pkk = np.array([np.mean(ft.ravel()[ibin[i]]) for i in range(nbin)])
    kk = np.array([np.mean(kki.ravel()[ibin[i]]) for i in range(nbin)])

    return kk, pkk/d3k*(2*np.pi)**3

def quick_check_plot(real_imgs, test_imgs, fig=None):

    if fig is None:
        fig = plt.figure(figsize = (15, 5))

    kx = np.fft.fftfreq(64, d=1.0)*2*np.pi
    ky = np.fft.fftfreq(64, d=1.0)*2*np.pi
    kz = np.fft.fftfreq(64, d=1.0)*2*np.pi
    kk = np.sqrt(kx[:, np.newaxis, np.newaxis]**2 
                 + ky[np.newaxis, :, np.newaxis]**2 
                 + kz[np.newaxis, np.newaxis, :]**2)

    pk1r = []
    pk2f = []
    Nim = len(real_imgs)
    for i in range(Nim):
        k1, pk1 = estimate_pk(kk, squash_inv(real_imgs[i]), 64.0)
        k2, pk2 = estimate_pk(kk, squash_inv(test_imgs[i]), 64.0)
        #k1, pk1 = estimate_pk(kk, real_imgs[i])
        #k2, pk2 = estimate_pk(kk, test_imgs[i])

        pk1r.append(pk1)
        pk2f.append(pk2)
    pk1r = np.array(pk1r)
    pk2f = np.array(pk2f)

    ax1 = plt.subplot(2,2,1)
    ax1.plot(k1, pk1r.mean(0), '-o')
    ax1.fill_between(k1, pk1r.mean(0) - pk1r.std(0), 
                     pk1r.mean(0) + pk1r.std(0), color='C0', alpha=0.1)
    ax1.plot(k2, pk2f.mean(0), '-o')
    ax1.fill_between(k2, pk2f.mean(0) - pk2f.std(0), 
                     pk2f.mean(0) + pk2f.std(0), color='C1', alpha=0.1)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    
    ax2 = plt.subplot(2,2,3)
    ax2.plot(k1, np.sqrt(pk2f.mean(0)/pk1r.mean(0)), '-o')
    krange = np.linspace(np.min(k1), np.max(k1), 100)
    ax2.fill_between(krange, np.ones_like(krange)*0.9, np.ones_like(krange)*1.1, color='C5', alpha=0.1)
    ax2.fill_between(krange, np.ones_like(krange)*0.95, np.ones_like(krange)*1.05, color='C5', alpha=0.2)
    ax2.axhline(1.0, ls='--', c='grey')
    ax2.set_xscale('log')
    ax2.set_ylim(0.75, 1.25)

    ax3 = plt.subplot(1,2,2)
    foo = ax3.hist((real_imgs).ravel(), bins=50, edgecolor='C0', alpha=0.1)
    foot = ax3.hist((test_imgs).ravel(), bins=50, edgecolor='C1', alpha=0.1)
    #foo = hist(squash_inv(real_imgs).ravel(), bins=50, edgecolor='C0', alpha=0.1)
    #foot = hist(squash_inv(test_imgs).ravel(), bins=50, edgecolor='C1', alpha=0.1)
    ax3.set_yscale('log')

    return fig

if __name__ == '__main__':

    # Load the true data
    fp = "/work/06147/pberger/maverick2/gadget_runs/cosmo1/n512_512Mpc_cosmo1_z0_gridpart.h5"

    with h5py.File(fp, 'r') as data:
    
        delta = []
        for dk in list(data.keys())[:2]:
            delta_i = squash(data[dk][:])
            delta_i = np.array(split3d(delta_i, 8))
            delta.extend(delta_i)
    delta = np.array(delta)
    real_imgs = delta[:1024]
    real_imgs = np.array(real_imgs)

    # Load the hyper parameters from config
    conf_files = sys.argv[1:]
    conf_file = conf_files[rank]
    print(rank, conf_file)
    config = load_config(conf_file)

    latent_dim = int(config['train_config']['latent_dim'])
    nepochs = int(config['train_config']['nepochs'])

    nlevels  = config['net_config']['generator']['args']['nlevels']
    nconvl   = config['net_config']['generator']['args']['nconvl']
    lr_gamma = config['optimizer_config']['lr_gamma']
    lr       = config['optimizer_config']['args']['lr']
    beta1    = config['optimizer_config']['args']['beta_1']
    beta2    = config['optimizer_config']['args']['beta_2']
    decay    = config['optimizer_config']['args']['decay']

    model_path = config['save_model']
    model_dir = os.path.dirname(model_path)
    model_name = model_path.split('/')[-1].split('_')[-1].split('.')[0]

    model_list = np.sort(glob.glob(model_path + '*'))

    # Create a directory to store the results
    save_dir = os.path.join(model_dir, model_name)
    try:
        os.mkdir(save_dir)
    except OSError:
        pass

    # Loop through selected models
    #mis = [0, 5, 6, 8, 9, 10, 11, 15, 17, 20, 25, 29, 30, 31, 32, 33, 34]

    #for mi in mis:
    for mi in range(len(model_list)):
        model = km.load_model(model_list[mi])
        print("Loaded model:", model_list[mi])

        # Gen some imgs
        test_imgs = []
        for _ in range(4):
            test_imgs.extend(model.predict(np.random.normal(0, 1, (256, latent_dim))).reshape(256, 64, 64, 64))
        test_imgs=np.array(test_imgs)

        fig = quick_check_plot(real_imgs, test_imgs)

        epoch = model_list[mi].split('-')[-1]

        stt='nlevels=%s, nconvl=%s, lr_gamma=%s, 1r=%s, beta1=%s, beta2=%s, decay=%s, epoch=%s'
        fig.suptitle(stt % (nlevels, nconvl, lr_gamma, lr, beta1, beta2, decay, epoch))

        fig.savefig(os.path.join(save_dir, model_list[mi].split('/')[-1] + ".png"), fmt='png')
