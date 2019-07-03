import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable, grad
from torch.nn.functional import binary_cross_entropy_with_logits as bce
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import sys
import time
import h5py
from matplotlib import cm
from scipy import stats
import numpy as np
import cPickle as pickle
from IPython.display import Image
from power_spec import *
from models import *
from helpers import *
import astropy
from astropy.io import fits
from powerbox import get_power
import Pk_library as PKL
from PIL import Image
import scipy.ndimage
from mpl_toolkits.mplot3d import Axes3D
    
class nbody_dataset():

    def __init__(self, cubedim=64, length=512):

        self.base_path = '/work/06224/rfederst/maverick2/'
        self.cubedim = cubedim
        self.data_path = '/work/06147/pberger/maverick2/gadget_runs/cosmo1/'
        self.name_base = 'n512_512Mpc_cosmo1_seed'
        self.datasims = []
        self.zlist = []
        self.length=length
        self.z_idx_dict = dict({10:'000',7.5:'001',5.:'002',3.:'003',2.:'004',1.5:'005',1.:'006',0.5:'007',0.25:'008',0.:'009'})
        self.redshift_bins = np.array([10.,7.5,5.,3.,2.,1.5,1.,0.5 ,0.25,0.])
        self.colormap = matplotlib.cm.jet(np.linspace(1, 0.1, len(self.redshift_bins))) # for plotting different zs


    def load_in_sims(self, nsims, loglike_a=None, redshift_idxs=None):
        if redshift_idxs is None:
            with h5py.File(self.data_path+'n512_512Mpc_cosmo1_z0_gridpart.h5', 'r') as ofile:
                for i in xrange(nsims):
                    sim = ofile['seed'+str(i+1)][()]
                    self.datasims = partition_cube(sim, self.length, self.cubedim, self.datasims, \
                                                   loglike_a=loglike_a)

            ofile.close()

        else:
            for i in xrange(nsims):
                with h5py.File(self.data_path + self.name_base + str(i+1)+'_gridpart.h5', 'r') as ofile:
                    for idx in redshift_idxs:
                        print(idx, self.redshift_bins[idx])
                        sim = ofile['%3.3d'%(idx)][()]
                        self.datasims, self.zlist = partition_cube(sim, self.length, self.cubedim, self.datasims, cparam_list=self.zlist, z=self.redshift_bins[idx],loglike_a=loglike_a)
                ofile.close()
        
    def compare_pk_diffz(self, model, pdict, redshift_idxs, nsamp=100, timestr=None):
        allgenpks = []
        allgenks = []
        allrealpks = []
        allrealks = []
        for zed_idx in redshift_idxs:
            with h5py.File(self.data_path + self.name_base + str(1)+'_gridpart.h5', 'r') as ofile:
                sim = ofile['%3.3d'%(zed_idx)][()]
                realsims = partition_cube(sim, self.length, self.cubedim, [], z=self.redshift_bins[zed_idx])
            
            
#             self.load_in_sims(1, redshift_idxs=[zed_idx])
                realpks, realks = self.compute_power_spectra(np.array(realsims)[:nsamp])
                gen_samps = self.get_samples(model, nsamp, pdict, n_conditional=1, c=self.redshift_bins[zed_idx])
                genpks, genks = self.compute_power_spectra(gen_samps, inverse_loglike_a=4.0)
                
                allgenpks.append(genpks)
                allgenks.append(genks)
                allrealpks.append(realpks)
                allrealks.append(realks)
                
                self.plot_powerspectra(genpk=genpks, genkbins=genks, realpk=realpks, realkbins=realks, timestr=timestr, z=self.redshift_bins[zed_idx])
            ofile.close()
        return allgenpks, allgenks, allrealpks, allrealks

    def restore_generator(self, timestring, epoch=None, n_condparam=0, discriminator=False):
        print('discriminator:', discriminator)
        filepath = self.base_path + '/results/' + timestring
        sizes = np.array([8., 4., 2., 1.])
        print(sizes)
        filen = open(filepath+'/params.txt','r')
        pdict = pickle.load(filen)
        model = DC_Generator3D(pdict['ngpu'], 1, pdict['latent_dim']+n_condparam, pdict['ngf'], sizes)
        
        if epoch is None:
            model.load_state_dict(torch.load(filepath+'/netG', map_location='cpu'))
        else:
            if discriminator:
                disc_model = DC_Discriminator3D(pdict['ngpu'], 1, pdict['ndf'], sizes, n_cond_features=n_condparam)
                disc_model.load_state_dict(torch.load(filepath+'/netD_epoch_'+str(epoch), map_location='cpu'))
                disc_model.eval()
            model.load_state_dict(torch.load(filepath+'/netG_epoch_'+str(epoch), map_location='cpu'))
        model.eval()
        if discriminator:
            return model, pdict, disc_model
        return model, pdict

    def get_samples(self, generator, nsamp, pdict, n_conditional=0, c=None, discriminator=None):
        z = torch.randn(nsamp, pdict['latent_dim']+n_conditional, 1, 1, 1).float()
        if c is not None:
            z[:,-1] = c
        gensamps = generator(z)
        if discriminator is not None:
            disc_outputs = discriminator(gensamps)
            return gensamps.detach().numpy(), disc_outputs.detach().numpy()
        return gensamps.detach().numpy()

    def view_sim_2d_slices(self, sim):
        fig, ax = plt.subplots(1, 1)
        print(ax)
        tracker = IndexTracker(ax, sim)
        fig.canvas.mpl_connect('button_press_event', tracker.onscroll)
        plt.show()
        
    def compute_power_spectra(self, vols, inverse_loglike_a=None, unsqueeze=True):

        pks, power_kbins = [], []
        if inverse_loglike_a is not None: # for generated data                                            
            vols = inverse_loglike_transform(vols, a=inverse_loglike_a)
            if unsqueeze:
                vols = vols[:,0,:,:,:] # gets rid of single channel                                           

        kbins = 10**(np.linspace(-1, 2, 30))

        for i in xrange(vols.shape[0]):
            pk, bins = get_power(vols[i]-np.mean(vols[i]), self.cubedim, bins=kbins)

            if np.isnan(pk).all():
                print('NaN for power spectrum')
                continue
            pks.append(pk)

        return np.array(pks), np.array(bins)

    def compute_average_cross_correlation(self, npairs=100, gen_samples=None, real_samples=None):
        xcorrs, kbin_list = [], []
        kbins = 10**(np.linspace(-1, 2, 30))
        for i in xrange(npairs):

            if gen_samples is None:
                idxs = np.random.choice(real_samples.shape[0], 2, replace=False)
                reali = real_samples[idxs[0]]-np.mean(real_samples[idxs[0]])
                realj = real_samples[idxs[1]]-np.mean(real_samples[idxs[1]])

                print(reali.shape)
                print(realj.shape)

                xc, ks = get_power(deltax=reali, boxlength=self.cubedim, deltax2=realj, log_bins=True)

            elif real_samples is None:
                idxs = np.random.choice(gen_samples.shape[0], 2, replace=False)
                geni = gen_samples[idxs[0]]-np.mean(gen_samples[idxs[0]])
                genj = gen_samples[idxs[1]]-np.mean(gen_samples[idxs[1]])
                xc, ks = get_power(geni, self.cubedim, deltax2=genj, log_bins=True)

            else:
                idxreal = np.random.choice(real_samples.shape[0], 1, replace=False)
                idxgen = np.random.choice(gen_samples.shape[0], 1, replace=False)
                real = real_samples[idxreal]-np.mean(real_samples[idxreal])
                gen = gen_samples[idxgen]-np.mean(gen_samples[idxgen])

                xc, ks = get_power(real, self.cubedim, deltax2=gen, log_bins=True)

            xcorrs.append(xc)
            kbin_list.append(ks)

        return np.array(xcorrs), np.array(kbin_list)


    def compute_matter_bispectrum(self, vols, k1=0.1, k2=0.5):
        thetas = np.linspace(0.0, 2.5, 10)
        bks = []
        for i in xrange(vols.shape[0]):
            bis = PKL.Bk(vols[i], float(self.cubedim), k1, k2, thetas)
            bks.append(bis.B)
        return bks, thetas

    def plot_voxel_pdf(self, real_vols=None, gen_vols=None, nbins=100, timestr=None, epoch=None, gen_vols2=None):

        plt.figure(figsize=(8,6))
        if real_vols is not None:
            _, bins, _ = plt.hist(real_vols.flatten(), bins=nbins, histtype='step',color='g', label='nbody', normed\
=True)
            maxval = np.max(real_vols[0])
        if gen_vols is not None:
            if real_vols is not None:
                binz = bins
            else:
                binz = nbins
            plt.hist(gen_vols.flatten(), bins=binz, histtype='step', label='GAN', color='b', normed=True)
            maxval = np.max(gen_vols[0])
        if gen_vols2 is not None:
            plt.hist(gen_vols2.flatten(), bins=binz, histtype='step', color='c', label='WGAN-GP', normed=True)

        plt.yscale('log')
        if maxval > 10: # if data are not scaled between -1 and 1                                         
            plt.xscale('log')
        plt.legend()
        plt.ylabel('Normalized Counts')
        plt.xlabel('Density')
        plt.title('Voxel PDF of Samples')
        if timestr is not None:
            plt.savefig('figures/gif_dir/'+timestr+'/voxel_pdf_epoch'+str(epoch)+'.pdf', bbox_inches='tight')
        plt.show()

    def plot_multi_z_vpdfs(self, model=None, pdict=None, zs=None, nsamp=50, timestr=None, epoch=None):
        pks, ks = [], []
        zslices = []
        if zs is not None:
            zeds = zs
        else:
            zeds = self.redshift_bins
            
        colormap = matplotlib.cm.jet(np.linspace(1, 0.1, len(zeds)))
        plt.figure(figsize=(8,6))
        if model is None:
            plt.title('GADGET N-body Samples')
        else:
            plt.title('Generated Samples')
        for i, zed in enumerate(zeds):
            if model is not None:
                gen_samps = self.get_samples(model, nsamp, pdict, n_conditional=1, c=zed)
                pk, kbins = self.compute_power_spectra(gen_samps, inverse_loglike_a=4.0)

            else:
                print('index is ', int(self.z_idx_dict[zed]))
                self.datasims = []
                self.load_in_sims(1, loglike_a=4., redshift_idxs=[int(self.z_idx_dict[zed])])
                print(len(self.datasims))
                gen_samps = np.array(self.datasims[:nsamp])
                print(gen_samps.shape)
                pk, kbins = self.compute_power_spectra(gen_samps, inverse_loglike_a=4.0, unsqueeze=False)
            pks.append(pk)
            ks.append(kbins)
            plt.hist(gen_samps.flatten(), bins=100, label='z='+str(zed), histtype='step', color=colormap[i], normed=True)
        plt.yscale('log')
        plt.ylabel('Scaled density (a=4)')
        plt.legend()
        if timestr is not None:
            if not os.path.isdir('figures/gif_dir/'+timestr):
                os.mkdir('figures/gif_dir/'+timestr)
            plt.savefig('figures/gif_dir/'+timestr+'/multiz_voxel_pdf_epoch'+str(epoch)+'.pdf', bbox_inches='tight')
        plt.show()
        
        
        
        plt.figure(figsize=(8,6))
        if model is None:
            plt.title('GADGET N-body Samples')
        else:
            plt.title('Generated Samples')
        for i in xrange(len(pks)):
            plt.fill_between(ks[i], np.percentile(pks[i], 16, axis=0), np.percentile(pks[i], 84, axis=0), alpha=0.3, color=colormap[i])
            plt.plot(ks[i], np.median(pks[i], axis=0), marker='.', c=colormap[i], label='z='+str(zeds[i]))
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('$k$', fontsize=14)
        plt.ylabel('P(k)', fontsize=14)
        plt.ylim(2e-1, 2e4)
        plt.legend()
        if timestr is not None:
            plt.savefig('figures/gif_dir/'+timestr+'/multiz_pk_epoch'+str(epoch)+'.pdf', bbox_inches='tight')
        plt.show()

    def make_zslice_gif(self, model, timestr, epoch, zs=None, fps=2):
        fixed_z = torch.randn(1, 201, 1, 1, 1).float()
        zslices = np.zeros((len(nbody.redshift_bins),64,64))
        images = []
        iteration = nbody.redshift_bins
        if zs is not None:
            iteration = zs

        for i in xrange(len(iteration)):
            fixed_z[:,-1] = iteration[i]
            gen_samp = model(fixed_z).detach().numpy()
            ploop = (cm.gist_earth((gen_samp[0][0][10,:,:]+1)/2)*255).astype('uint8')
            images.append(Image.fromarray(ploop).resize((512,512)))
        imageio.mimsave('figures/gif_dir/'+timestr+'/zslicegif_epoch_'+str(epoch)+'.gif', images, fps=fps)
        
    def make_gif_slices(self, vol, name='test', timestr=None, length=None):
        images = []
        gifdir = 'figures/gif_dir/'
        if timestr is not None:

            if not os.path.isdir(gifdir+timestr):
                os.mkdir(gifdir+timestr)
            gifdir += timestr+'/'
        print('Saving to ', gifdir)
        if length is None:
            length = len(nbody.redshift_bins)
        for i in xrange(length):
            plooop = (cm.gist_earth((vol[i,:,:]+1)/2)*255).astype('uint8')
            images.append(Image.fromarray(plooop).resize((512,512)))            
        imageio.mimsave(gifdir+name+'.gif', images, fps=2)

    def plot_gradnorms(self, timestring, save=False):
        filepath = self.base_path + '/results/' + timestring
        gen_grad_norms = np.loadtxt(filepath+'/generator_grad_norm.txt')
        disc_grad_norms = np.loadtxt(filepath+'/discriminator_grad_norm.txt')
        print(gen_grad_norms.shape)
        plt.figure()
        plt.plot(np.arange(len(gen_grad_norms)), gen_grad_norms)
        plt.title('Generator gradient norms')
        plt.xlabel('Batch Iteration')
        plt.ylabel('Gradient Norm')
        plt.show()

        plt.figure()
        plt.plot(np.arange(len(disc_grad_norms)), disc_grad_norms)
        plt.title('Discriminator gradient norms')
        plt.xlabel('Batch Iteration')
        plt.ylabel('Gradient Norm')
        if save:
            plt.savefig('figures/gif_dir/'+timestring+'/grad_norms.pdf', bbox_inches='tight')
        plt.show()

        
    def plot_losses(self, timestring, save=False):
        filepath = self.base_path + '/results/' + timestring
        gen_losses = np.loadtxt(filepath+'/lossG.txt')
        disc_losses = np.loadtxt(filepath+'/lossD.txt')
        plt.figure()
        plt.plot(np.arange(len(gen_losses)), gen_losses, marker='.')
        plt.title('Generator')
        plt.xlabel('Batch Iteration')
        plt.ylabel('Loss')
        plt.show()

        plt.figure()
        plt.plot(np.arange(len(disc_losses)), disc_losses, marker='.')
        plt.title('Discriminator gradient norms')
        plt.xlabel('Batch Iteration')
        plt.ylabel('Loss')
        if save:
            plt.savefig('figures/gif_dir/'+timestring+'/gen_disc_losses.pdf', bbox_inches='tight')
        plt.show()
    
    def plot_powerspectra(self, genpks=[], genkbins=[], labels=[],realpk=None, realkbins=None, timestr=None, z=None):
        title = 'Comparison of Power Spectra with 1$\\sigma$ Shaded Regions'
        colors = ['darkslategrey', 'royalblue','m', 'maroon']
        plt.figure(figsize=(8,6))
        if z is not None:
            title += ', z='+str(z)
#         plt.title(title)
        
        if realpk is not None:
            plt.fill_between(realkbins, np.percentile(realpk, 16, axis=0), np.percentile(realpk, 84, axis\
=0), facecolor='green', alpha=0.4)
            plt.plot(realkbins, np.median(realpk, axis=0), label='nbody', color='forestgreen', marker='.')
            plt.plot(realkbins, np.percentile(realpk, 16, axis=0), color='forestgreen')
            plt.plot(realkbins, np.percentile(realpk, 84, axis=0), color='forestgreen')
        
        if len(genpks)>0:
            for i, genpk in enumerate(genpks):

                plt.fill_between(genkbins[i], np.percentile(genpk, 16, axis=0), np.percentile(genpk, 84, axis=0), alpha=0.2, color=colors[i])
                plt.plot(genkbins[i], np.median(genpk, axis=0), label=labels[i], marker='.', color=colors[i])
                plt.plot(genkbins[i], np.percentile(genpk, 16, axis=0), linewidth=0.75, color=colors[i])
                plt.plot(genkbins[i],  np.percentile(genpk, 84, axis=0), linewidth=0.75, color=colors[i])

        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('k (h $Mpc^{-1}$)', fontsize=14)
        plt.ylabel('P(k)', fontsize=14)
        if timestr is not None:
            plt.savefig('figures/gif_dir/'+timestr+'/power_spectra.pdf', bbox_inches='tight')
        plt.show()
            
            
    def discriminator_rejection_sampling(self, generator, discriminator, pdict, batch_size=32, eps=0.01, gamma=0, n_samp=0, n_conditional=0):
        
        n = 0
        counter = 0
        gen_samps = []
        while n<n_samp:
            z = torch.randn(batch_size, pdict['latent_dim']+n_conditional, 1, 1, 1).float()
            gensamps = generator(z)
            discriminator_outputs = discriminator(gensamps).detach().numpy()
            disc_logit_outputs = -np.log((1./discriminator_outputs)-1)
#             print('logit out:', disc_logit_outputs)
            dm_star = np.max(disc_logit_outputs)
            fs = disc_logit_outputs - dm_star - np.log(1-np.exp(disc_logit_outputs-dm_star-eps))-gamma
            acceptance_probs = 1./(1+np.exp(-fs))
            print('acceptance probs:', acceptance_probs)
            rand = np.random.uniform(size=len(acceptance_probs))
#             print('rand:', rand)
            accept = rand<acceptance_probs
            print('accept:', accept)
            if np.sum(accept)>0:
                gensamps = gensamps.detach().numpy()[accept]
                
                for samp in gensamps:
                    gen_samps.append(samp)
                
                n += np.sum(accept)
                print('n + '+str(np.sum(accept)), 'n='+str(n))
            
            counter += 1
            if counter > 100:
                print('counter overload!!!')
                return
        return np.array(gen_samps)
            
            
