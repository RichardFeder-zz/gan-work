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
from plotting_fns import *
import astropy
from astropy.io import fits
from powerbox import get_power
import Pk_library as PKL
from PIL import Image
import scipy.ndimage
from mpl_toolkits.mplot3d import Axes3D
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

    
class nbody_dataset():
    device = torch.device("cuda:0")
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
        
    def compare_pk_diffz(self, model, pdict, redshift_idxs, nsamp=100, timestr=None, age=False, loglike_a=4.0):
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
                cond = self.redshift_bins[zed_idx]
                if age:
                    cond = (cosmo.age(cond).value)/cosmo.age(0).value
                    print('cond:', cond)
                
                gen_samps = self.get_samples(model, nsamp, pdict, n_conditional=1, c=cond)
                genpks, genks = self.compute_power_spectra(gen_samps, inverse_loglike_a=loglike_a)
                
                allgenpks.append(genpks)
                allgenks.append(genks)
                allrealpks.append(realpks)
                allrealks.append(realks)
                
                self.plot_powerspectra(genpks=[genpks], genkbins=[genks], realpk=realpks, realkbins=realks, timestr=timestr, z=self.redshift_bins[zed_idx], labels=['gan'])
            ofile.close()
        return allgenpks, allgenks, allrealpks, allrealks


    def discriminator_rejection_sampling(self, generator, discriminator, pdict, batch_size=\
32, eps=0.01, gamma=0, n_samp=0, n_conditional=0, dmstar_n_samp=200, ngpu=1, redshift=None)\
:

        n = 0
        counter = 0
        gen_samps = []
        device = torch.device("cuda:0")
        print 'Estimating D_M*...'
        # estimate dm_star                                                                  
        z = torch.randn(dmstar_n_samp, pdict['latent_dim']+n_conditional, 1, 1, 1, device=self.device).float()
        output1shape = (pdict['cubedim']/2, pdict['cubedim']/2, pdict['cubedim']/2)
        print 'output1shape:', output1shape

        if redshift is not None:
            z[:,-1] = redshift

        gensamps = generator(z)
        if redshift is not None:
            discriminator_outs = discriminator(gensamps, cond_features=make_feature_maps(z[\
:,-1].cpu(), output1shape, device)).cpu().detach().numpy()

        else:
            discriminator_outs = discriminator(gensamps).cpu().detach().numpy()
        disc_logit_outs = -np.log((1./discriminator_outs)-1)
        dm_star = np.max(disc_logit_outs)


        while n<n_samp:
            z = torch.randn(batch_size, pdict['latent_dim']+n_conditional, 1, 1, 1, device=\
self.device).float()

            if redshift is not None:
                z[:,-1] = redshift

            gensamps = generator(z)

            if redshift is not None:
                discriminator_outputs = discriminator(gensamps, cond_features=make_feature_maps(z[:,-1].cpu(), output1shape, device)).cpu().detach().numpy()
            else:
                discriminator_outputs = discriminator(gensamps).cpu().detach().numpy()
            disc_logit_outputs = -np.log((1./discriminator_outputs)-1)
            fs = disc_logit_outputs - dm_star - np.log(1-np.exp(disc_logit_outputs-dm_star-\
eps))-gamma
            acceptance_probs = 1./(1+np.exp(-fs))
            rand = np.random.uniform(size=len(acceptance_probs))
            accept = rand<acceptance_probs
            if np.sum(accept)>0:
                gensamps = gensamps.cpu().detach().numpy()[accept]
                for samp in gensamps:
                    gen_samps.append(samp)
                n += np.sum(accept)
                print('n + '+str(np.sum(accept)), 'n='+str(n))

            counter += 1
            if counter > 100:
                print('counter overload!!!')
                return np.array(gen_samps)
        return np.array(gen_samps)



    def get_samples(self, generator, nsamp, pdict, n_conditional=0, c=None, discriminator=None, sigma=1, df=None):
        if df is not None:
            z = sigma*torch.distributions.StudentT(scale=sigma,df=df).rsample(sample_shape=(nsamp, pdict['latent_dim']+n_conditional, 1, 1, 1)).float().to(self.device)
        else:
            z = sigma*torch.randn(nsamp, pdict['latent_dim']+n_conditional, 1, 1, 1, device=self.device).float()
        print('self.device:', self.device)
        if c is not None:
            for i in xrange(n_conditional):
                z[:,-i] = c[i]
            #z[:,-1] = c                                                                    
        gensamps = generator(z)
        if discriminator is not None:
            disc_outputs = discriminator(gensamps)
            return gensamps.detach().numpy(), disc_outputs.detach().numpy()
        return gensamps.cpu().detach().numpy()

    def load_in_sims(self, nsims, loglike_a=None, redshift_idxs=None, idxs=None, ds_factor=1, fac=1):
        
        if idxs is not None:
            simrange = idxs
        else:
            simrange = np.arange(nsims)
        if redshift_idxs is None:
            #with h5py.File(self.data_path+'n512_512Mpc_cosmo1_z0_gridpart.h5', 'r') as ofile:
            #    for i in xrange(nsims):
            ##        sim = ofile['seed'+str(i+1)][()]
            #        self.datasims = partition_cube(sim, self.length, self.cubedim, self.datasims, loglike_a=loglike_a)

            #ofile.close()
            


            for i in simrange:
                print(i)
                with h5py.File(self.data_path + self.name_base + str(i+1)+'_gridpart.h5', 'r') as ofile:
                    sim = ofile["009"][()]
                    self.datasims = partition_cube(sim, self.length, self.cubedim, self.datasims, loglike_a=loglike_a, ds_factor=ds_factor, fac=fac)
                ofile.close()

        else:
            
            for i in simrange:
                with h5py.File(self.data_path + self.name_base + str(i+1)+'_gridpart.h5', 'r') as ofile:
                    for idx in redshift_idxs:
                        print(idx, self.redshift_bins[idx])
                        sim = ofile['%3.3d'%(idx)][()]
                        self.datasims, self.zlist = partition_cube(sim, self.length, self.cubedim, self.datasims, cparam_list=self.zlist, loglike_a=loglike_a, ds_factor=ds_factor, fac=fac)
                ofile.close()
                

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

    def plot_multi_z_vpdfs(self, model=None, pdict=None, zs=None, nsamp=50, timestr=None, age=False, loglike_a=4.0, epoch=None):
        pks, ks = [], []
        zslices = []
        if zs is not None:
            cond = zs
        else:
            cond = self.redshift_bins
            
        if age:
            cond = cosmo.age(cond).value/cosmo.age(0).value

        colormap = matplotlib.cm.jet(np.linspace(1, 0.1, len(cond)))
        plt.figure(figsize=(8,6))
        if model is None:
            plt.title('GADGET N-body Samples')
        else:
            plt.title('Generated Samples')
        for i, c in enumerate(cond):
            if model is not None:
                gen_samps = self.get_samples(model, nsamp, pdict, n_conditional=1, c=c)
                pk, kbins = self.compute_power_spectra(gen_samps, inverse_loglike_a=loglike_a)

            else:
                self.datasims = []
                self.load_in_sims(1, loglike_a=loglike_a, redshift_idxs=[int(self.z_idx_dict[zs[i]])])
                print(len(self.datasims))
                gen_samps = np.array(self.datasims[:nsamp])
                print(gen_samps.shape)
                pk, kbins = self.compute_power_spectra(gen_samps, inverse_loglike_a=loglike_a, unsqueeze=False)
            pks.append(pk)
            ks.append(kbins)
            plt.hist(gen_samps.flatten(), bins=100, label='z='+str(zs[i]), histtype='step', color=colormap[i], normed=True)
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
            plt.plot(ks[i], np.median(pks[i], axis=0), marker='.', c=colormap[i], label='z='+str(zs[i]))
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('$k$', fontsize=14)
        plt.ylabel('P(k)', fontsize=14)
        plt.ylim(2e-1, 2e4)
        plt.legend()
        if timestr is not None:
            plt.savefig('figures/gif_dir/'+timestr+'/multiz_pk_epoch'+str(epoch)+'.pdf', bbox_inches='tight')
        plt.show()
            
    def restore_generator(self, timestring, epoch=None, n_condparam=0, extra_conv_layers=0, discriminator=False):
        print('device:', self.device)
        filepath = self.base_path + '/results/' + timestring
        sizes = np.array([8., 4., 2., 1.])
        print(sizes)
        filen = open(filepath+'/params.txt','r')
        pdict = pickle.load(filen)
        model = DC_Generator3D_simpler(pdict['ngpu'], 1, pdict['latent_dim']+n_condparam, pdict['ngf'], sizes, extra_conv_layers=extra_conv_layers).to(self.device)

        if epoch is None:
            model.load_state_dict(torch.load(filepath+'/netG', map_location=self.device))
        else:
            if discriminator:
                disc_model = DC_Discriminator3D_simpler(pdict['ngpu'], 1, pdict['ndf'], sizes, self.device, n_cond_features=n_condparam).to(self.device)
                disc_model.load_state_dict(torch.load(filepath+'/netD_epoch_'+str(epoch), map_location=self.device))
                disc_model.eval()
            model.load_state_dict(torch.load(filepath+'/netG_epoch_'+str(epoch), map_location=self.device))
        model.eval()
        if discriminator:
            return model, pdict, disc_model
        return model, pdict

            
def find_best_epoch(timestr, nsamp=200, epochs=None, k1 = 0.2, k2 = 0.3, ncond=0, loglike_a = 4.0, extra_conv_layers=0):
    nbody = nbody_dataset(cubedim=64)
    nbody.load_in_sims(2)
    
    if epochs is None:
        epochs = np.arange(0, 40)
    
    top_epochs = []
    
    pks, kbins = nbody.compute_power_spectra(np.array(nbody.datasims[-nsamp:]))

    median_pk = np.median(pks, axis=0)
    upper_sig = np.percentile(pks, 84, axis=0)
    lower_sig = np.percentile(pks, 16, axis=0)
    
    bks, thetas = nbody.compute_matter_bispectrum(np.array(nbody.datasims[-nsamp:]), k1=k1, k2=k2)
    median_bk = np.median(bks, axis=0)
        
    bks2, thetas = nbody.compute_matter_bispectrum(np.array(nbody.datasims[-nsamp:]), k1=0.5, k2=1.0)
    median_bk2 = np.median(bks2, axis=0)

    for ep in epochs:
        print 'epoch '+str(ep)+'..'

        gen, pdict = nbody.restore_generator(timestr, epoch=ep, n_condparam=ncond, extra_conv_layers=extra_conv_layers)
        print pdict
        loglike_a =  pdict['loglike_a']
        gen_samps = nbody.get_samples(gen, nsamp, pdict, n_conditional=ncond)
        print(inverse_loglike_transform(np.mean(gen_samps[gen_samps>np.percentile(gen_samps, 99.9995)]), loglike_a))
        gen_samps *= -1.0
        true_gen_samps = inverse_loglike_transform(gen_samps, loglike_a)
        
        pk, kbin = compute_power_spectra(true_gen_samps, unsqueeze=True)
        
        
        genbks, theta = compute_matter_bispectrum(inverse_loglike_transform(gen_samps[:,0,:,:,:], loglike_a), k1=k1, k2=k2)
        gen_median_bk = np.median(genbks, axis=0)
            
        genbks2, theta = compute_matter_bispectrum(inverse_loglike_transform(gen_samps[:,0,:,:,:], loglike_a), k1=0.5, k2=1.0)
        gen_median_bk2 = np.median(genbks2, axis=0)
        
        plt.figure(figsize=(8,6))
        plt.plot(thetas, np.median(bks, axis=0), marker='.', label='nbody', color='forestgreen')
        plt.fill_between(thetas, np.percentile(bks, 16, axis=0), np.percentile(bks, 84, axis=0), alpha=0.2, color='forestgreen')
        plt.plot(thetas, np.percentile(bks, 16, axis=0), color='forestgreen')
        plt.plot(thetas, np.percentile(bks, 84, axis=0), color='forestgreen')
        
        plt.plot(thetas, np.median(genbks, axis=0), marker='.', label='GAN', color='royalblue')
        plt.fill_between(thetas, np.percentile(genbks, 16, axis=0), np.percentile(genbks, 84, axis=0), alpha=0.2, color='royalblue')
        plt.plot(thetas, np.percentile(genbks, 16, axis=0), color='royalblue')
        plt.plot(thetas, np.percentile(genbks, 84, axis=0), color='royalblue')

        plt.legend()
        plt.xlabel('$\\theta$ (radian)', fontsize=14)
        plt.ylabel('$B(k)$ $(h^{-6} Mpc^6)$', fontsize=14)
        plt.title('Bispectrum ($k_1=$'+str(k1)+', $k_2=$'+str(k2)+') $\pm 1\\sigma$ shaded regions', fontsize=14)
        plt.yscale('log')
        plt.show()
        
        
        plt.figure(figsize=(8,6))
        plt.plot(thetas, np.median(bks2, axis=0), marker='.', label='nbody', color='forestgreen')
        plt.fill_between(thetas, np.percentile(bks2, 16, axis=0), np.percentile(bks2, 84, axis=0), alpha=0.2, color='forestgreen')
        plt.plot(thetas, np.percentile(bks2, 16, axis=0), color='forestgreen')
        plt.plot(thetas, np.percentile(bks2, 84, axis=0), color='forestgreen')
        
        plt.plot(thetas, np.median(genbks2, axis=0), marker='.', label='GAN', color='royalblue')
        plt.fill_between(thetas, np.percentile(genbks2, 16, axis=0), np.percentile(genbks2, 84, axis=0), alpha=0.2, color='royalblue')
        plt.plot(thetas, np.percentile(genbks2, 16, axis=0), color='royalblue')
        plt.plot(thetas, np.percentile(genbks2, 84, axis=0), color='royalblue')
        
        plt.legend()
        plt.xlabel('$\\theta$ (radian)', fontsize=14)
        plt.ylabel('$B(k)$ $(h^{-6} Mpc^6)$', fontsize=14)
        plt.title('Bispectrum ($k_1=$'+str(0.5)+', $k_2=$'+str(1.0)+') $\pm 1\\sigma$ shaded regions', fontsize=14)
        plt.yscale('log')
        plt.show()

        plot_powerspectra(realpk=pks, realkbins=kbins, genpks=[pk], genkbins=[kbin], labels=['epoch '+str(ep)])

        gen_median_pk = np.median(pk, axis=0)
        gen_upper_sig = np.percentile(pk, 84, axis=0)
        gen_lower_sig = np.percentile(pk, 16, axis=0)
        
        frac_err = (median_pk - gen_median_pk)/median_pk
        upper_err = (upper_sig - gen_upper_sig)/upper_sig
        lower_err = (lower_sig - gen_lower_sig)/lower_sig
        bk_err = (median_bk - gen_median_bk)/median_bk
        bk2_err = (median_bk2 - gen_median_bk2)/median_bk2
        
        errs = np.array([np.sum(frac_err[~np.isnan(frac_err)]**2), \
                np.sum(lower_err[~np.isnan(lower_err)]**2), np.sum(bk_err[~np.isnan(bk_err)]**2), \
                       np.sum(bk2_err[~np.isnan(bk2_err)]**2)])
        
        print('errs:', errs)
        print('sum of errs:', np.sum(errs))



def compute_covariance_matrix(kbins, powerspec_1, powerspec_2):
    covariance_matrix = np.zeros((len(kbins), len(kbins)))
    
    mean_1 = np.mean(powerspec_1, axis=0)
    mean_2 = np.mean(powerspec_2, axis=0)
    
    dmean_1 = np.array([p - mean_1 for p in powerspec_1])
    dmean_2 = np.array([p - mean_2 for p in powerspec_2])    
    
    for i in range(len(kbins)):
        for j in range(len(kbins)):
            covariance_matrix[i,j] = (1./(powerspec_1.shape[0]-1))*np.sum([dmean_1[k][i]*dmean_2[k][j] for k in range(dmean_1.shape[0])])
                                             
    return covariance_matrix   


def compute_average_cross_correlation(npairs=100, gen_samples=None, cubedim=64., real_samples=None):
    xcorrs = []
    kbins = 10**(np.linspace(-1, 2, 30))
    for i in xrange(npairs):

        if gen_samples is None:
            idxs = np.random.choice(real_samples.shape[0], 2, replace=False)
            reali = real_samples[idxs[0]]-np.mean(real_samples[idxs[0]])
            realj = real_samples[idxs[1]]-np.mean(real_samples[idxs[1]])

            xc, ks = get_power(deltax=reali, boxlength=cubedim, deltax2=realj, log_bins=True)

        elif real_samples is None:
            idxs = np.random.choice(gen_samples.shape[0], 2, replace=False)
            geni = gen_samples[idxs[0]]-np.mean(gen_samples[idxs[0]])
            genj = gen_samples[idxs[1]]-np.mean(gen_samples[idxs[1]])
            xc, ks = get_power(geni[0], cubedim, deltax2=genj[0], log_bins=True)

        else:
            idxreal = np.random.choice(real_samples.shape[0], 1, replace=False)[0]
            idxgen = np.random.choice(gen_samples.shape[0], 1, replace=False)[0]
            real = real_samples[idxreal]-np.mean(real_samples[idxreal])
            gen = (gen_samples[idxgen]-np.mean(gen_samples[idxgen]))[0]

            xc, ks = get_power(np.array(real), cubedim, deltax2=np.array(gen), log_bins=True)

        xcorrs.append(xc)

    return np.array(xcorrs), ks

def compute_power_spectra(vols, cubedim=64, inverse_loglike_a=None, unsqueeze=False):

    pks, power_kbins = [], []
    if inverse_loglike_a is not None: # for generated data                             \                   

        vols = inverse_loglike_transform(vols, a=inverse_loglike_a)
        
    if unsqueeze:
        vols = vols[:,0,:,:,:] # gets rid of single channel                            \                   


    kbins = 10**(np.linspace(-1, 2, 30))

    for i in xrange(vols.shape[0]):
        pk, bins = get_power(vols[i]-np.mean(vols[i]), cubedim, bins=kbins)

        if np.isnan(pk).all():
            print('NaN for power spectrum')
            continue
        pks.append(pk)

    return np.array(pks), np.array(bins)

def compute_matter_bispectrum(vols, cubedim=64, k1=0.1, k2=0.5):
    thetas = np.linspace(0.0, 2.5, 10)
    bks = []
    for i in xrange(vols.shape[0]):
        with HiddenPrints():
            bis = PKL.Bk(vols[i]-np.mean(vols[i]), float(cubedim), k1, k2, thetas)
            bks.append(bis.B)
    return bks, thetas
