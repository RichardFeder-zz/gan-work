import matplotlib
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import numpy as np
from IPython.display import Image
from helpers import *
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        self.ax.set_title('use scroll wheel to navigate images')
    
        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = rows/2

        self.im = ax.imshow(self.X[self.ind,:, :])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 1:
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

def view_sim_2d_slices(sim):
        fig, ax = plt.subplots(1, 1)
        tracker = IndexTracker(ax, sim)
        fig.canvas.mpl_connect('button_press_event', tracker.onscroll)
        plt.show()

def make_gif_slices(vol, name='test', timestr=None, length=None, save=False):
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
    if save:
        imageio.mimsave(gifdir+name+'.gif', images, fps=2)
    return images

def plot_diag(base_path, timestring, mode='grad_norm', epoch=None, save=False, show=True, alpha=None, show_gen=True, show_disc=True, marker=None):
    filepath = base_path + '/results/' + timestring
    if mode=='grad_norm':
        gname = 'generator_grad_norm.txt'
        dname = 'discriminator_grad_norm.txt'
        if epoch is not None:
            gname = gname[:-4]+'_epoch_'+str(epoch)+gname[-4:]
            dname = dname[:-4]+'_epoch_'+str(epoch)+dname[-4:]
        ylabel= 'Gradient Norm'
    elif mode=='loss':
        gname = 'lossG.txt'
        dname = 'lossD.txt'
        ylabel = 'Loss'
    elif mode=='loss_epoch':
        gname = 'generator_loss_epoch_'+str(epoch)+'.txt'
        dname = 'discriminator_loss_epoch_'+str(epoch)+'.txt'
        ylabel = 'Loss'
    if alpha is None:
        alpha = 0.5
    gen = np.loadtxt(filepath+'/'+gname)
    disc = np.loadtxt(filepath+'/'+dname)
    
    f = plt.figure()
    if show_gen:
        plt.plot(np.arange(len(gen)), gen, label='Generator', alpha=alpha, marker=marker)
    if show_disc:
        plt.plot(np.arange(len(disc)), disc, label='Discriminator', alpha=alpha, marker=marker)
    plt.xlabel('Batch Iteration')
    plt.ylabel(ylabel)
    plt.legend()
    if save:
        plt.savefig('figures/gif_dir/'+timestring+'/'+mode+'.pdf', bbox_inches='tight')
    if show:
        plt.show()
    return f

def plot_bispectra(k1, k2, genbks=[], thetabins=[], labels=[], realbk=None, realthetabins=[],timestr=None, z=None, title=None):
    fig = plt.figure(figsize=(8,6))
    colors = ['darkslategrey', 'royalblue','m', 'maroon']
    if realbk is not None:
        plt.fill_between(realthetabins, np.percentile(realbk, 16, axis=0), np.percentile(realbk, 84, axis=0), \
facecolor='green', alpha=0.4)
        plt.plot(realthetabins, np.median(realbk, axis=0), label='GADGET-2', color='forestgreen', marker='.')
        plt.plot(realthetabins, np.percentile(realbk, 16, axis=0), color='forestgreen')
        plt.plot(realthetabins, np.percentile(realbk, 84, axis=0), color='forestgreen')

    if len(genbks)>0:
        for i, genbk in enumerate(genbks):
            plt.fill_between(thetabins[i], np.percentile(genbk, 16, axis=0), np.percentile(genbk, 84, axis=0),\
 alpha=0.2, color=colors[i])
            plt.plot(thetabins[i], np.median(genbk, axis=0), label=labels[i], marker='.', color=colors[i])
            plt.plot(thetabins[i], np.percentile(genbk, 16, axis=0), linewidth=0.75, color=colors[i])
            plt.plot(thetabins[i],  np.percentile(genbk, 84, axis=0), linewidth=0.75, color=colors[i])


    plt.legend(loc=1, fontsize=14, frameon=False)
    plt.xlabel('$\\theta$ [radian]', fontsize=14)
    plt.ylabel('$B(k)$ $[h^{-6} Mpc^6]$', fontsize=14)
    plt.yscale('log')
    plt.text(0.1, 1.1e7,'$k_1=$'+str(k1), fontsize=16)
    plt.text(0.1, 8e6,'$k_2=$'+str(k2), fontsize=16)
    plt.ylim(1.5e5, 1.5e7)

    if timestr is not None:
        plt.savefig('figures/gif_dir/'+timestr+'/bispectra_'+str(k1)+'_'+str(k2)+'.pdf', bbox_inches='tight')
    plt.show()

    return fig

def plot_powerspectra(genpks=[], genkbins=[], labels=[],realpk=None, realkbins=None, frac=False, timestr=None, z=None, title=None, colors=None):

    if title is None:
        title = 'Comparison of power spectra with 1$\\sigma$ shaded regions'
    if colors is None:
        colors = ['darkslategrey', 'royalblue','m', 'maroon']
    
    fig, (ax) = plt.subplots(1,1, figsize=[8,6])
                                                                                                     
    if z is not None:
        title += ', z='+str(z)

    if realpk is not None:
        ax.fill_between(realkbins, np.percentile(realpk, 16, axis=0), np.percentile(realpk, 84, axis=0), facecolor='green', alpha=\
0.4)
        ax.plot(realkbins, np.median(realpk, axis=0), label='GADGET-2', color='forestgreen', marker='.')
        ax.plot(realkbins, np.percentile(realpk, 16, axis=0), color='forestgreen')
        ax.plot(realkbins, np.percentile(realpk, 84, axis=0), color='forestgreen')

    if len(genpks)>0:
        print(len(genpks))
        for i, genpk in enumerate(genpks):
            print(colors[i])
            print(labels[i])
            ax.fill_between(genkbins[i], np.percentile(genpk, 16, axis=0), np.percentile(genpk, 84, axis=0), alpha=0.2, color=colors[i])
            ax.plot(genkbins[i], np.median(genpk, axis=0), label=labels[i], marker='.', color=colors[i])
            ax.plot(genkbins[i], np.percentile(genpk, 16, axis=0), linewidth=1., color=colors[i])
            ax.plot(genkbins[i],  np.percentile(genpk, 84, axis=0), linewidth=1., color=colors[i])

    ax.legend(fontsize=14, frameon=False)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlabel('k [h $Mpc^{-1}$]', fontsize=16)
    plt.ylabel('P(k) [$h^{-3}$$Mpc^3$]', fontsize=16)
    if timestr is not None:
        plt.savefig('figures/gif_dir/'+timestr+'/power_spectra.pdf', bbox_inches='tight')
    
    
    if frac:
        axins = inset_axes(ax, width=2.5, height=1.5, loc=2, bbox_to_anchor=(130.0, 200.5, 0.5, 0.5))

        axins.set(title='Fractional deviation', ylim=(1e-3, 2e0), xscale='log', yscale='log')
        axins.set_ylabel('$P_{gen}(k)/P_{real}(k)$-1', fontsize=12)
        for i, genpk in enumerate(genpks):
        
            frac_diff_pk = np.abs((np.median(genpk, axis=0)/np.median(realpk, axis=0))-1)
            frac_diff_pk2 = np.abs((np.mean(genpk, axis=0)/np.mean(realpk, axis=0))-1)
            
            axins.plot(realkbins, frac_diff_pk, marker='.', label='Median', color='darkslategrey', linestyle='solid')
            axins.plot(realkbins, frac_diff_pk2, marker='.', label='Mean', color='darkslategrey', linestyle='dashed')
            
        axins.legend(loc=3, frameon=False)

    plt.tight_layout()
    plt.show()

    return fig

def plot_voxel_pdf(real_vols=None, gen_vols=None, nbins=49, mode='scaled', timestr=None, epoch=None, gen_vols2=None, reallabel='GADGET-2', genlabel='GAN', gen2label=None, show=True):
    f = plt.figure(figsize=(8,6))
    if mode=='scaled':
        bins = np.linspace(-1, 1, 50)
    else:
        bins = 10**(np.linspace(-4, 4, 50))
    midbins = 0.5*(bins[1:]+bins[:-1])
        
    if real_vols is not None:
            
        voxel_hists = np.array([np.histogram(real_vols[i], bins=bins)[0] for i in range(real_vols.shape[0])])
        print(np.mean(voxel_hists, axis=0).shape)
        plt.errorbar(midbins, np.mean(voxel_hists, axis=0)/real_vols.shape[0], yerr=np.std(voxel_hists, axis=0)/real_vols.shape[0], label=reallabel, color='g')
        maxval = np.max(real_vols[0])
    if gen_vols is not None:
        if real_vols is not None:
            binz = bins
        else:
                binz = nbins
        gen_voxel_hists = np.array([np.histogram(gen_vols[i], bins=bins)[0] for i in range(gen_vols.shape[0])])
        plt.errorbar(midbins, np.mean(gen_voxel_hists, axis=0)/gen_vols.shape[0], yerr=np.std(gen_voxel_hists, axis=0)/gen_vols.shape[0], label=genlabel, color='b')
        maxval = np.max(gen_vols[0])
        if gen_vols2 is not None:
            gen2_voxel_hists = np.array([np.histogram(gen_vols2[i], bins=bins)[0] for i in range(gen_vols2.shape[0])])
            plt.errorbar(midbins, np.mean(gen2_voxel_hists, axis=0)/gen_vols2.shape[0], yerr=np.std(gen2_voxel_hists, axis=0)/gen_vols2.shape[0], label=gen2label, color='b')

    plt.yscale('log')
    if mode=='scaled':
        plt.xlabel('Scaled Density $s(x)$', fontsize=14)
    else:
        plt.xlabel('Matter Density $x$', fontsize=14)
        plt.xscale('log')

    plt.legend(fontsize=14, frameon=False)
    plt.ylabel('Normalized Counts', fontsize=14)
    if timestr is not None:
        plt.savefig('figures/gif_dir/'+timestr+'/voxel_pdf_epoch'+str(epoch)+'.pdf', bbox_inches='tight'\
)
    if show:
        plt.show()
        
    return f


def plot_corr_cov_matrices(real, gen, mode='cov', \
                           real_title = 'GADGET-2 Simulations', \
                           gen_title='GAN Samples', \
                           vmin=None, vmax=None, show=True):
    if mode=='cov':
        ylabel='Covariance'
        ylabel2 = 'Fractional Difference'
        diff = np.abs((real - gen)) / real
        imreal = np.log10(real)
        imgen = np.log10(gen)
        if vmax is None:
            vmax = 0.5
        vmin = None
    else:
        ylabel = 'Correlation'
        ylabel2 = 'Difference'
        diff = real - gen
        imreal = real
        imgen = gen
        if vmin is None:
            vmin = 0.5
        if vmax is None:
            vmax = 0.2
        
    f = plt.figure(figsize=(15, 4))
    ax = plt.subplot(1,3,1)
    plt.title(real_title, fontsize=14)

    plt.imshow(imreal, vmin=vmin)
    plt.colorbar()
    plt.tick_params(axis='both', bottom=False, left=False)
    ax.set_yticklabels([])
    ax.set_xticklabels([''])
    ax.set_ylabel(ylabel, fontsize=30)
    ax = plt.subplot(1,3,2)
    plt.title(gen_title, fontsize=14)
    plt.imshow(imgen, vmin=vmin)
    plt.colorbar()
    plt.tick_params(axis='both', bottom=False, left=False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax = plt.subplot(1,3,3)
    plt.title(ylabel2, fontsize=14)
    
    plt.imshow(diff, vmax=vmax)
    plt.colorbar()
    plt.tick_params(axis='both', bottom=False, left=False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return f


def plot_comp_resources(timearray, directory):

    labels = ['Sample Drawing', 'Generator', 'Discriminator + Loss', 'Gradient', 'Backprop/Update']
    timesum = np.sum(timearray, axis=1)

    plt.figure()
    plt.pie(timesum, labels=labels, autopct='%1.1f%%', shadow=True)
    plt.savefig(directory+'/comp_resources.png', bbox_inches='tight')
    plt.close()

def plot_info_iterations(lossI_vals, directory):
    n = len(lossI_vals)
    np.savetxt(directory+'/lossI.txt', lossI_vals)
    plt.figure()
    plt.scatter(np.arange(n), lossI_vals, s=2)
    plt.yscale('log')
    plt.ylabel('Mutual Information Loss')
    plt.xlabel('Iteration')
    plt.savefig(directory+'/loss_I.png', bbox_inches='tight')
    plt.close()


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


def plot_filter_powerspec(real, imag, freqs):
    plt.figure()
    plt.errorbar(freq, np.mean(real, axis=0), yerr=np.std(real,axis=0), fmt='.', label='real')
    plt.errorbar(freq, np.mean(imag, axis=0), yerr=np.std(imag,axis=0), fmt='.', label='imaginary')
    plt.legend()
    plt.show()

def plot_cross_spectra(real_xcorr, gen_xcorr, real_gen_xcorr, bins, show=True, reallabel='N-body', genlabel='GAN'):
           
    f = plt.figure(figsize=(8,6))
    plt.plot(bins, bins**3*np.mean(real_xcorr, axis=0), marker='.', label=reallabel+' x '+reallabel, color='forestgreen')
    plt.fill_between(bins, bins**3*np.percentile(real_xcorr, 16, axis=0), bins**3*np.percentile(real_xcorr, 84, axis=0), alpha=0.2, color='forestgreen')
    plt.plot(bins, bins**3*np.mean(gen_xcorr, axis=0), marker='.', label=genlabel+' x '+genlabel, color='royalblue')
    plt.fill_between(bins, bins**3*np.percentile(gen_xcorr, 16, axis=0), bins**3*np.percentile(gen_xcorr, 84, axis=0), alpha=0.2, color='royalblue')
    plt.plot(bins, bins**3*np.mean(real_gen_xcorr, axis=0), marker='.', label=reallabel+' x '+genlabel, color='m')
    plt.fill_between(bins, bins**3*np.percentile(real_gen_xcorr, 16, axis=0), bins**3*np.percentile(real_gen_xcorr, 84, axis=0), alpha=0.2, color='m')
    plt.xscale('log')
    plt.xlim(0.2, 6)
    plt.legend()
    plt.xlabel('$k$ [h $Mpc^{-1}$]', fontsize=16)
    plt.ylabel('$k^3 P(k)$', fontsize=16)
    if show:
        plt.show()
        
    return f

def plot_powerspec_and_field(k, power_interpolation, best_fit, field):
    pspec_size = field.shape[0]*field.shape[1]
    print 'size here is ', pspec_size
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title('Field')
    plt.imshow(field.real, interpolation='none')
    plt.subplot(1,2,2)
    plt.title('Power Spectrum')
    plt.plot(k, best_fit(np.log10(k)), color='g', label=best_fit)
    plt.scatter(k, np.log10(power2DMean(k, power_interpolation, pspec_size)), label='Noise Spec')
    plt.ylabel('log10(Power)')
    plt.xlabel('k')
    plt.xscale('log')
    plt.xlim(0.9*np.min(k), np.max(k)*1.1)
    plt.legend()
    plt.show()
