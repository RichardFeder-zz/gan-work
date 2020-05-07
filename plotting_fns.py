import matplotlib as mpwplol
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

def convert_png_to_gif(n_image, gifdir='figures/frame_dir', name='multiz', fps=2):
    images = []
    for i in range(n_image):
        print(i)
        a = Image.open(gifdir+'/vol_'+str(i)+'.png')
        images.append(a)
    
    imageio.mimsave(gifdir+'/'+name+'.gif', images, fps=fps)

def get_vol_pts(vol, npoints=300000, thresh=-0.95, logl_a=4., cubedim=64):    
    plotpts = []
    
    if npoints < cubedim**3:
        print('sampling random')
        for n in xrange(npoints):
            xyz = np.random.randint(0, 63, size=3)
            plotpts.append([xyz[0], xyz[1], xyz[2], vol[xyz[0],xyz[1],xyz[2]]])
    else:
        print('sampling all points on grid')
        for i in np.arange(cubedim):
            for j in np.arange(cubedim):
                for k in np.arange(cubedim):
                    plotpts.append([i, j, k, vol[i,j,k]])

    plotpts = np.array(plotpts)
    print(len(plotpts[plotpts[:,3]<thresh]))
    plotpts = plotpts[plotpts[:,3]>thresh]
    
    sizes = loglike_transform(inverse_loglike_transform(plotpts[:,3], a=logl_a), a=15.)
    
    return plotpts, sizes


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


def plot_data_scalings(mode='loglike', a_range=[1, 4, 10, 50], eps=None, lin=None, show=True, cmap=None, title='$s(x)=\\frac{2x}{x+a}-1$'):
    colors = cmap
    if cmap is None:
        colors = plt.cm.YlGnBu(np.linspace(0.5,1.0,len(a_range)))

    f = plt.figure()
    if mode=='loglike':
        plt.title(title, fontsize=14)
        for i, a in enumerate(a_range):
            plt.plot(lin, loglike_transform(lin, a=a), label='$\\kappa=$'+str(a), color=colors[i])
    elif mode=='higan':
        plt.title('Log scaling $s(x) = \\frac{\\log_{10} x + \\epsilon}{\\log_{10} x_{max} + \\epsilon}')
        for e in eps:
            plt.plot(log_transform_HI(lin, eps=e), lin, label=str(e))

    plt.legend()
    plt.xscale('log')
    plt.ylabel('Scaled density $c$', fontsize=14)
    plt.xlabel('Matter density $\\widetilde{\\rho}$ [$10^{10}M_{\\odot}$/cell]', fontsize=14)
    if show:
        plt.show()

    return f

def plot_bispectra_rf(k1, k2, genbks=[], thetas=[], labels=[], colors=['royalblue', 'darkgreen', 'darkslategrey'], reallabel='GADGET-2',\
                      realbk=None, realthetabins=[],timestr=None, \
                    z=None, title=None, legend=True, mode='median', ymin=1.5e5, ymax=1.5e7, \
                    realcolor='forestgreen', realfillcolor='limegreen', alpha_real=0.45, alpha_gen=0.1, \
                    reallinewidth=2., genlinewidth=2., legend_fs=14, k_fs=12, label_fs=14, figsize=(8,6), \
                     return_fig_content=False, fig_content_file=None, title_fs=16):
    
    
    if fig_content_file is not None:
        pkl_file = open(fig_content_file, 'rb')
        fig_content = pickle.load(pkl_file)
        pkl_file.close()   
        
        thetas = fig_content['thetas']
        
        median_realbk = fig_content['median_realbk']
        mean_realbk = fig_content['mean_realbk']
        realbk_16 = fig_content['realbk_16']
        realbk_84 = fig_content['realbk_84']
        
        mean_genbks = fig_content['mean_genbks']
        median_genbks = fig_content['median_genbks']

        genbks_16 = fig_content['genbks_16']
        genbks_84 = fig_content['genbks_84']
        
    else:
        
        if realbk is not None:
            median_realbk = np.median(realbk, axis=0)
            mean_realbk = np.mean(realbk, axis=0)
            realbk_16 = np.percentile(realbk, 16, axis=0)
            realbk_84 = np.percentile(realbk, 84, axis=0)
        
        mean_genbks, median_genbks, genbks_16, genbks_84 = [[] for x in range(4)]
        
        if genbks is not None:
            for i, genbk in enumerate(genbks):
                mean_genbks.append(np.mean(genbk, axis=0))
                median_genbks.append(np.median(genbk, axis=0))
                genbks_16.append(np.percentile(genbk, 16, axis=0))
                genbks_84.append(np.percentile(genbk, 84, axis=0))
            
             
    
    f, (ax) = plt.subplots(1, 1, figsize=figsize)                                                                                                     \

    plt.title('$k_1=$'+str(k1)+', $k_2=$'+str(k2), fontsize=title_fs)
    
    if median_realbk is not None:
        plt.fill_between(thetas, realbk_16, realbk_84, facecolor=realfillcolor, alpha=alpha_real)

        if mode=='median':
            plt.plot(thetas, median_realbk, label=reallabel, color=realcolor, marker='.', linewidth=reallinewidth)
        else:
            plt.plot(thetas, mean_realbk, color='forestgreen', marker='.', linestyle='dashed')

        plt.plot(thetas, realbk_16, color=realcolor, linewidth=reallinewidth)
        plt.plot(thetas, realbk_84, color=realcolor, linewidth=reallinewidth)

    if len(mean_genbks)>0:
        for i, genbk in enumerate(mean_genbks):
            plt.fill_between(thetas, genbks_16[i], genbks_84[i], alpha=alpha_gen, color=colors[i])
            if mode=='median':
                plt.plot(thetas, median_genbks[i], label=labels[i], marker='.', color=colors[i], linewidth=genlinewidth)
            else:
                plt.plot(thetas, genbk, marker='.', color=colors[i], linestyle='dashed')

            plt.plot(thetas, genbks_16[i], color=colors[i], linewidth=genlinewidth)
            plt.plot(thetas,  genbks_84[i],  color=colors[i], linewidth=genlinewidth)

    if legend:
        plt.legend(loc=1, fontsize=legend_fs, frameon=False)                                                                                                \


    plt.xlabel('$\\theta$ [radian]', fontsize=label_fs)
    plt.ylabel('$B(\\theta)$ $[h^{-6} Mpc^6]$', fontsize=label_fs)
    plt.yscale('log')

    plt.tick_params(width=1.5,length=3, labelsize=12)
    if ymin is not None:
        plt.ylim(ymin, ymax)


    plt.tight_layout()
    plt.show()

    
    if return_fig_content:
        fig_content = dict({'mean_realbk':mean_realbk, 'median_realbk':median_realbk, 'realbk_16':realbk_16,'realbk_84':realbk_84,'mean_genbks':mean_genbks,\
                           'median_genbks':median_genbks, 'genbks_16':genbks_16,'genbks_84':genbks_84, 'thetas':thetas})
        return f, fig_content
    
    return f


def plot_powerspectra_rf(genpks=[], genkbins=[], labels=[],reallabel='GADGET-2', \
                       realpk=None, realkbins=None, frac=False, timestr=None, z=None, \
                       title=None, colors=None, mode='median', realcolor='forestgreen',\
                       frac_median=True, frac_mean=True, alpha_real=0.6, alpha_gen=0.2,\
                       reallinewidth=1.5, linewidth=1., realfillcolor='limegreen', \
                      inset_legend=True, inset_loc=2, inset_ymin=1e-3, inset_ymax=2.0, legend=True, \
                      return_fig_content=False, fig_content_file=None):

    
    if fig_content_file is not None:

        pkl_file = open(fig_content_file, 'rb')
        fig_content = pickle.load(pkl_file)
        pkl_file.close()

        if realpk is None:
            
            median_realpk = fig_content['median_realpk']
            mean_realpk = fig_content['mean_realpk']
            realpk_16 = fig_content['realpk_16']
            realpk_84 = fig_content['realpk_84']
            realkbins = fig_content['realkbins']
  
        
    
        median_genpks = fig_content['median_genpks']
        mean_genpks = fig_content['mean_genpks']
        genpks_16 = fig_content['genpks_16']
        genpks_84 = fig_content['genpks_84']
        genkbins = fig_content['genkbins']
    
    if title is None:
        title = 'Comparison of power spectra with 1$\\sigma$ shaded regions'
    if colors is None:
        colors = ['darkslategrey', 'royalblue','m', 'maroon']

    fig, (ax) = plt.subplots(1,1, figsize=[8,6])

    if z is not None:
        title += ', z='+str(z)

    if realpk is not None:
        median_realpk = np.median(realpk, axis=0)
        mean_realpk = np.mean(realpk, axis=0)
        realpk_16 = np.percentile(realpk, 16, axis=0)
        realpk_84 = np.percentile(realpk, 84, axis=0)
        
    if median_realpk is not None:
        ax.fill_between(realkbins, realpk_16, realpk_84, facecolor=realfillcolor, alpha=alpha_real)
        if mode=='median':
            ax.plot(realkbins, median_realpk, label=reallabel, color=realcolor, marker='.')
        elif mode=='mean':
            ax.plot(realkbins, mean_realpk, label=reallabel, color=realcolor, marker='.')
        
        ax.plot(realkbins, realpk_16, color=realcolor, linewidth=reallinewidth)
        ax.plot(realkbins, realpk_84, color=realcolor, linewidth=reallinewidth)

    if len(genpks)>0:
        genpks_16, genpks_84, median_genpks, mean_genpks = [], [], [], []
        for i, genpk in enumerate(genpks):
            print('genpk shape is ', genpk.shape)
            genpks_16.append(np.percentile(genpk, 16, axis=0))
            genpks_84.append(np.percentile(genpk, 84, axis=0))
            median_genpks.append(np.median(genpk, axis=0))
            mean_genpks.append(np.mean(genpk, axis=0))

    for i, genpk_16 in enumerate(genpks_16):
        ax.fill_between(genkbins[i], genpk_16, genpks_84[i], alpha=alpha_gen, color=colors[i])

        if mode=='median':
            ax.plot(genkbins[i], median_genpks[i], label=labels[i], marker='.', color=colors[i], linewidth=linewidth)
        elif mode=='mean':
            ax.plot(genkbins[i], mean_genpks[i], label=labels[i], marker='.', color=colors[i], linewidth=linewidth)
        ax.plot(genkbins[i], genpk_16, linewidth=linewidth, color=colors[i])
        ax.plot(genkbins[i],  genpks_84[i], linewidth=linewidth, color=colors[i])

    if legend:
        ax.legend(fontsize=16, frameon=False, loc=1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(which='major', width=2, length=4, labelsize=12)
    ax.tick_params(which='minor', width=1.5, length=3)
    plt.xlabel('k [h $Mpc^{-1}$]', fontsize=18)
    plt.ylabel('P(k) [$h^{-3}$$Mpc^3$]', fontsize=18)
    if timestr is not None:
        plt.savefig('figures/gif_dir/'+timestr+'/power_spectra.pdf', bbox_inches='tight')


    if frac:
        axins = inset_axes(ax, width=3, height=1.8, loc=2, bbox_to_anchor=(130.0, 220.5, 0.5, 0.5))
        axins.set(ylim=(inset_ymin, inset_ymax), xscale='log', yscale='log')
        axins.tick_params(which='both', length=3, labelsize=11)
#         axins.set(title='Mean Fractional Deviation', ylim=(inset_ymin, inset_ymax), xscale='log', yscale='log')
        axins.set_ylabel('$|\overline{P}_{gen}(k)/\overline{P}_{real}(k)$-1|', fontsize=16)
        for i, genpk_16 in enumerate(genpks_16):

            frac_diff_pk = np.abs((median_genpks[i]/median_realpk)-1)
            frac_diff_pk2 = np.abs((mean_genpks[i]/mean_realpk)-1)
            
            if frac_median:
                axins.plot(realkbins, frac_diff_pk, marker='.', label='Median', color='darkslategrey', linestyle='dashed')
            
            if frac_mean:
                axins.plot(realkbins, frac_diff_pk2, marker='.', label=labels[i], color=colors[i], linestyle='solid')
                
            if frac_median or frac_mean:
                if inset_legend: 
                    axins.legend(frameon=False, loc=inset_loc, fontsize=16)

    plt.tight_layout()
    plt.show()
    
    if return_fig_content:
        fig_contents = dict({'realkbins':realkbins, 'genkbins':genkbins, 'median_realpk':median_realpk, 'mean_realpk':mean_realpk, \
                             'realpk_16':realpk_16, 'realpk_84':realpk_84, 'genpks_16':genpks_16, 'genpks_84':genpks_84, 'median_genpks':median_genpks, 'mean_genpks':mean_genpks})
        
        return fig, fig_contents
    
    return fig


def plot_corr_cov_matrices_rf(real=None, gen=None, mode='cov', \
                           real_title = 'GADGET-2 Simulations', \
                           gen_title='GAN Samples', \
                           vmin=None, vmax=None, show=True, kbins=None, return_fig_content=False, fig_content_file=None):
    
    
    if fig_content_file is not None:
        pkl_file = open(fig_content_file, 'rb')
        fig_content = pickle.load(pkl_file)
        pkl_file.close()
        
        gen = fig_content['gen']
        real = fig_content['real']
    
    if mode=='cov':
        ylabel='Covariance'
        ylabel2 = 'Fractional Difference'
        diff = np.abs((real - gen)) / real
        imreal = real
        imgen = gen
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

    tick_spaces = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    tick_labels = np.array(['', '', '', '','','','','', '1.0', '','','','',''])

    f = plt.figure(figsize=(15, 4))
    ax = plt.subplot(1,3,1)
    plt.title(real_title, fontsize=18)
    if mode=='cov':
        plt.imshow(imreal, vmin=vmin,vmax=1.5e7,norm=matplotlib.colors.LogNorm(), extent=[np.log10(kbins[0]), np.log10(kbins[-1]), np.log10(kbins[-1]), np.log10(kbins[0])])
    if mode=='corr':
        plt.imshow(imreal, vmin=vmin, extent=[np.log10(kbins[0]), np.log10(kbins[-1]), np.log10(kbins[-1]), np.log10(kbins[0])])

    plt.colorbar()
    plt.tick_params(axis='both', bottom=True, left=True)


    ax.set_yticks(np.log10(tick_spaces))
    ax.set_yticklabels(tick_labels)
    ax.set_xticks(np.log10(tick_spaces))
    ax.set_xticklabels(tick_labels)
    
    ax.tick_params(which='major', width=1.5, length=4, labelsize=12)


    ax.set_xlabel('$k$ [$h$ Mpc$^{-1}$]', fontsize=18)
    ax.set_ylabel(ylabel, fontsize=30)


    ax = plt.subplot(1,3,2)
    plt.title(gen_title, fontsize=18)
    
    if mode=='cov':
        plt.imshow(imgen,vmin=2, vmax=1.5e7, norm=matplotlib.colors.LogNorm(), extent=[np.log10(kbins[0]), np.log10(kbins[-1]), np.log10(kbins[-1]), np.log10(kbins[0])])
    if mode=='corr':
        plt.imshow(imgen, vmin=vmin, extent=[np.log10(kbins[0]), np.log10(kbins[-1]), np.log10(kbins[-1]), np.log10(kbins[0])])

    plt.colorbar()
    plt.tick_params(axis='both', bottom=True, left=True)

    ax.set_yticks(np.log10(tick_spaces))
    ax.set_yticklabels(tick_labels)
    ax.set_xticks(np.log10(tick_spaces))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('$k$ [$h$ Mpc$^{-1}$]', fontsize=18)
    ax.tick_params(which='major', width=1.5, length=4, labelsize=12)

    ax = plt.subplot(1,3,3)

    plt.title(ylabel2, fontsize=18)

    plt.imshow(diff, vmax=vmax, extent=[np.log10(kbins[0]), np.log10(kbins[-1]), np.log10(kbins[-1]), np.log10(kbins[0])])
    plt.colorbar()
    plt.tick_params(axis='both', bottom=True, left=True)
    ax.tick_params(which='major', width=1.5, length=4, labelsize=12)

    ax.set_yticks(np.log10(tick_spaces))
    ax.set_yticklabels(tick_labels)
    ax.set_xticks(np.log10(tick_spaces))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('$k$ [$h$ Mpc$^{-1}$]', fontsize=18)

    plt.tight_layout()

    if show:
        plt.show()
        
    if return_fig_content:
        fig_content = dict({})
        return f, fig_content
    
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

def plot_cross_spectra_rf(real_xcorr=None, gen_xcorr=None, real_gen_xcorr=None, bins=None, show=True, reallabel='N-body', genlabels=['GAN'], \
                      linewidth=1.5, return_fig_content=False, fig_content_file=None, gencolors=['C0', 'C3']):

    
    if fig_content_file is not None:
        pkl_file = open(fig_content_file, 'rb')
        fig_content = pickle.load(pkl_file)
        pkl_file.close()
        
        bins = fig_content['bins']
        mean_real_xc = fig_content['mean_real_xc']
        real_xc_16 = fig_content['real_xc_16']
        real_xc_84 = fig_content['real_xc_84']
        
        mean_gen_xcs = fig_content['mean_gen_xcs']
        gen_xcs_16 = fig_content['gen_xcs_16']
        gen_xcs_84 = fig_content['gen_xcs_84']
        
        mean_real_gen_xcs = fig_content['mean_real_gen_xcs']
        real_gen_xcs_16 = fig_content['real_gen_xcs_16']
        real_gen_xcs_84 = fig_content['real_gen_xcs_84']
        
        
    else:
        print('real xcorr has shape ', real_xcorr)
        mean_real_xc = np.mean(real_xcorr, axis=0)
        real_xc_16 = np.percentile(real_xcorr, 16, axis=0)
        real_xc_84 = np.percentile(real_xcorr, 84, axis=0)
        
        mean_gen_xcs, gen_xcs_16, gen_xcs_84, mean_real_gen_xcs, real_gen_xcs_16, real_gen_xcs_84 = [[] for x in range(6)]
        
        for i in range(len(gen_xcorr)):
            mean_gen_xcs.append(np.mean(gen_xcorr[i], axis=0))
            gen_xcs_16.append(np.percentile(gen_xcorr[i], 16, axis=0))
            gen_xcs_84.append(np.percentile(gen_xcorr[i], 84, axis=0))
            
            if real_gen_xcorr is not None:
            
                mean_real_gen_xcs.append(np.mean(real_gen_xcorr[i], axis=0))
                real_gen_xcs_16.append(np.percentile(real_gen_xcorr[i], 16, axis=0))
                real_gen_xcs_84.append(np.percentile(real_gen_xcorr[i], 84, axis=0))
        
        
    f = plt.figure(figsize=(8,6))
    plt.plot(bins, bins**3*mean_real_xc, marker='.', label=reallabel+' x '+reallabel, color='black')
    plt.fill_between(bins, bins**3*real_xc_16, bins**3*real_xc_84, alpha=0.1, color='black')
    plt.plot(bins, bins**3*real_xc_16, linewidth=linewidth, color='black')
    plt.plot(bins,  bins**3*real_xc_84, linewidth=linewidth, color='black')
    

    for i in range(len(mean_gen_xcs)):
        if i==1:
            crosslabel = reallabel+' x '+'GAN'
        else:
            crosslabel = None
            
        plt.plot(bins, bins**3*mean_gen_xcs[i], marker='.', label=genlabels[i]+' x '+genlabels[i], color=gencolors[i])
        plt.fill_between(bins, bins**3*gen_xcs_16[i], bins**3*gen_xcs_84[i], alpha=0.1, color=gencolors[i])
        plt.plot(bins, bins**3*gen_xcs_16[i], linewidth=linewidth, color=gencolors[i])
        plt.plot(bins,  bins**3*gen_xcs_84[i], linewidth=linewidth, color=gencolors[i])

        
        if len(mean_real_gen_xcs) > 0:
            plt.plot(bins, bins**3*mean_real_gen_xcs[i], marker='.', label=crosslabel, color=gencolors[i], linestyle='dashed')
            plt.plot(bins, bins**3*real_gen_xcs_16[i], linewidth=linewidth, color=gencolors[i], linestyle='dashed')
            plt.plot(bins,  bins**3*real_gen_xcs_84[i], linewidth=linewidth, color=gencolors[i], linestyle='dashed')
    
    plt.tick_params(which='major', width=2, length=4, labelsize=12)
    plt.tick_params(which='minor', width=1.5, length=3)
    plt.xlim(0.13, 7)
    plt.xscale('log')
    plt.legend(frameon=False, fontsize=14, loc=2)
    plt.xlabel('$k$ [h $Mpc^{-1}$]', fontsize=18)
    plt.ylabel('$k^3 P(k)$', fontsize=18)
    plt.ylim(-13, 23)
    if show:
        plt.show()
        
    if return_fig_content:
        fig_content = dict({'bins':bins, 'mean_real_xc':mean_real_xc, 'real_xc_16':real_xc_16, 'real_xc_84':real_xc_84, \
                           'mean_gen_xcs':mean_gen_xcs, 'gen_xcs_16':gen_xcs_16, 'gen_xcs_84':gen_xcs_84, \
                           'mean_real_gen_xcs':mean_real_gen_xcs, 'real_gen_xcs_16':real_gen_xcs_16, 'real_gen_xcs_84':real_gen_xcs_84})
        return f, fig_content
    
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

def plot_data_scalings(mode='loglike', a_range=[1, 4, 10, 50], eps=None, lin=None, show=True, cmap=None):
    colors = cmap
    if cmap is None:
        colors = plt.cm.YlGnBu(np.linspace(0.5,1.0,len(a_range)))
        
    f = plt.figure()
    if mode=='loglike':
        plt.title('Loglike scaling $s(x)=\\frac{2x}{x+a}-1$', fontsize=14)
        for i, a in enumerate(a_range):
            plt.plot(lin, loglike_transform(lin, a=a), label='a='+str(a), color=colors[i])
    elif mode=='higan':
        plt.title('Log scaling $s(x) = \\frac{\\log_{10} x + \\epsilon}{\\log_{10} x_{max} + \\epsilon}')
        for e in eps:
            plt.plot(log_transform_HI(lin, eps=e), lin, label=str(e))

    plt.legend()
    plt.xscale('log')
    plt.ylabel('Scaled density s(x)', fontsize=14)
    plt.xlabel('Matter density x', fontsize=14)

    if show:
        plt.show()

    return f

def plot_average_density_hist_rf(avmass_real=None, avmass_gens=None, show=True,\
                                 labels=['GAN'], colors=['darkslategrey', 'royalblue'],\
                                 alpha_real=0.8, alpha_gen=0.8, reallabel='GADGET-2',\
                                 realcolor='forestgreen', median_lines=False, linewidth=1.0,\
                                 return_fig_content=False, fig_content_file=None):
    
    if fig_content_file is not None:

        pkl_file = open(fig_content_file, 'rb')
        fig_content = pickle.load(pkl_file)
        pkl_file.close()

        avmass_real = fig_content['avmass_real']
        avmass_gens = fig_content['avmass_gens']

    bins = None
    f = plt.figure()
    if avmass_real is not None:
        
        
        _, bins, _ = plt.hist(np.log10(avmass_real), bins=np.linspace(-0.48, 0.48, 24), label=reallabel, histtype='step',linewidth=linewidth, color=realcolor,alpha=alpha_real, density=True)
        if median_lines:
            plt.axvline(np.median(np.log10(avmass_real)), color=realcolor, linestyle='dashed')
    if avmass_gens is not None:
        if bins is None:
            bins = 20
        for i, avmass in enumerate(avmass_gens):
            plt.hist(np.log10(avmass), bins=bins, label=labels[i], color=colors[i],linewidth=linewidth, alpha=alpha_gen, density=True, histtype='step')
            if median_lines:
                plt.axvline(np.median(np.log10(avmass)), color=colors[i], linestyle='dashed')
    plt.xlabel('$\\log_{10}(\overline{\\rho}_{subvolume})$', fontsize='large')
    
    plt.legend(frameon=False, fontsize=11)
    plt.ylabel('Probablility density', fontsize='large')

    if show:
        plt.show()
        
    if return_fig_content:
        fig_content = dict({'avmass_real':avmass_real, 'avmass_gens':avmass_gens})
        
        return f, fig_content

    return f

def plot_network_weights(gen, minp=-2.0, maxp=2.0, nbin=50):
    params = []
    for p in gen.parameters():
        params.extend(p.detach().cpu().numpy().ravel())
    f = plt.figure()
    plt.hist(params, bins=np.linspace(minp, maxp, nbin), histtype='step')
    plt.legend()
    plt.yscale('symlog')
    plt.xlabel('Weight value')
    plt.show()
    return f

def plot_voxel_pdf_rf(real_vols=None, gen_vols=None, nbins=49, mode='scaled', \
                    timestr=None, epoch=None, reallabel='GADGET-2', genlabels=['GAN'], realcolor='green', gencolors=['b', 'royalblue'], \
                    show=True, capsize=5, return_fig_content=False, fig_content_file=None, reallinewidth=1, realcapthick=1, \
                     genlinewidth=1., gencapthick=1., legend=True):


    f = plt.figure()
    
    if fig_content_file is not None:
        
        pkl_file = open(fig_content_file, 'rb')
        fig_content = pickle.load(pkl_file)
        pkl_file.close()

        midbins = fig_content['midbins']
        realmeanhist = fig_content['realmeanhist']
        realstdhist = fig_content['realstdhist']
        genmeanhists = fig_content['genmeanhists']
        genstdhists = fig_content['genstdhists']
        
    else:
        if mode=='scaled':
            bins = np.linspace(-1, 1, nbins+1)
        else:
            bins = 10**(np.linspace(-4, 4, nbins+1))

        midbins = 0.5*(bins[1:]+bins[:-1])
    
        if return_fig_content:
            genmeanhists = []
            genstdhists = []
            

        if real_vols is not None:

            voxel_hists = np.array([np.histogram(real_vols[i], bins=bins)[0] for i in range(real_vols.shape[0])])
            realmeanhist = np.mean(voxel_hists, axis=0)/real_vols.shape[0]
            realstdhist = np.std(voxel_hists, axis=0)/real_vols.shape[0]
            
            print('realstdhist:', realstdhist)
        
        if gen_vols is not None:

            if real_vols is not None:
                binz = bins
            else:
                binz = nbins

            for k in range(len(gen_vols)):

                gen_voxel_hists = np.array([np.histogram(gen_vols[k][i], bins=bins)[0] for i in range(gen_vols[k].shape[0])])

                genmeanhist = np.mean(gen_voxel_hists, axis=0)/gen_vols[k].shape[0]
                genstdhist = np.std(gen_voxel_hists, axis=0)/gen_vols[k].shape[0]

                if return_fig_content:
                    genmeanhists.append(genmeanhist)
                    genstdhists.append(genstdhist)
    
    if realmeanhist is not None:
        plt.errorbar(midbins, realmeanhist, yerr=realstdhist, label=reallabel, color=realcolor, capsize=capsize, linewidth=reallinewidth, capthick=realcapthick)
    
            
    if genmeanhists is not None:
        for k, genmeanhist in enumerate(genmeanhists):
            plt.errorbar(midbins, genmeanhist, yerr=genstdhists[k], capthick=gencapthick, linewidth=genlinewidth, label=genlabels[k], color=gencolors[k], capsize=capsize)

    plt.yscale('log')

    if mode=='scaled':
        plt.xlabel('Scaled density $c(x)$', fontsize='large')
    else:
        plt.xlabel('Matter density $\\widetilde{\\rho}$ [$10^{10} M_{\\odot}/$cell]', fontsize='large')
        plt.xscale('log')
    
    if legend:
        plt.legend(fontsize='large', frameon=False, loc=1)
    
    plt.ylabel('Normalized counts', fontsize='large')

    if timestr is not None:
        plt.savefig('figures/gif_dir/'+timestr+'/voxel_pdf_epoch'+str(epoch)+'.pdf', bbox_inches='tight')
    if show:
        plt.show()


    if return_fig_content:
        fig_content = dict({'midbins':midbins, 'realmeanhist':realmeanhist, 'realstdhist':realstdhist, 'genmeanhists':genmeanhists, \
                        'genstdhists':genstdhists})
        
        return f, fig_content

    return f


def plot_volume_multiple_z(gen, minz=0.0, maxz=3.0, nz=8, latent_dim=200, npoints=300000, thresh=-0.95, cmap='winter'):
    pt = [[0,0,0], [0,0,64],[0,64,0],[0,64,64],[64,0,0],[64,0,64],[64,64,0],[64,64,64]]
    pairs = np.array([np.array([pt[0], pt[1]]), np.array([pt[0], pt[2]]), np.array([pt[0], pt[4]]), \
            np.array([pt[3], pt[1]]), np.array([pt[3], pt[2]]), np.array([pt[6], pt[7]]), np.array([pt[5], pt[7]]), \
            np.array([pt[5], pt[4]]), np.array([pt[6], pt[4]]), np.array([pt[7], pt[3]]), np.array([pt[2], pt[6]]), np.array([pt[5], pt[1]])])
    
    cond_zs = np.flip(np.linspace(minz, maxz, nz))
    fixed_z = torch.randn(1, latent_dim+1, 1, 1, 1).float().to(device)
    
    fig = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.0, hspace=0.0)

    for i, z in enumerate(cond_zs):
        ax = fig.add_subplot(2, 4, i+1, projection='3d')
        ax.set_title('$z=$'+str(np.round(z, 3)), fontsize=14)
        print('z=', z)

        fixed_z[:,-1] = z
        samp = gen(fixed_z).cpu().detach().numpy()

        
        ps, sizes = get_vol_pts(samp[0][0], npoints=npoints, thresh=thresh)

        for p in pairs:
            plt.plot(p[:,0], p[:,1], p[:,2], color='k')
        p = ax.scatter3D(ps[:,0], ps[:,1], ps[:,2], c=ps[:,0], s=(sizes+1)**4, cmap=cmap)
        
        plt.xlim(0, 64)
        plt.ylim(0,64)
        ax.grid(False)
        ax.set_axis_off()

    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

    plt.show()
    
    return fig

def plot_pk_multiple_z(gen, minz=0.0, maxz=3.0, nz=60, latent_dim=200):
    cond_zs = np.linspace(minz, maxz, nz)
    pks = []
    fixed_z = torch.randn(1, latent_dim+1, 1, 1, 1).float().to(device)
    norm = mpl.colors.Normalize(vmin=minz, vmax=maxz)

    colormap = matplotlib.cm.ScalarMappable(norm=norm, cmap='rainbow')
    colormap.set_array(cond_zs/maxz)
    colorz = cm.rainbow(cond_zs/maxz)
    
    for i, z in enumerate(cond_zs):
        fixed_z[:,-1] = z
        samp = gen(fixed_z).cpu().detach().numpy()
        
        pk, kbin = compute_power_spectra(samp, cubedim=64)
        pks.append(pk[0])
        
    fig = plt.subplots(figsize=(6, 4))
    
    for i, pk in enumerate(pks):
        plt.plot(kbin, pk, color=colorz[i], linewidth=0.5)
        
    plt.colorbar(colormap)
    
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('k [h $Mpc^{-1}$]', fontsize=16)
    plt.ylabel('P(k) [$h^{-3}$$Mpc^3$]', fontsize=16)
    plt.tight_layout()
    
    plt.show()
    
    return fig

def plot_volume_gif_multiple_z(gen, frame_dir = 'figures/frame_dir', minz=0.0, maxz=3.0, nz=20, latent_dim=200, npoints=300000, thresh=-0.95, cmap='winter', fps=3):
    pt = [[0,0,0], [0,0,64],[0,64,0],[0,64,64],[64,0,0],[64,0,64],[64,64,0],[64,64,64]]
    
    pairs = np.array([np.array([pt[0], pt[1]]), np.array([pt[0], pt[2]]), np.array([pt[0], pt[4]]), \
            np.array([pt[3], pt[1]]), np.array([pt[3], pt[2]]), np.array([pt[6], pt[7]]), np.array([pt[5], pt[7]]), \
            np.array([pt[5], pt[4]]), np.array([pt[6], pt[4]]), np.array([pt[7], pt[3]]), np.array([pt[2], pt[6]]), np.array([pt[5], pt[1]])])

    cond_zs = np.flip(np.linspace(minz, maxz, nz))
    fixed_z = torch.randn(1, latent_dim+1, 1, 1, 1).float().to(device)

    volume_frames = []
    
    for i, z in enumerate(cond_zs):
        
        fig = plt.figure(figsize=(6,6))
        ax = plt.axes(projection='3d')
        ax.set_title('$z=$'+str(np.round(z, 3)), fontsize=14)
        print('z=', z)

        fixed_z[:,-1] = z
        samp = gen(fixed_z).cpu().detach().numpy()

        ps, sizes = get_vol_pts(samp[0][0], npoints=npoints, thresh=thresh)

        for p in pairs:
            plt.plot(p[:,0], p[:,1], p[:,2], color='k')
        p = ax.scatter3D(ps[:,0], ps[:,1], ps[:,2], c=ps[:,0], s=(sizes+1)**4, cmap=cmap)

        plt.xlim(0, 64)
        plt.ylim(0,64)
        ax.grid(False)
        ax.set_axis_off()

        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

        plt.show()
        
        volume_frames.append(fig)
        
    i=0
    for f in volume_frames:
        f.savefig(frame_dir+'/vol_'+str(i)+'.png', dpi=100)
        i+=1

    print('Converting to gif..')
    convert_png_to_gif(nz, gifdir=frame_dir, fps=fps)
    return volume_frames

def plot_gan_real_drs_samps(gen_samps, real_samps, drs_samps, kbins=None, thetas=None, alpha=0.1,\
                            all_in_one=False, colors=['forestgreen', 'royalblue'], ymax=1e5):
    
        
    f = plt.figure(figsize=(12,6))
    plt.subplot(1,3,1)
    plt.title('GAN', fontsize=20)
    if kbins is not None:
        for i, genpk in enumerate(gen_samps):
            plt.plot(kbins, genpk, c='b', alpha=alpha)
    elif thetas is not None:
        for i, genbk in enumerate(gen_samps):
            plt.plot(thetas, genbk, c='b', alpha=alpha)
            
    plt.yscale('log')
    plt.ylim(1e5,1e10)

    if kbins is not None:
        plt.xscale('log')
        plt.ylim(1, ymax)
        plt.xlabel('$k$', fontsize=16)
        plt.ylabel('$P(k)$', fontsize=16)
            
    plt.subplot(1,3,2)
    plt.title('GADGET-2', fontsize=20)
    if kbins is not None:
        for i, realpk in enumerate(real_samps):
            plt.plot(kbins, realpk, c='b', alpha=alpha)
    elif thetas is not None:
        for i, realbk in enumerate(real_samps):
            plt.plot(thetas, realbk, c='b', alpha=alpha)
            
    plt.yscale('log')
    plt.ylim(1e5,1e10)

    if kbins is not None:
        plt.xscale('log')
        plt.ylim(1, ymax)
        plt.xlabel('$k$', fontsize=16)
        plt.ylabel('$P(k)$', fontsize=16)
            
    plt.subplot(1,3,3)
    plt.title('DRS', fontsize=20)
    if kbins is not None:
        for i, drs_pk in enumerate(drs_samps):
            plt.plot(kbins, drs_pk, c='b', alpha=alpha)
    elif thetas is not None:
        for i, drs_bk in enumerate(drs_samps):
            plt.plot(thetas, drs_bk, c='b', alpha=alpha)
            
    plt.yscale('log')
    plt.ylim(1e5,1e10)

    if kbins is not None:
        plt.xscale('log')
        plt.ylim(1, ymax)
        plt.xlabel('$k$', fontsize=16)
        plt.ylabel('$P(k)$', fontsize=16)
    
    plt.tight_layout()
    plt.show()
    
    return f

def plot_redshift_two_panel_rf(reds=[0.0], zstrs=None, return_fig_content=False, fig_content_file=None, colors=None, gfacs=gfacs):

    
    gadget_samples, median_pks = [], []
    
    if fig_content_file is not None:
        
        pkl_file = open(fig_content_file, 'rb')
        fig_content = pickle.load(pkl_file)
        pkl_file.close()   
        gadget_samples = fig_content['gadget_samples']
        
        gfacs = fig_content['gfacs']
        reds = fig_content['reds']
        median_pks = fig_content['median_pks']
        kbin = fig_content['kbin'] 
        
    else:
        gadget_samples = []
        median_pks = []
        for i, red in enumerate(reds):
            print('z = ', reds[i], gfacs[i])
            with open('npoint_stats/median_pk_gadget2_z='+zstrs[i]+'_3_21.pkl', 'r') as f:
                gadget_samples.append(np.array(pickle.load(f)))
                
            ps_sample = np.load('npoint_stats/cGAN_mean_median_pk_z='+zstrs[i]+'_3_23_20.npz')  
            kbin = ps_sample['kbin']
            print('kbin:', kbin)
            median_pks.append(ps_sample['median_pk'])
            
            
                
    
    colors = plt.cm.coolwarm(np.linspace(0,1,len(reds)))
    norm = mpl.colors.Normalize(vmin=0, vmax=3.0)

    colormap = matplotlib.cm.ScalarMappable(norm=norm, cmap='coolwarm')
    colormap.set_array(np.array(reds)/3.0)
    
    f, (ax1, ax2) = plt.subplots(ncols=2, nrows=1,  sharey='row',
                        gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(11, 4))

    
    for i in range(len(reds)):

        if i==0:
            label='cGAN ($z=3$) with\nLinear Evolution'
            nbodylabel = 'N-body'
        else:
            label=None
            nbodylabel = None
            
        ax1.plot(kbin, gadget_samples[i],label=nbodylabel, linestyle='solid', color='grey')
        ax1.plot(kbin, median_pks[-1]*gfacs[i]/1.26114795, label=label, linestyle='dashed', color=colors[i], linewidth=2)
    
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlim(1.4e-1, 6.5)
    ax1.set_ylabel('$P(k) [h^{-3}$ Mpc$^3]$', fontsize=16)
    ax1.set_xlabel('$k [h$ Mpc$^{-1}]$', fontsize=16)
    ax1.legend(loc=3, frameon=False, fontsize=14)

    for i in range(len(reds)):
        if i==0:
            label='cGAN'
            nbodylabel = 'N-body'
        else:
            label=None
            nbodylabel = None

        ax2.plot(kbin, gadget_samples[i],label=nbodylabel, color='grey', linestyle='solid')
        ax2.plot(kbin, median_pks[i], label=label, linestyle='dashed', color=colors[i], linewidth=2)
    
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlabel('$k [h$ Mpc$^{-1}]$', fontsize=16)

    ax2.legend(loc=3, frameon=False, fontsize=14)
    ax2.set_xlim(1.4e-1, 6.5)
#     ax2.set_ylim(5e-1, 3e3)
    plt.tight_layout()

    plt.colorbar(colormap, ax=[ax1, ax2], pad=0.02).set_label(label='Redshift $z$',size=16)
    
    plt.show()
    
    if return_fig_content:
        fig_content = dict({'gadget_samples':gadget_samples,'kbin':kbin, 'median_pks':median_pks, 'gfacs':gfacs, 'reds':reds, 'zstrs':zstrs})
        return f, fig_content

    return f

def plot_latent_z_distribution_rf(gan_norms=None, drs_norms=None, return_fig_content=False, fig_content_file=None, \
                                 linewidth=2, hist_alpha=0.5):
    
    if fig_content_file is not None:
        pkl_file = open(fig_content_file, 'rb')
        print(pkl_file)
        fig_content = pickle.load(pkl_file)
        pkl_file.close()  
        
        gan_norms = fig_content['gan_norms']
        drs_norms = fig_content['drs_norms']
    
    f = plt.figure()
    
    _, bins, _ = plt.hist(gan_norms, bins=25, histtype='stepfilled', alpha=hist_alpha, label='GAN', density=True, linewidth=linewidth, color='royalblue')
    plt.axvline(np.median(gan_norms), linestyle='dashed', color='darkblue', linewidth=linewidth)
    plt.hist(drs_norms, bins=bins, histtype='stepfilled', label='GAN+DRS', linewidth=linewidth, density=True, color='purple', alpha=hist_alpha)
    plt.axvline(np.median(drs_norms), linestyle='dashed', color='purple', linewidth=linewidth)

    plt.legend()
    plt.xlabel('$|\\vec{z}|_2$', fontsize=14)
    plt.ylabel('Probability density', fontsize=14)

    if return_fig_content:
        fig_content = dict({'gan_norms':np.array(gan_norms), 'drs_norms':np.array(drs_norms)})
        
        return f, fig_content
    return f
