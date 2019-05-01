import sys
import matplotlib
from torch.utils import data
import os
from astropy.io import fits 

if sys.platform=='darwin':
    base_dir = '/Users/richardfeder/Documents/caltech/gan_work/results/'
    matplotlib.use('tkAgg')
elif sys.platform=='linux2':
    base_dir = '/home1/06224/rfederst/gan-work/results/'
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from scipy import interpolate
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class GRFDataset(data.Dataset):
    def __init__(self, root_dir, nsamp=10, transform=None):
        """                                                                                                                                                                                         
        Args:                                                                                                                                                                                       
            root_dir (string): Directory with all the images.                                                                                                                                       
            transform (callable, optional): Optional transform to be applied                                                                                                                        
                on a sample.                                                                                                                                                                                                                                                                                                                                                                           
            nsamp (int):                                                                                                                                                                            
        """

        self.root_dir = root_dir
        self.transform = transform
        self.ngrfs = nsamp

    def __len__(self):
        return self.ngrfs

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'grf_'+str(idx)+'.fits')
        image = fits.open(img_name)
        grf = image[0].data.byteswap().newbyteorder()
        params = image[0].header

        if self.transform:
            nrot = np.random.choice([0, 1, 2, 3])
            grf = np.rot90(grf, nrot).copy()

        sample = {'image': grf, 'params':params}

        return sample


def fftIndgen(n):
    a = range(0, n/2+1)
    b = range(1, n/2)
    b.reverse()
    b = [-i for i in b]
    return a + b

def gaussian_random_field(n_samples, alpha, size = 100):
    def Pk2(kx, ky, alph):
        if kx == 0 and ky == 0:
            return 0.0
        return np.sqrt((np.sqrt(kx**2 + ky**2)**alph))

    grfs = np.zeros((n_samples, size, size))
    
    noise = np.fft.fft2(np.random.normal(size = (n_samples, size, size)))

    if type(alpha)==float:
        amplitude = np.zeros((size,size))
        for i, kx in enumerate(fftIndgen(size)):
            for j, ky in enumerate(fftIndgen(size)):
                amplitude[i, j] = Pk2(kx, ky, alpha)

    else:
        amplitude = np.zeros((len(alpha), size, size))
        #for s in xrange(len(alpha)):
        for i, kx in enumerate(fftIndgen(size)):
            for j, ky in enumerate(fftIndgen(size)):            
                amplitude[:, i, j] = Pk2(kx, ky, alpha)

    grfs = np.fft.ifft2(noise * amplitude, axes=(-2,-1))

    #return grfs.real
    return grfs.real, np.array(noise*amplitude)
   
def power2DMean(k, power_interpolation, size, N=256):
    """ Mean 2D Power works! """
    theta = np.linspace(-np.pi, np.pi, N, False)
    power = np.empty_like(k)
    for i in xrange(k.size):
        kE = np.sin(theta) * k[i]
        kN = np.cos(theta) * k[i]
        power[i] = np.median(power_interpolation.ev(kE, kN) * 4 * np.pi)
        # Median is more stable than the mean here
    return power / size

def func_powerlaw(x, m, c):
    return x**m * c
    
def generate_grf_dataset(nsamp, alpha, size):
    ims = gaussian_random_field(nsamp, alpha, size=size)
    for i, im in enumerate(ims):
        hdr = fits.header.Header()
        hdr['imdim']=size
        hdr['alpha']=alpha
        fits.writeto('data/ps'+str(size)+'/grf_'+str(i)+'.fits', im, hdr, overwrite=True) 
    print('Saved '+str(nsamp)+' GRF realizations.')

def fit_2d_powerspectrum(image, dspace = 1):

    shift = np.fft.fftshift
    nx, ny = image.shape[0], image.shape[1]
    kE = np.fft.fftfreq(nx, d=dspace)
    kN = np.fft.fftfreq(ny, d=dspace)
    k = kN if kN.size < kE.size else kE
    k = k[k > 0]
    k_rad = np.sqrt(kN[:, np.newaxis]**2 + kE[np.newaxis, :]**2)
    
    spec = shift(np.fft.fft2(image))
    pspec = np.abs(spec)**2
    pspec[k_rad == 0.] = 0.
    pspec_size = pspec.size
    
    # shift values so strictly increasing and centered on zero
    kE = shift(np.fft.fftfreq(spec.shape[1]))
    kN = shift(np.fft.fftfreq(spec.shape[0]))

    power_interp = interpolate.RectBivariateSpline(kN, kE, pspec)

    line = np.poly1d(np.polyfit(np.log10(k), np.log10(power2DMean(k, power_interp, pspec_size)), 1))
    
    return k, line, power_interp, pspec_size


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

def generate_grf_dataset(nsamp, alpha, size):
    ims = gaussian_random_field(nsamp, alpha, size=size)
    for i, im in enumerate(ims):
        hdr = fits.header.Header()
        hdr['imdim']=size
        hdr['alpha']=alpha
        fits.writeto('data/ps'+str(size)+'/grf_'+str(i)+'.fits', im, hdr, overwrite=False)
    print('Saved '+str(nsamp)+' GRF realizations.')
