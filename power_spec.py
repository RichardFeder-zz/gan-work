import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

def fftIndgen(n):
    a = range(0, n/2+1)
    b = range(1, n/2)
    b.reverse()
    b = [-i for i in b]
    return a + b

def gaussian_random_field(Pk = lambda k : k**-3.0, size = 100):
    def Pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0
        return np.sqrt(Pk(np.sqrt(kx**2 + ky**2)))
    noise = np.fft.fft2(np.random.normal(size = (size, size)))
    amplitude = np.zeros((size,size))
    for i, kx in enumerate(fftIndgen(size)):
        for j, ky in enumerate(fftIndgen(size)):            
            amplitude[i, j] = Pk2(kx, ky)
    return np.fft.ifft2(noise * amplitude)
   
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
    plt.plot(k, line(np.log10(k)), color='g', label=line)
    plt.scatter(k, np.log10(power2DMean(k, power_interpolation, pspec_size)), label='Noise Spec')
    plt.ylabel('log10(Power)')
    plt.xlabel('k')
    plt.xscale('log')
    plt.xlim(0.9*np.min(k), np.max(k)*1.1)
    plt.legend()
    plt.show()