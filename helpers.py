import os
import time
import argparse
import h5py
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
import Pk_library as PKL
from models import *
import matplotlib
if sys.platform=='darwin':
    base_dir = '/Users/richardfeder/Documents/caltech/gan_work/results/'
    matplotlib.use('tkAgg')
elif sys.platform=='linux2':
    #base_dir = '/home1/06224/rfederst/gan-work/results/'
    base_dir = '/work/06224/rfederst/maverick2/results/'
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

nparam_dict = dict({'2d_gaussian':5, '1d_gaussian':2, 'bernoulli':2, 'ring':2, 'grid':2})
outparam_dict = dict({'2d_gaussian':2, '1d_gaussian':1, 'bernoulli':1, 'ring':2, 'grid':2})

Device = 'cpu'

mu_range = np.linspace(-1.0, 1.0, 8)
sig_range = np.linspace(0.1, 2.0, 8)

def setup_result_directories():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    new_dir, frame_dir = create_directories(timestr)
    fake_dir = frame_dir+'/fake'
    os.makedirs(fake_dir)
    return timestr, new_dir, frame_dir, fake_dir

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

def make_size_array(imgsize, n_powers=None):
    if n_powers is None:
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
            images.append(imageio.imread(dir+'/'+file).resize((512,512)))
        imageio.mimsave(dir+'/movie.gif', images, fps=3)
    else:
        print('No images to save for make_gif().')

def sample_noise(bs, d):
    z = torch.randn(bs, d).float()
    return Variable(z.to(Device), requires_grad=True)


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


# for hyperparameter sweep tests
def grf_hparam_tests(param_name, vals, imagesize=128, nepochs=20000, trainsize=0):
    command_list = []
    base_command = 'python dcgan_powerspec.py --ndf=32 --ngf=32 --cuda --imageSize='+str(imagesize)
    base_command += ' --n_epochs='+str(nepochs)+' --trainSize='+str(trainsize)
    for val in vals:
        command_list.append(base_command + ' --'+param_name+'='+str(val))
        
    for command in command_list:
        print(command)
        os.system(command)
    return command_list

def inverse_loglike_transform(s, a=4):
    im = a*(s+1)/(1-s)
    return im
    
def loglike_transform(x, a=5):
    return (2*x/(x+a)) - 1

def loglike_transform_2(x, k=5.):
    return (1/float(k))*np.log10(x+1)

def inverse_loglike_transform_2(s, k=5.):
    return 10**(float(k)*s)-1

def blockwise_average_3D(A,S):    
    # A is the 3D input array
    # S is the blocksize on which averaging is to be performed
    m,n,r = np.array(A.shape)//S
    return A.reshape(m,S[0],n,S[1],r,S[2]).mean((1,3,5))


def partition_cube(sim, length, cubedim, sim_boxes, cparam_list=None, z=None, loglike_a=None):
    for x in xrange(int(length/cubedim)):
        for y in xrange(int(length/cubedim)):
            for r in xrange(int(length/cubedim)):
                if loglike_a is None:
                    sim_boxes.append(sim[x*cubedim:(x+1)*cubedim, y*cubedim:(y+1)*cubedim, r*cubedim:(r+1)*cubedim])
                else:
                    sim_boxes.append(loglike_transform(sim[x*cubedim:(x+1)*cubedim, y*cubedim:(y+1)*cubedim, r*cubedim:(r+1)*cubedim], a=loglike_a))
                
                # maybe construct cparam list outside of this function
                if cparam_list is not None:
                    cparam_list.append(z)
    if cparam_list is not None:
        return sim_boxes, cparam_list
    else:
        return sim_boxes

def load_in_simulations(opt):
    nsims = opt.nsims
    sim_boxes = []
    length = 512

    # generalized cond
    #if opt.n_cond_params > 0:
    #    cparam_list = []
        # TODO
    #    return np.array(sim_boxes), np.array(cparam_list)

    if opt.redshift_code:
        zlist = [] # array storing the redshift slice of a given volume in sim_boxes                                    
        for i in xrange(nsims):
            with h5py.File(opt.base_path + 'n512_512Mpc_cosmo1_seed'+str(i+1)+'_gridpart.h5', 'r') as ofile:
                for j, idx in enumerate(opt.redshift_idxs):
                    sim = ofile["%3.3d"%(idx)][()].copy()
                    sim_boxes, zlist = partition_cube(sim, length, opt.cubedim, sim_boxes, cparam_list=zlist, z=opt.redshift_bins[j], loglike_a=opt.loglike_a)
        return np.array(sim_boxes), np.array(zlist)
    else:
        with h5py.File(opt.base_path + opt.file_name, 'r') as ofile:
            for i in xrange(nsims):
                sim = ofile['seed'+str(i+1)][()].copy()
                sim_boxes = partition_cube(sim, length, opt.cubedim, sim_boxes, loglike_a=opt.loglike_a)

        return np.array(sim_boxes)


def get_parsed_arguments(dattype):

    parser = argparse.ArgumentParser()
    parser.add_argument('--trainSize', type=int, default=0, help='size of training dataset, if 0 then use unlimited samples')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--latent_dim', type=int, default=200, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--ndf', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--lr_g', type=float, default=0.0025, help='learning rate of generator, default=0.0025')
    parser.add_argument('--lr_d', type=float, default=0.00001, help='learning rate of discriminator, default=0.00002')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=4, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--wgan', type=bool, default=False, help='use Wasserstein GAN loss/training')
    parser.add_argument('--grad_lam', type=float, default=1., help='coefficient in gradient peanlty term')

    if dattype=='nbody':
        parser.add_argument('--base_path', default='/work/06147/pberger/maverick2/gadget_runs/cosmo1/')
        parser.add_argument('--file_name', default='n512_512Mpc_cosmo1_z0_gridpart.h5')
        parser.add_argument('--datadim', type=int, default=3, help='2 to train on slices, 3 to train on volumes') # not currently functional for 2d   
        parser.add_argument('--loglike_a', type=float, default=4., help='scaling parameter in loglike transformation of data')
        parser.add_argument('--redshift_code', type=bool, default=False, help='determines whether redshift conditional information used for cGAN')
        parser.add_argument('--n_genstep', type=int, default=1, help='number of generator steps for each discriminator step')
        parser.add_argument('--nsims', type=int, default=32, help='number of simulation boxes to get samples from')
        parser.add_argument('--cubedim', type=int, default=128, help='the height / width of the input image to network')
        parser.add_argument('--lcdm', type=int, default=0, help='number of conditional parameters to use for training. lcdm=1 --> Omega_m, lcdm=2 --> Omega_m + Sigma8')
    elif dattype=='grf':
        parser.add_argument('--alpha', type=float, default=-2.0, help='Slope of power law for gaussian random field')

    parser.add_argument('--info', type=bool, default=False, help='determines whether infoGAN is employed')
    parser.add_argument('--get_optimal_discriminator', type=bool, default=False, help='if True, train discriminator for some number of epochs until its loss is minimized')
    parser.add_argument('--disc_only_epochs', type=int, default=0, help='Number of epochs to train the discriminator after fixing generator, default=0')
    
    return parser.parse_known_args()[0]

def compute_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def calc_gradient_penalty(netD, opt, real_data, fake_data, realz_feat=None, fakez_feat=None):
    alpha0 = torch.rand(opt.batchSize, 1)
    alpha = alpha0.expand(opt.batchSize, real_data.nelement()/opt.batchSize).view(opt.batchSize, 1, opt.cubedim, opt.cubedim, opt.cubedim)
    alpha = alpha.cuda() if opt.cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()
        alpha0 = alpha0.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    if opt.redshift_code: # interpolate between redshifts and use for netD
        zfeature_interpolates = alpha0 * realz_feat + ((1 - alpha0) * fakez_feat)
        if opt.cuda:
            zfeature_interpolates = zfeature_interpolates.cuda()
        zfeature_interpolates = Variable(zfeature_interpolates, requires_grad=True)
        disc_interpolates = netD(interpolates, zfeatures=zfeature_interpolates)
    else:
        disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if opt.cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.grad_lam
    return gradient_penalty


def make_feature_maps(array_vals, featureshape, device):
    feature_maps = torch.from_numpy(np.array([np.full(featureshape, val) for val in array_vals])).to(device)
    return torch.unsqueeze(feature_maps, 1).float()



