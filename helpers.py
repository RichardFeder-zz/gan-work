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
    base_dir = '/work/06147/pberger/maverick2/results/'
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

nparam_dict = dict({'2d_gaussian':5, '1d_gaussian':2, 'bernoulli':2, 'ring':2, 'grid':2})
outparam_dict = dict({'2d_gaussian':2, '1d_gaussian':1, 'bernoulli':1, 'ring':2, 'grid':2})

Device = 'cpu'

def add_figure_key(fig, fig_list, key=None, key_list=None):
    fig_list.append(fig)
    if key is not None:
        key_list.append(key)
        
    return fig_list, key_list  

def blockwise_average_3D(A,S, fac=1):
    # A is the 3D input array                                                                                       
    # S is the blocksize on which averaging is to be performed                                                      
    m,n,r = np.array(A.shape)//S
    return fac*A.reshape(m,S[0],n,S[1],r,S[2]).mean((1,3,5))


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

def compute_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

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

def draw_training_batch(sim_boxes, sim_idxs, opt):
    choice = np.random.choice(sim_idxs, opt.batchSize, replace=False)
    dat = []
    for c in choice:
        nrots = np.random.choice([0,1,2,3],3)
        # random rotation wrt each plane + random flip                                                              
        if np.random.uniform() > 0.5:
            dat.append(np.fliplr(np.rot90(np.rot90(np.rot90(sim_boxes[c], nrots[0], axes=(0,1)), nrots[1], axes=(1,\
2)), nrots[2], axes=(0,2))))
        else:
            dat.append(np.rot90(np.rot90(np.rot90(sim_boxes[c], nrots[0], axes=(0,1)), nrots[1], axes=(1,2)), nrots\
[2], axes=(0,2)))

    return dat, choice


def filter_nan_pk(kbin, pk):
    mask = ~np.isnan(kbin)
    kbinz = kbin[mask]
    pkz = pk[:,mask]
        
    return kbinz, pkz

def get_parsed_arguments(dattype):

    parser = argparse.ArgumentParser()
    parser.add_argument('--trainSize', type=int, default=0, help='size of training dataset, if 0 then use unlimited\
 samples')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--df', type=int, default=None, help='degrees of freedom in student-t distribution')
    parser.add_argument('--latent_dim', type=int, default=200, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--ndf', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--lr_g', type=float, default=0.0025, help='learning rate of generator, default=0.0025')
    parser.add_argument('--lr_d', type=float, default=0.00001, help='learning rate of discriminator, default=0.0000\
2')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--acgd', type=bool, default=False, help='enables competitve gradient descent optimizer')
    parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=4, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--schedule', type=bool, default=False, help='set to True for learning rate scheduling throughout training')
    parser.add_argument('--step_size', type=int, default=1000, help='if using LR scheduler, it steps every opt.step_size batches')
    parser.add_argument('--gamma', type=float, default=1.0, help='learning rate gets decreased by this factor every opt.step_size batches')
    parser.add_argument('--save_frac', type=int, default=1, help='save one of save_frac models')
    parser.add_argument('--wgan', type=bool, default=False, help='use Wasserstein GAN loss/training')
    parser.add_argument('--grad_lam', type=float, default=1., help='coefficient in gradient peanlty term')
    parser.add_argument('--extra_conv_layers', type=int, default=0, help='number of extra convolutional layers at t\
he end of generator network, default 0')
    parser.add_argument('--cond_scale_fac', type=float, default=1.0, help='scale factor for conditional information')
    if dattype=='nbody':
        parser.add_argument('--base_path', default='/work/06147/pberger/maverick2/gadget_runs/cosmo1/')
        parser.add_argument('--file_name', default='n512_512Mpc_cosmo1_z0_gridpart.h5')
        parser.add_argument('--datadim', type=int, default=3, help='2 to train on slices, 3 to train on volumes') #not currently functional for 2d                                                                                    
        parser.add_argument('--fac', type=float, default=1.0, help='multiply by this factor when downsampling')
        parser.add_argument('--loglike_a', type=float, default=None, help='scaling parameter in loglike transformatio\
n of data')
        parser.add_argument('--log_scaling', type=bool, default=False, help='use log scaling from HIGAN paper')
        parser.add_argument('--piecewise_scaling', type=bool, default=False, help='use piecewise scaling')
        parser.add_argument('--c', type=float, default=100, help='pivot point for piecewise transformation')
        parser.add_argument('--xmax', type=float, default=50000, help='maximum value to pin s(xmax)=1 to in piecewise transformation')
        parser.add_argument('--redshift_code', type=bool, default=False, help='determines whether redshift conditio\
nal information used for cGAN')
        parser.add_argument('--separate_sims', type=bool, default=False, help='reduce overlap of simulations when conditioning on several redshift snapshots')
        parser.add_argument('--age_bins', type=bool, default=False, help='set to True if using fractional times t_z as conditional information rather than redshift z')
        parser.add_argument('--n_genstep', type=int, default=1, help='number of generator steps for each discrimina\
tor step')
        parser.add_argument('--single_z_idx', type=int, default=9, help='redshift index for single redshift runs')
        parser.add_argument('--nseed', type=int, default=1, help='number of seeds to choose from training data')
        parser.add_argument('--nsims', type=int, default=32, help='number of simulation boxes to get samples from')
        parser.add_argument('--n_max_sims', type=int, default=32, help='maximum number of simulations in set, mainly to be used w conditional redshift snapshots')
        parser.add_argument('--cubedim', type=int, default=128, help='the height / width of the input image to netw\
ork')
        parser.add_argument('--ds_factor', type=int, default=1, help='specify if downsampling data for coarser resolution')
        parser.add_argument('--lcdm', type=int, default=0, help='number of conditional parameters to use for training. lcdm=1 --> Omega_m, lcdm=2 --> Omega_m + Sigma8')
    elif dattype=='grf':
        parser.add_argument('--alpha', type=float, default=-2.0, help='Slope of power law for gaussian random field\
')

    parser.add_argument('--info', type=bool, default=False, help='determines whether infoGAN is employed')
    parser.add_argument('--get_optimal_discriminator', type=bool, default=False, help='if True, train discriminator\
 for some number of epochs until its loss is minimized')
    parser.add_argument('--disc_only_epochs', type=int, default=0, help='Number of epochs to train the discriminato\
r after fixing generator, default=0')

    return parser.parse_known_args()[0]

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


def load_in_omegam_s8_sims(opt):
    
    omegas = [0.28724518, 0.3132452, 0.3392452]
    sigma8s = [0.801, 0.829, 0.857]
    omegam_s8_dict = dict({1:[0.801, 0.28724518], 2:[0.801, 0.3132452], 3:[0.801, 0.3392452], \
                           4:[0.829, 0.28724518], 5:[0.829, 0.3132452], 6:[0.829, 0.3392452], \
                           7:[0.857, 0.28724518], 8:[0.857, 0.3132452], 9:[0.857, 0.3392452]})
    
    print('loading in omega/s8 sims..')
    nsims = opt.nsims
    sim_boxes = []
    length = 512
    cparam_list = []
    conds = []
    
    for i in xrange(nsims):
        if opt.lcdm > 0:
            cond = omegam_s8_dict[i+1]
            # these lines get the conditional codes between -1 and 1
            cond[0] -= np.mean(sigma8s)
            cond[0] /= (np.mean(sigma8s)-np.min(sigma8s))
            cond[1] -= np.mean(omegas)
            cond[1] /= (np.mean(omegas)-np.min(omegas))
        else:
            cond = []
    
        if len(opt.redshift_idxs)>0:
            cond.append(0.)
            for j, idx in enumerate(opt.redshift_idxs):
                cond[-1] = opt.redshift_bins[j]
                condoboi = np.array(cond).copy()
                conds.append(condoboi)
        else:
            conds.append(cond)
        with h5py.File(opt.base_path+'n512_512Mpc_cosmo'+str(i+1)+'_gridpart.h5', 'r') as ofile:

            for seed in xrange(opt.nseed):
            
                if len(opt.redshift_idxs)>0:
                    print('redshift time baby')
                    for j, idx in enumerate(opt.redshift_idxs):
                        cond[-1] = opt.redshift_bins[j]
                        print('cond is now', cond)
                        sim = ofile['seed'+str(seed+1)+'_z'+"%3.3d"%(idx)][()].copy()
                        sim_boxes, cparam_list = partition_cube(sim, length, opt.cubedim, sim_boxes, cparam_list=cparam_list, cond=cond, loglike_a=opt.loglike_a)
                else:
                    print('just using last redshift')
                    sim = ofile['seed'+str(seed+1)+'_z010'][()].copy()
                    sim_boxes, cparam_list = partition_cube(sim, length, opt.cubedim, sim_boxes, cparam_list=cparam_list, cond=cond, loglike_a=opt.loglike_a)
    
    assert len(sim_boxes)==len(cparam_list)
    return sim_boxes, cparam_list, conds

def load_in_simulations(opt):
    nsims = opt.nsims
    sim_boxes = []
    length = 512

    if opt.redshift_code:
        zlist = [] # array storing the redshift slice of a given volume in sim_boxes                               \
                        

        if opt.separate_sims:
            nload = 0
            for j, idx in enumerate(opt.redshift_idxs):

                #istart = nload % opt.n_max_sims
                #print('istart=', istart)
                for i in xrange(nsims):
                    print('loading sim', i, 'of', nsims, 'for redshift', opt.redshift_bins[j], 'idx=', nload%opt.n_max_sims +1)
                    with h5py.File(opt.base_path + 'n512_512Mpc_cosmo1_seed'+str(nload%opt.n_max_sims + 1)+'_gridpart.h5', 'r') as ofile:
                        sim = ofile["%3.3d"%(idx)][()].copy()
                        sim_boxes, zlist = partition_cube(sim, length, opt.cubedim, sim_boxes, cparam_list=zlist, cond=opt.redshift_bins[j], loglike_a=opt.loglike_a)
                        #sim_boxes, zlist = partition_cube(sim, length, opt.cubedim, sim_boxes, cparam_list=zlist, cond=[opt.redshift_bins[j]], loglike_a=opt.loglike_a, ds_factor=opt.ds_factor, fac=opt.fac)           
                    nload += 1
        else:

            for i in xrange(nsims):
                print('loading sim ', i, 'of ', nsims)
                with h5py.File(opt.base_path + 'n512_512Mpc_cosmo1_seed'+str(i+1)+'_gridpart.h5', 'r') as ofile:
                    for j, idx in enumerate(opt.redshift_idxs):
                        sim = ofile["%3.3d"%(idx)][()].copy()
                        sim_boxes, zlist = partition_cube(sim, length, opt.cubedim, sim_boxes, cparam_list=zlist, cond=opt.redshift_bins[j], loglike_a=opt.loglike_a)
                        #sim_boxes, zlist = partition_cube(sim, length, opt.cubedim, sim_boxes, cparam_list=zlist, cond=[opt.redshift_bins[j]], loglike_a=opt.loglike_a, ds_factor=opt.ds_factor, fac=opt.fac)
        #print('zlist:')
        #print(zlist)
        return np.array(sim_boxes), np.array(zlist)
    else:
        for i in xrange(nsims):
            with h5py.File(opt.base_path + 'n512_512Mpc_cosmo1_seed'+str(i+1)+'_gridpart.h5', 'r') as ofile:
                print('redshift index = ', opt.single_z_idx)
                sim = ofile["%3.3d"%(opt.single_z_idx)][()].copy()
                sim_boxes = partition_cube(sim, length, opt.cubedim, sim_boxes, loglike_a=opt.loglike_a, ds_factor=opt.ds_factor, fac=opt.fac)
            ofile.close()
#with h5py.File(opt.base_path + opt.file_name, 'r') as ofile:
#            for i in xrange(nsims):
#                sim = ofile['seed'+str(i+1)][()].copy()
#                sim_boxes = partition_cube(sim, length, opt.cubedim, sim_boxes, loglike_a=opt.loglike_a)

        return np.array(sim_boxes)

def loglike_transform(x, a=5):
    return (2*x/(x+a)) - 1

def loglike_transform_2(x, k=5.):
    return (1/float(k))*np.log10(x+1)

def log_transform_HI(x, eps=0.01, xmax=20000):
    return np.log10(x+eps)/np.log10(xmax+eps)

def model_weight_norm(model):
    totalnorm = 0.
    for x in model.parameters():
        totalnorm += x.norm(2)
    return totalnorm.item()

def inverse_log_transform_HI(s, eps=0.01, xmax=20000):
    return (xmax+eps)**s - eps

def inverse_loglike_transform_2(s, k=5.):
    return 10**(float(k)*s)-1

def piecewise_transform(x, a=20., c=100., xmax=50000):
    s_c = 2*c/(c+a)-1
    A = (1.-s_c)/np.log(xmax/c)
    s = np.piecewise(x, [x<=c, x>c], [lambda x: 2*x/(x+a)-1, lambda x: (2*c/(c+a)-1)+A*(np.log(x/c))])
    return s

def inverse_piecewise_transform(s, a=20., c=100., xmax=50000):
    s_c = 2*c/(c+a)-1
    x = np.piecewise(s, [s<=s_c, s>s_c], [lambda s: a*(s+1)/(1-s), lambda s: c*np.exp(((s-s_c)/(1.-s_c))*np.log(xmax/c))])
    return x


def make_feature_maps(array_vals, featureshape, device):
    #feature_maps = torch.from_numpy(np.array([[np.full(featureshape, v) for v in val] for val in array_vals])).to(device)
    feature_maps = torch.from_numpy(np.array([np.full(featureshape, val) for val in array_vals])).to(device)
    return torch.unsqueeze(feature_maps, 1).float()
    #return feature_maps.float()

def make_gif(dir):
    images = []
    files = os.listdir(dir)
    if len(files) > 0:
        for file in os.listdir(dir):
            images.append(imageio.imread(dir+'/'+file).resize((512,512)))
        imageio.mimsave(dir+'/movie.gif', images, fps=3)
    else:
        print('No images to save for make_gif().')

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

def partition_cube(sim, length, cubedim, sim_boxes, cparam_list=None, cond=None, loglike_a=None, ds_factor=1, fac=1):
    for x in xrange(int(length/cubedim)):
        for y in xrange(int(length/cubedim)):
            for r in xrange(int(length/cubedim)):
                if loglike_a is None:
                    if ds_factor != 1:
                        sim_boxes.append(blockwise_average_3D(sim[x*cubedim:(x+1)*cubedim, y*cubedim:(y+1)*cubedim, r*cubedim:(r+1)*cubedim], [ds_factor, ds_factor, ds_factor], fac=fac))
                    else:
                        sim_boxes.append(sim[x*cubedim:(x+1)*cubedim, y*cubedim:(y+1)*cubedim, r*cubedim:(r+1)*cubedim]\
\
)
                else:
                    if ds_factor != 1:
                        sim_boxes.append(loglike_transform(blockwise_average_3D(sim[x*cubedim:(x+1)*cubedim, y*cubedim:(y+1)*cubedim, r*cubedim:(r+1)*cubedim], [ds_factor, ds_factor, ds_factor], fac=fac)))
                    else:
                        sim_boxes.append(loglike_transform(sim[x*cubedim:(x+1)*cubedim, y*cubedim:(y+1)*cubedim, r*cubedim:(r+1)*cubedim], a=loglike_a))
                if cparam_list is not None:
                    cparam_list.append(cond)
    if cparam_list is not None:
        return sim_boxes, cparam_list
    else:
        return sim_boxes

def setup_result_directories():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    new_dir, frame_dir = create_directories(timestr)
    fake_dir = frame_dir+'/fake'
    os.makedirs(fake_dir)
    return timestr, new_dir, frame_dir, fake_dir

def save_current_state(ganopt, epoch, new_dir, gnorms, dnorms, lossGs, lossDs, weightG, weightD):
    save_nn(ganopt.netG, new_dir+'/netG_epoch_'+str(epoch))
    save_nn(ganopt.netD, new_dir+'/netD_epoch_'+str(epoch))

    np.savetxt(new_dir+'/generator_weight_norm_epoch_'+str(epoch)+'.txt', np.array(weightG))
    np.savetxt(new_dir+'/discriminator_weight_norm_epoch_'+str(epoch)+'.txt', np.array(weightD))


    np.savetxt(new_dir+'/generator_grad_norm_epoch_'+str(epoch)+'.txt',np.array(gnorms))
    np.savetxt(new_dir+'/discriminator_grad_norm_epoch_'+str(epoch)+'.txt',np.array(dnorms))
    np.savetxt(new_dir+'/generator_loss_epoch_'+str(epoch)+'.txt',np.array(lossGs))
    np.savetxt(new_dir+'/discriminator_loss_epoch_'+str(epoch)+'.txt',np.array(lossDs))

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

def sample_noise(bs, d):
    z = torch.randn(bs, d).float()
    return Variable(z.to(Device), requires_grad=True)

def weight_norm_vs_epoch(timestr):
    
    # first look at parameters and find how many epochs there are
    i = 0
    gen, pdict = nbody.restore_generator(timestr, epoch=0, n_condparam=ncond, discriminator=False, extra_conv_layers=0)
    n_epochs = pdict['n_epochs']
    norms = np.zeros(shape=(n_epochs,))
    
    for i in range(n_epochs):
        with HiddenPrints():
            gen, pdict = nbody.restore_generator(timestr, epoch=i, n_condparam=ncond, discriminator=False, extra_conv_layers=0)
            norms[i] = model_weight_norm(gen)
        
    return norms
