from __future__ import print_function
import numpy as np
import h5py
import argparse
import os
import random
import torch
torch.backends.cudnn.benchmark = True
from optimizers import ACGD
#from cgd_utils import zero_grad
import time
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from astropy.cosmology import FlatLambdaCDM
from torchvision import transforms
from torch.utils import data
from power_spec import *
from models import *
from helpers import *
from plotting_fns import *

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
opt = get_parsed_arguments('nbody')

richard_workdir = '/work/06224/rfederst/maverick2/results/'

if not torch.cuda.is_available():
    opt.cuda=False
    
device = torch.device("cuda:0" if opt.cuda else "cpu")
nc = 1 # just one channel for images                                                                                                                
real_label = 1
fake_label = 0
n_cond_params = 0

if opt.redshift_code:
    n_cond_params += 1
if opt.lcdm > 0:
    n_cond_params += opt.lcdm
    opt.base_path = "/work/06147/pberger/maverick2/gadget_runs/cgrid0/"

opt.n_cond_params = n_cond_params

sizes = np.array([8,4,2,1])

#redshift_bins = np.array([1., 0.5, 0.25, 0.])
#redshift_strings = np.array(['006', '007', '008', '009'])
redshift_bins = np.array([3., 1.5, 0.5, 0.])

cond_bins = np.array([[[0.801, 0.28724518], [0.801, 0.3132452],  [0.801, 0.3392452 ]],
             [[0.829, 0.28724518], [0.829, 0.3132452 ], [0.829, 0.3392452 ]],
             [[0.857, 0.28724518], [0.857, 0.3132452 ], [0.857, 0.3392452 ]]])


age_bins = cosmo.age(redshift_bins)/cosmo.age(0)
redshift_strings = np.array(['003', '005','007', '009'])
#redshift_bins = np.array([5., 3., 1., 0.5, 0.])
#redshift_strings = np.array(['002', '003', '006', '007', '009'])
#redshift_bins = np.array([10., 7.5, 5., 3., 2., 1.5, 1., 0.5, 0.25, 0.]) 
#redshift_strings = np.array(["%3.3d"%(i) for i in xrange(len(redshift_bins))])

if opt.redshift_code:
    opt.redshift_bins=redshift_bins
    opt.redshift_idxs = np.array([int(r) for r in redshift_strings])
    output1shape = (opt.cubedim/2, opt.cubedim/2, opt.cubedim/2) # for conditional feature maps in discriminator  
    print('Redshifts:', redshift_bins)
    print('Redshift strings:', redshift_strings)
    print('Redshift idxs:', opt.redshift_idxs)
else:
    opt.redshift_bins=[]
    opt.redshift_idxs=[]

endact = 'tanh'
if opt.wgan:
    print('Using Wasserstein GAN')
    endact = 'linear'

timestr, new_dir, frame_dir, fake_dir = setup_result_directories()
opt.timestr = timestr

# save metadata of run
save_params(new_dir, opt)

# Initialize Generator and Discriminator                                                                                   
print('Endact:', endact)
netG = DC_Generator3D_simpler(opt.ngpu, 1, opt.latent_dim+n_cond_params, opt.ngf, sizes, extra_conv_layers=opt.extra_conv_layers).to(device)
netD = DC_Discriminator3D_simpler(opt.ngpu, 1, opt.ndf, sizes, device, n_cond_features = n_cond_params).to(device)

if opt.netG=='':
    netG.apply(weights_init)
    netD.apply(weights_init)
else:
    netG.load_state_dict(torch.load(richard_workdir+opt.netG+'/netG_epoch_29', map_location=device))
    netD.load_state_dict(torch.load(richard_workdir+opt.netG+'/netD_epoch_29', map_location=device))
    netG.train()
    netD.train()

print(netG)
print(netD)

# Set loss                                                                                                                                           
criterion = nn.BCELoss()
print('device:', device)
if opt.acgd:
    optimizer = ACGD(max_params=netG, min_params=netD, lr=opt.lr_g, device=device)

else:                                                                         
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))

# sets learning rate scheduler to decay as training continues

if opt.schedule:
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=opt.step_size, gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=opt.step_size, gamma=opt.gamma)
else:
    schedulerD = None
    schedulerG = None

lossD_vals, lossG_vals = [[], []]


class GAN_optimization():
    def __init__(self, optimizerD, optimizerG, netD, netG, cond_params=None, conds=None, schedulerD=None, schedulerG=None, optimizer=None):
        if optimizer is not None:
            self.optimizer = optimizer
            print('self.optimizer:', self.optimizer)
        else:
            self.optimizerD = optimizerD
            self.optimizerG = optimizerG
        self.netD = netD
        self.netG = netG
        self.criterion = criterion
        if cond_params is None:
            self.cond_params = cond_params
        else:
            self.cond_params = np.array(cond_params)
        self.conds = np.array(conds)
        if opt.trainSize > 0:
            self.batchSize = np.minimum(opt.trainSize, opt.batchSize)
        else:
            self.batchSize = opt.batchSize
        if schedulerD is not None:
            self.schedulerD = schedulerD
        if schedulerG is not None:
            self.schedulerG = schedulerG

    def draw_latent_z(self, latent_dim, device, redshifts=None, fixed_c=None, df=None):
        if df is not None:
            noise = torch.distributions.StudentT(df=df).rsample(sample_shape=(self.batchSize, latent_dim, 1, 1, 1)).float().to(device)
        else:
            noise = torch.randn(self.batchSize, latent_dim, 1, 1, 1, device=device)
        if self.cond_params is not None:
            if fixed_c is not None:
                c = np.repeat([fixed_c], opt.batchSize, axis=0)
            else:
                cidx = np.random.choice(np.arange(len(self.conds)), opt.batchSize)
                c = self.conds[cidx]
            noise = torch.cat((noise, torch.from_numpy(c).float().view(-1,c.shape[1],1,1,1).to(device)),1)
            return noise, c
        return noise, None

    def acgd_step(self, real_cpu, c=None, zs=None, df=None):
        label = torch.full((self.batchSize,), real_label, device=device)
        real_cpu = torch.unsqueeze(real_cpu, 1).float()
        d_real = self.netD(real_cpu, cond=c)
        #loss_real = self.get_loss(d_real, opt, label=label)
        loss_real = self.criterion(d_real, label)
        noise, c = self.draw_latent_z(opt.latent_dim, device, df=df)
        fake = self.netG(noise)
        label.fill_(fake_label)
        d_fake = self.netD(fake.detach(), cond=c)
        #loss_fake = self.get_loss(d_fake, opt, label=label)
        loss_fake = self.criterion(d_fake, label)
        loss = loss_real + loss_fake
        #print('loss:', loss)
        #loss.backward()
        #loss_fake.backward()
        self.optimizer.zero_grad()
        self.optimizer.step(loss=loss)
        
        return loss_real, loss_fake

    def discriminator_step(self, real_cpu, disc_opt=False, c=None, zs=None, df=None):
        self.netD.zero_grad()

        # train with real
        label = torch.full((self.batchSize,), real_label, device=device)
        real_cpu = torch.unsqueeze(real_cpu, 1).float() # reshape for 1 channel images                        
        output = self.netD(real_cpu, cond=c)
        errD_real = self.get_loss(output, opt, label=label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise, c = self.draw_latent_z(opt.latent_dim, device, df=df)
        fake = self.netG(noise)
        #print('fake has shape,', fake.shape)
        label.fill_(fake_label)
        output = self.netD(fake.detach(), cond=c)
        errD_fake = self.get_loss(output, opt, label=label)
        errD_fake.backward()

        # train with gradient penalty, will need to fix at some point
        if opt.wgan:
            if zs is not None:
                gradient_penalty = calc_gradient_penalty(self.netD, opt, real_cpu, fake, realz_feat=make_feature_maps(zs, output1shape, device), fakez_feat = make_feature_maps(fake_zs, output1shape, device))
            else:
                gradient_penalty = calc_gradient_penalty(self.netD, opt, real_cpu, fake) 
            print('Gradient penalty:', gradient_penalty.item())
            gradient_penalty.backward()

        dnorm = compute_gradient_norm(self.netD)
        D_G_z1 = output.mean().item()

        if opt.wgan:
            D_cost = errD_fake - errD_real + gradient_penalty
            errD = errD_real - errD_fake
            self.optimizerD.step()
        else:
            errD = errD_real + errD_fake
            if D_G_z1 > 0.2 or disc_opt==True: # might not need this for wgan
                self.optimizerD.step()

        return errD, D_x, D_G_z1, dnorm

    def get_loss(self, output, opt, label=None):
        if opt.wgan:
            return self.criterion(output, label)
            #return output.mean()
        else:
            return self.criterion(output, label)

    def generator_step(self, zs=None, df=None):
        
        noise, c = self.draw_latent_z(opt.latent_dim, device, df=df)
        fake = self.netG(noise)
        label = torch.full((self.batchSize,), real_label, device=device)
        self.netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost  
        output = self.netD(fake, cond=c)        
        errG = self.get_loss(output, opt, label=label)
        errG.backward()
        gnorm = compute_gradient_norm(self.netG)
        D_G_z2 = output.mean().item()
        self.optimizerG.step()
        return errG, D_G_z2, gnorm

    def single_train_step(self, real_cpu, disc_opt=False, zs=None, cparams=None, df=None):
        if opt.acgd:
            loss_real, loss_fake = self.acgd_step(real_cpu, zs=zs, c=cparams, df=df)
            return loss_real, loss_fake
        else:
            errD, D_x, D_G_z1, dnorm = self.discriminator_step(real_cpu, disc_opt=disc_opt, zs=zs, c=cparams, df=df)
            for i in xrange(opt.n_genstep):
                errG, D_G_z2, gnorm = self.generator_step(zs=zs, df=df)
            if opt.get_optimal_discriminator:
                return errD, D_x, D_G_z1, dnorm
            return errD, errG, D_x, D_G_z1, D_G_z2, gnorm, dnorm


def training_epoch(opt, ganopt, epoch, sim_boxes, lossGs, lossDs, gnorms, dnorms, cparam_list=None, disc_opt=False):
    print('disc opt is', disc_opt)
    if opt.df is not None:
        print('Latent space is described by Student-T distribution with', opt.df,'degrees of freedom')
    sim_idxs = np.arange(len(sim_boxes))
    nbatch = 0
    while len(sim_idxs)>0:
        nbatch += 1
        dat, choice = draw_training_batch(sim_boxes, sim_idxs, opt)
        sim_idxs = np.array([x for x in sim_idxs if x not in choice])
        real_cpu = torch.from_numpy(np.array(dat)).to(device)
        if cparam_list is not None:
            cparams = cparam_list[choice]
            if opt.get_optimal_discriminator:
                errD, D_x, D_G_z1, dn = ganopt.single_train_step(real_cpu, cparams=cparams, disc_opt=disc_opt, df=opt.df)
            else:
                if opt.acgd:
                    loss_real, loss_fake = ganopt.single_train_step(real_cpu, cparams=cparams, df=opt.df)
                else:
                    errD, errG, D_x, D_G_z1, D_G_z2, gn, dn = ganopt.single_train_step(real_cpu, cparams=cparams, df=opt.df)
        else:
            if opt.get_optimal_discriminator:
                errD, D_x, D_G_z1, dn = ganopt.single_train_step(real_cpu, disc_opt=disc_opt, df=opt.df)
            else:
                if opt.acgd:
                    loss_real, loss_fake = ganopt.single_train_step(real_cpu, df=opt.df)
                else:
                    errD, errG, D_x, D_G_z1, D_G_z2, gn, dn = ganopt.single_train_step(real_cpu, df=opt.df)
        if opt.get_optimal_discriminator:
            print('errD:', errD.item(), 'D_G_z1:', D_G_z1, 'Dnorm:', dn)
        elif opt.acgd:
            lossGs.append(loss_fake.item())
            lossDs.append(loss_real.item())
            print('loss_real', loss_real.item(), 'loss_fake:', loss_fake)
        else:
            #print('errG:', errG.item(),'errD:', errD.item(), 'D_G_z1:', D_G_z1, 'Gnorm:', gn, 'Dnorm:',dn)
            print('errG:', np.round(errG.item(), 3),'errD:', np.round(errD.item(), 3), 'D_G_z1:', np.round(D_G_z1, 3), 'Gnorm:', np.round(gn, 3), 'Dnorm:',np.round(dn, 3))
            assert errG.item() != 0.0 # these make sure we're not falling into mode collapse or an outperforming discriminator
            gnorms.append(gn)
            lossGs.append(errG.item())
        
            assert errD.item() != 0.0
            dnorms.append(dn)
            lossDs.append(errD.item())

    if not opt.get_optimal_discriminator:
        
        print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (i, opt.n_epochs, np.round(np.mean(lossDs[-nbatch:]), 4), np.round(np.mean(lossGs[-nbatch:]), 4), np.round(D_x, 4), np.round(D_G_z1, 4), np.round(D_G_z2, 4)))

    save_current_state(ganopt, epoch, new_dir, gnorms, dnorms, lossGs, lossDs)

    return lossGs, lossDs, gnorms, dnorms, ganopt


if n_cond_params > 0:
    if opt.lcdm > 0:
        sim_boxes, cparam_list, conds = load_in_omegam_s8_sims(opt)
    else:
        sim_boxes, cparam_list = load_in_simulations(opt)
else:
    sim_boxes = load_in_simulations(opt)
    #sim_boxes *= -1.0 # tested this, doesn't make a difference on final results
    cparam_list = None
    conds = None

opt.himax = np.max(sim_boxes)
print('maximum value is', opt.himax)
if opt.loglike_a > 0.0:
    opt.log_scaling = False
if opt.log_scaling:
    print('Using log scaling..')
    sim_boxes = log_transform_HI(sim_boxes, xmax=opt.himax)
elif opt.piecewise_scaling:
    print('Using piecewise scaling..')
    max_sim_val = np.max(sim_boxes)
    if max_sim_val>opt.xmax:
        print('xmax set to', max_sim_val)
    sim_boxes = piecewise_transform(sim_boxes, a=opt.loglike_a, c=opt.c, xmax=np.maximum(opt.xmax, max_sim_val))


print(len(sim_boxes), 'loaded into memory..')
print('min/max values:', np.min(sim_boxes), np.max(sim_boxes))
gnorms, dnorms, lossGs, lossDs = [[] for x in xrange(4)]

if opt.acgd:
    ganopt = GAN_optimization(optimizer, optimizer, netD, netG, cond_params=cparam_list, conds=conds, optimizer=optimizer)
else:
    ganopt = GAN_optimization(optimizerD, optimizerG, netD, netG, cond_params=cparam_list, conds=conds)



for i in xrange(opt.n_epochs):
    lossGs, lossDs, gnorms, dnorms, ganopt = training_epoch(opt, ganopt, i, sim_boxes, lossGs, lossDs, gnorms, dnorms, cparam_list=cparam_list)
    
print('saving models..')
save_nn(ganopt.netG, new_dir+'/netG')
save_nn(ganopt.netD, new_dir+'/netD')

np.savetxt(new_dir+'/generator_grad_norm.txt', np.array(gnorms))
np.savetxt(new_dir+'/discriminator_grad_norm.txt', np.array(dnorms))

plot_loss_iterations(np.array(lossDs), np.array(lossGs), new_dir)

if opt.get_optimal_discriminator:
    '''set number of generator steps to zero and train discriminator until it minimizes loss'''                                               
    print('Training discriminator only to minimize its loss..')
    opt.n_genstep = 0
    for i in xrange(opt.disc_only_epochs):
        print('Epoch '+str(i)+' of '+str(opt.disc_only_epochs))
        print
        lossGs, lossDs, gnorms, dnorms, ganopt = training_epoch(opt, ganopt, opt.n_epochs+i, sim_boxes, lossGs, lossDs, gnorms, dnorms, disc_opt=True)
        save_nn(ganopt.netD, richard_workdir+opt.netG+'/netD_optimal_epoch_'+str(i))
        
#save_nn(ganopt.netD, new_dir+'/netD_optimal')
save_nn(ganopt.netD, richard_workdir+opt.netG+'/netD_optimal')

#plot_loss_iterations(np.array(lossDs), np.array(lossGs), new_dir)
#plot_loss_iterations(np.array(lossD_vals), np.array(lossG_vals), new_dir)
