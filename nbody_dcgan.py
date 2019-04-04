import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
from __future__ import print_function
import argparse
import os
import random
import torch
import time
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from torchvision import transforms
from torch.utils import data
from power_spec import *
from models import *
from helpers import *

parser = argparse.ArgumentParser()
parser.add_argument('--trainSize', type=int, default=0, help='size of training dataset, if 0 then use unlimited samples')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--cubedim', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--latent_dim', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--n_epochs', type=int, default=10, help='number of is to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--cuda', type=bool, default=False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=4, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--alpha', type=float, default=-2.0, help='Slope of power law for gaussian random field')
parser.add_argument('--base_path', default='/work/06147/pberger/maverick2/gadget_runs/cosmo1/')
parser.add_argument('--file_name', default='n512_512Mpc_cosmo1_z0_gridpart.h5')
parser.add_argument('--datadim', type=int, default=3, help='2 to train on slices, 3 to train on volumes')

opt = parser.parse_known_args()[0]
if torch.cuda.is_available():
    opt.cuda=True
else:
    opt.cuda=False
    
opt.batchSize=32

device = torch.device("cuda:0" if opt.cuda else "cpu")
nc = 1 # just one channel for images                                                                                                                                                   
real_label = 1
fake_label = 0

sizes = make_size_array(opt.cubedim)
print(opt.cuda)

#Initialize Generator                                                                                                                                                                 
netG = DC_Generator3D(opt.ngpu, 1, opt.latent_dim, opt.ngf, sizes).to(device)
netG.apply(weights_init)
# Initialize Discriminator                                                                                                                                                             
netD = DC_Discriminator3D(opt.ngpu, 1, opt.ndf, sizes).to(device)
netD.apply(weights_init)

print(netG)
print(netD)

# Set loss                                                                                                                                                                             
criterion = nn.BCELoss()

# fixed noise used for sample generation comparisons at different points in training                                                                                                   
fixed_noise = torch.randn(opt.batchSize, opt.latent_dim, 1, 1, device=device)

# set up optimizers                                                                                                                                                                    
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


class GAN_optimization():
    def __init__(self, optimizerD, optimizerG, netD, netG):
        self.optimizerD = optimizerD
        self.optimizerG = optimizerG
        self.netD = netD
        self.netG = netG
        self.criterion = criterion
        if opt.trainSize > 0:
                self.batchSize = np.minimum(opt.trainSize, opt.batchSize)
        else:
                self.batchSize = opt.batchSize


    def discriminator_step(self, real_cpu):
        self.netD.zero_grad()

        label = torch.full((self.batchSize,), real_label, device=device)
        # reshape needed for images with one channel                                                                                                                          
        real_cpu = torch.unsqueeze(real_cpu, 1).float()
        output = self.netD(real_cpu)
        errD_real = self.criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake                                                                                                                                               

        noise = torch.randn(self.batchSize, opt.latent_dim, 1, 1, 1, device=device)
        fake = self.netG(noise)
        label.fill_(fake_label)
        output = self.netD(fake.detach())
        errD_fake = self.criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        return errD, D_x, D_G_z1

    def generator_step(self):
        noise = torch.randn(self.batchSize, opt.latent_dim, 1, 1, 1, device=device)
        fake = self.netG(noise)
        label = torch.full((self.batchSize,), real_label, device=device)
        self.netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost                                                                                                    \

        output = self.netD(fake)
        errG = self.criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        self.optimizerG.step()
        return errG, D_G_z2

    def single_train_step(self, real_cpu):
        errD, D_x, D_G_z1 = self.discriminator_step(real_cpu)
        for i in xrange(2):
                errG, D_G_z2 = self.generator_step()
        return errD, errG, D_x, D_G_z1, D_G_z2

ganopt = GAN_optimization(optimizerD, optimizerG, netD, netG)

# load in sims to memory
nsims = 1
sim_boxes = []
with h5py.File(opt.base_path + opt.file_name, 'r') as ofile:
    length = 512
    cubedim = 128
    for i in xrange(nsims):
        sim = ofile['seed'+str(i+1)][()].copy()
        for x in xrange(int(length/cubedim)):
            for y in xrange(int(length/cubedim)):
                for z in xrange(int(length/cubedim)):
                    sim_boxes.append(sim[x*cubedim:(x+1)*cubedim, y*cubedim:(y+1)*cubedim, z*cubedim:(z+1)*cubedim])

        
sim_boxes = np.array(sim_boxes)
print(len(sim_boxes), 'loaded into memory..')

for i in xrange(opt.n_epochs):
    lossGs, lossDs = [], []
    sim_idxs = np.arange(len(sim_boxes))
    while len(sim_idxs)>0:
        choice = np.random.choice(sim_idxs, opt.batchSize, replace=False)
        sim_idxs = np.array([x for x in sim_idxs if x not in choice])
        
        real_cpu = torch.from_numpy(sim_boxes[choice]).to(device)
        errD, errG, D_x, D_G_z1, D_G_z2 = ganopt.single_train_step(real_cpu)
    
