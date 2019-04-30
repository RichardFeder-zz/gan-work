from __future__ import print_function
import argparse
import itertools
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
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--latent_dim', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--n_epochs', type=int, default=10, help='number of is to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--alpha', type=float, default=-2.0, help='Slope of power law for gaussian random field')
parser.add_argument('--code_dim', type=int, default=0, help='latent code')
opt = parser.parse_args()
print(opt)



timestr = time.strftime("%Y%m%d-%H%M%S")
sizes = make_size_array(opt.imageSize)
if opt.manualSeed is None:
	opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")


device = torch.device("cuda:0" if opt.cuda else "cpu")
nc = 1 # just one channel for images
real_label = 1
fake_label = 0


new_dir, frame_dir = create_directories(timestr)
fake_dir = frame_dir+'/fake'
os.makedirs(fake_dir)

# Initialize Generator
netG = DC_Generator(opt.ngpu, nc, opt.latent_dim, opt.ngf, sizes).to(device)
netG.apply(weights_init)
if opt.netG != '':
	netG.load_state_dict(torch.load(opt.netG))
print(netG)

# Initialize Discriminator
netD = DC_Discriminator(opt.ngpu, nc, opt.ndf, sizes).to(device)
netD.apply(weights_init)
if opt.netD != '':
	netD.load_state_dict(torch.load(opt.netD))
print(netD)

# Set loss
criterion = nn.BCELoss()

if opt.code_dim > 0: # for infoGAN
        mutualinfo_loss = nn.MSELoss()
        lam_loss = 0.1 # coefficient in loss term

# fixed noise used for sample generation comparisons at different points in training
fixed_noise = torch.randn(opt.batchSize, opt.latent_dim, 1, 1, device=device)

# set up optimizers
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
if opt.code_dim > 0:
        optimizer_info = optim.Adam(itertools.chain(netG.parameters(), netD.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))

lossD_vals, lossG_vals = [[], []]

# Parameters
params = {'batch_size': opt.batchSize,
          'shuffle': True,
          'num_workers': opt.ngpu}


if opt.trainSize > 0:
	train_set = GRFDataset(root_dir='data/ps'+str(opt.imageSize), nsamp=opt.trainSize, transform=True)
	training_generator = data.DataLoader(train_set, **params)



class GAN_optimization():

	def __init__(self, optimizerD, optimizerG, netD, netG, optimizer_info=None):
		self.optimizerD = optimizerD
		self.optimizerG = optimizerG
                if optimizer_info is not None:
                        self.optimizer_info = optimizer_info
		self.netD = netD
		self.netG = netG
		self.criterion = criterion
		if opt.trainSize > 0:
			self.batchSize = np.minimum(opt.trainSize, opt.batchSize)
		else:
			self.batchSize = opt.batchSize 


        def info_step(self):
                self.optimizer_info.zero_grad()
                code_input = np.random.normal(-1, 1, (opt.batch_size, opt.code_dim))
                noise = torch.randn(self.batchSize, opt.latent_dim, 1, 1, device=device)
                gen_input = torch.cat((noise, torch.from_numpy(code_input).float()), 1)
                fake = self.netG(gen_input)
                _, pred_code = self.netD(fake)
                info_loss = lam_info * mutualinfo_loss(pred_code, code_input)
                info_loss.backward()
                optimizer_info.step()

                return info_loss

        def discriminator_step(self, real_cpu):
                self.netD.zero_grad()

                label = torch.full((self.batchSize,), real_label, device=device)
                # reshape needed for images with one channel                                                                                                                                         
                real_cpu = torch.unsqueeze(real_cpu, 1).float()
                if opt.code_dim > 0:
                        output, latent_code = self.netD(real_cpu)
                else:
                        output = self.netD(real_cpu)

                errD_real = self.criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                # train with fake                                                                                                                                                                    
                noise = torch.randn(self.batchSize, opt.latent_dim, 1, 1, device=device)
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
                noise = torch.randn(self.batchSize, opt.latent_dim, 1, 1, device=device)
                fake = self.netG(noise)
                label = torch.full((self.batchSize,), real_label, device=device)
                self.netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost                                                                                                                   
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
                if opt.code_dim > 0:
                        errI = self.info_step()
                        return errD, errG, errI, D_x, D_G_z1, D_G_z2

		return errD, errG, D_x, D_G_z1, D_G_z2

if opt.code_dim > 0:
        ganopt = GAN_optimization(optimizerD, optimizerG, netD, netG, optimizer_info)
else:
        ganopt = GAN_optimization(optimizerD, optimizerG, netD, netG)




for i in xrange(opt.n_epochs):

	if opt.trainSize > 0:
                lossGs, lossDs = [], []
		for local_batch in training_generator:
			real_cpu = local_batch['image'].to(device)
			errD, errG, D_x, D_G_z1, D_G_z2 = ganopt.single_train_step(real_cpu)
			lossGs.append(errG.item())
            lossDs.append(errD.item())

            lossG_vals.append(np.mean(np.array(lossGs)))
            lossD_vals.append(np.mean(np.array(lossDs)))

	else:
                if opt.code_dim > 0:
                        alphas = np.random.uniform(opt.alpha-1, opt.alpha+1, opt.batchSize)
                        dat, amps = gaussian_random_field(opt.batchSize, alphas, opt.imageSize)
                else:
                        dat, amps = gaussian_random_field(opt.batchSize, opt.alpha, opt.imageSize)
                
                normreal = amps.real/np.max(np.abs(amps.real))
                real_cpu = torch.from_numpy(normreal).to(device)
		
                if opt.code_dim > 0:
                        real_code = torch.from_numpy(alphas).to(device)
                        errD, errG, errI, D_x, D_G_z1, D_G_z2 = ganopt.single_train_step(real_cpu)
                        lossI_vals.append(errI.item())
                else:
                        errD, errG, D_x, D_G_z1, D_G_z2 = ganopt.single_train_step(real_cpu)
		lossG_vals.append(errG.item())
		lossD_vals.append(errD.item())


        if opt.n_epochs > 1000:
                if i % int(opt.n_epochs/100) == 0:
                        print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                              % (i, opt.n_epochs,
                                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                        if opt.code_dim > 0:
                                print('Loss_I: %.4f' % (errI.item()))
                        #vutils.save_image(real_cpu[:4],
                        #                  '%s/real_samples.png' % frame_dir,
                        #                  normalize=True)
        
                if i % int(opt.n_epochs/10) == 1 and opt.code_dim==0: # temporary for now, might change with infogan
                        fake = ganopt.netG(fixed_noise)
                        vutils.save_image(fake.detach()[:4],
                        		'%s/fake_samples_i_%03d.png' % (fake_dir, i),
                        			normalize=True)


        else:
            print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (i, opt.n_epochs, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            if i % int(opt.n_epochs/10) == 1:
                fake = ganopt.netG(fixed_noise)
                vutils.save_image(fake.detach()[:4],'%s/fake_samples_i_%03d.png' % (fake_dir, i), normalize=True)


fake = ganopt.netG(fixed_noise)
vutils.save_image(fake.detach()[:4], '%s/fake_samples_final.png' % (fake_dir), normalize=True)

save_nn(ganopt.netG, new_dir+'/netG')
save_nn(ganopt.netD, new_dir+'/netD')

save_params(new_dir, opt)
plot_loss_iterations(np.array(lossD_vals), np.array(lossG_vals), new_dir)
if opt.code_dim > 0:
        plot_info_iterations(np.array(lossI_vals), new_dir)
make_gif(fake_dir)
