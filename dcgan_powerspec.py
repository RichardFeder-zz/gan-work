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

# fixed noise used for sample generation comparisons at different points in training
fixed_noise = torch.randn(opt.batchSize, opt.latent_dim, 1, 1, device=device)


# set up optimizers
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

lossD_vals, lossG_vals = [[], []]

# Parameters
params = {'batch_size': opt.batchSize,
          'shuffle': True,
          'num_workers': opt.ngpu}



if opt.trainSize > 0:
	train_set = GRFDataset(root_dir='data/ps'+str(opt.imageSize))
	training_generator = data.DataLoader(train_set, **params)



class GAN_optimization():

	def __init__(self, optimizerD, optimizerG, netD, netG):
		self.optimizerD = optimizerD
		self.optimizerG = optimizerG
		self.netD = netD
		self.netG = netG
		self.criterion = criterion

	def single_train_step(self, real_cpu):
		self.netD.zero_grad()
		label = torch.full((opt.batchSize,), real_label, device=device)	
		# reshape needed for images with one channel
		real_cpu = torch.unsqueeze(real_cpu, 1).float()
		output = self.netD(real_cpu)
		errD_real = self.criterion(output, label)
		errD_real.backward()
		D_x = output.mean().item()

		# train with fake
		noise = torch.randn(opt.batchSize, opt.latent_dim, 1, 1, device=device)
		fake = self.netG(noise)
		label.fill_(fake_label)
		output = self.netD(fake.detach())
		errD_fake = self.criterion(output, label)
		errD_fake.backward()
		D_G_z1 = output.mean().item()
		errD = errD_real + errD_fake
		optimizerD.step()

		############################
		# (2) Update G network: maximize log(D(G(z)))
		###########################
		self.netG.zero_grad()
		label.fill_(real_label)  # fake labels are real for generator cost
		output = self.netD(fake)
		errG = self.criterion(output, label)
	        #print(errG.grad, errD.grad)
		errG.backward()
		D_G_z2 = output.mean().item()
		self.optimizerG.step()
		return errD, errG


ganopt = GAN_optimization(optimizerD, optimizerG, netD, netG)




for i in xrange(opt.n_epochs):
	############################
	# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
	###########################
	# train with real
	
	# if limited size dataset, iterate through all data in dataloader
	if opt.trainSize > 0:
		for j, local_batch in enumerate(training_generator):
			print('j:', j)
			real_cpu = local_batch.to(device)
			errD, errG = ganopt.single_train_step(real_cpu)
			lossG_vals.append(errG.item())
			lossD_vals.append(errD.item())


	else:
		data = torch.from_numpy(gaussian_random_field(opt.batchSize, opt.alpha, opt.imageSize))
		real_cpu = data.to(device)
		errD, errG = ganopt.single_train_step(real_cpu)
		print(errD.item(), errG.item())
		lossG_vals.append(errG.item())
		lossD_vals.append(errD.item())





	# data = torch.from_numpy(gaussian_random_field(opt.batchSize, opt.alpha, opt.imageSize))
	# real_cpu = data.to(device)
	# label = torch.full((opt.batchSize,), real_label, device=device)

	# # reshape needed for images with one channel
	# real_cpu = torch.unsqueeze(real_cpu, 1).float()

	# output = netD(real_cpu)
	# errD_real = criterion(output, label)
	# errD_real.backward()
	# D_x = output.mean().item()

	# # train with fake
	# noise = torch.randn(opt.batchSize, opt.latent_dim, 1, 1, device=device)
	# fake = netG(noise)
	# label.fill_(fake_label)
	# output = netD(fake.detach())
	# errD_fake = criterion(output, label)
	# errD_fake.backward()
	# D_G_z1 = output.mean().item()
	# errD = errD_real + errD_fake
	# optimizerD.step()

	# ############################
	# # (2) Update G network: maximize log(D(G(z)))
	# ###########################
	# netG.zero_grad()
	# label.fill_(real_label)  # fake labels are real for generator cost
	# output = netD(fake)
	# errG = criterion(output, label)
 #        #print(errG.grad, errD.grad)
	# errG.backward()
	# D_G_z2 = output.mean().item()
	# optimizerG.step()

	# lossG_vals.append(errG.item())
	# lossD_vals.append(errD.item())



        if opt.n_epochs > 1000:
                if i % int(opt.n_epochs/1000) == 0:
                        print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                              % (i, opt.n_epochs,
                                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                        vutils.save_image(real_cpu[:4],
                                          '%s/real_samples.png' % frame_dir,
                                          normalize=True)
                        fake = netG(fixed_noise)
                        #vutils.save_image(fake.detach()[:4],
                        #		'%s/fake_samples_i_%03d.png' % (fake_dir, i),
                        #			normalize=True)


save_nn(netG, new_dir+'/netG')
save_nn(netD, new_dir+'/netD')

save_params(new_dir, opt)
plot_loss_iterations(np.array(lossD_vals), np.array(lossG_vals), new_dir)
make_gif(fake_dir)
