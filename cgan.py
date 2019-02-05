import argparse
import os
import sys
import numpy as np
import math
import time
import cPickle as pickle

from torch.autograd import Variable, grad
import torch.nn as nn
import torch
import imageio

import matplotlib
matplotlib.use('tkAgg') 
import matplotlib.pyplot as plt
from helpers import *

Device = 'cpu'

# random seed for initialization
np.random.seed(20190129)

nparam_dict = dict({'2d_gaussian':5, '1d_gaussian':2, 'bernoulli':2, 'ring':0, 'grid':0})
outparam_dict = dict({'2d_gaussian':2, '1d_gaussian':1, 'bernoulli':1, 'ring':2, 'grid':2})
lamD = 10.

base_dir = '/Users/richardfeder/Documents/caltech/gan_work/results/'
timestr = time.strftime("%Y%m%d-%H%M%S")
# parse initial parameters, set to default if parameters left unspecified

parser = argparse.ArgumentParser()
parser.add_argument('--n_iterations', type=int, default=1000, help='number of iterations in training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--n_hidden', type=int, default=512)
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--extraD', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--loss_func', type=str, default='GAN', help='objective loss function used during training')
parser.add_argument('--sample_type', type=str, default='1d_gaussian', help='type of distribution to learn')
parser.add_argument('--verbosity', type=int, default=0, help='Verbosity level')
parser.add_argument('--conditional_gan', type=int, default=1, help='1 to use conditional GAN, 0 otherwise')
parser.add_argument('--gamma', type=float, default=1.0, help='Regularizing parameter for Roth GAN')
parser.add_argument('--sampling_method', type=str, default='Unif', help='Uniform (Unif) sampling or Latin Hypercube (LH) sampling on params during training')
parser.add_argument('--ngrid', type=int, default=8, help='number of partitions for each parameter when doing Latin Hypercube sampling')
parser.add_argument('--nmix', type=int, default=4, help='number of different param values to mix in a batch')
parser.add_argument('--activation', type=str, default='ReLU', help='type of activation function to use')

opt = parser.parse_args()

comments = str(raw_input("Any comments? "))

opt.n_cond_params = nparam_dict[opt.sample_type]
print(opt)


cuda = True if torch.cuda.is_available() else False


if opt.verbosity > 0:
	print 'Number of conditional parameters:'
	print opt.n_cond_params
	print 'Cuda: ', cuda

mu_range = np.linspace(-1.0, 1.0, opt.ngrid)
sig_range = np.linspace(0.1, 2.0, opt.ngrid)
# sig_range = np.linspace(0.01, 0.02, opt.ngrid)

array_list = [mu_range, sig_range, mu_range, sig_range, mu_range]

new_dir, frame_dir = create_directories(timestr)


class Perceptron(torch.nn.Module):
    def __init__(self, sizes, activation, final=None):
        super(Perceptron, self).__init__() # what does this line do?
        layers = []
        print 
        print 'Initializing Neural Net'
        print 'Activation: '+activation
        for i in xrange(len(sizes) - 1):
            print 'Layer', i, ':', sizes[i], sizes[i+1]
            layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
            if i != (len(sizes) - 2):
                if activation=='ReLU':
                    layers.append(torch.nn.ReLU())
                elif activation=='LeakyReLU':
                    layers.append(torch.nn.LeakyReLU()) # leaky relu, alpha=0.01
                elif activation=='ELU':
                    layers.append(torch.nn.ELU())
                elif activation=='PReLU':
                    layers.append(torch.nn.PReLU())

        if final is not None:
            layers.append(final())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
	'''Initialize the discriminator and generator from the Perceptron class, use Adam optimizer'''

	# take in (x,y) position along with conditional params, return softmax output
	netD = Perceptron([outparam_dict[opt.sample_type]+opt.n_cond_params] + [opt.n_hidden] * opt.n_layers + [1], opt.activation) 
	# take in latent vector and generate (x,y) position
	netG = Perceptron([opt.latent_dim+opt.n_cond_params] + [opt.n_hidden] * opt.n_layers + [outparam_dict[opt.sample_type]], opt.activation)
	
	netD.to(Device)
	netG.to(Device)
	optD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
	optG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
	lossD_vals, lossG_vals = [],[]

	times = np.zeros((5, opt.n_iterations))

	if opt.loss_func=='WGAN':
		# wasserstein gan objective for now
		objective = objective_wgan
	else:
		objective = objective_gan

	if opt.sampling_method == 'LH':
		print array_list[:2]
		samples = LHS(array_list[:nparam_dict[opt.sample_type]])
	else:
		samples = None

	if opt.verbosity > 1:
		print 'Samples:'
		print samples


	# ----------
	#  Training
	# ----------

	for i in xrange(opt.n_iterations):

		# training for discriminator with opt.extraD updates per one generator update

		dt0, dt1, dt2, dt3, dt4 = [0 for x in xrange(5)]

		for j in xrange(opt.extraD+1):
			
			t0 = time.clock()
			if opt.n_cond_params > 0:
				real, cond_params = draw_true_samples(opt.batch_size, opt, samp_type=opt.sample_type, LH=samples)
			else:
				real = draw_true_samples(opt.batch_size, opt, samp_type=opt.sample_type)
			dt0 += time.clock()-t0

			if opt.verbosity > 1:
				print 'conditional parameters:', real[0, 1:]

			gen_input = sample_noise(opt.batch_size, opt.latent_dim)
			
			if opt.n_cond_params > 0:
				gen_input = torch.cat((gen_input, cond_params), 1)
				if opt.verbosity > 1:
					print 'gen_input has shape', gen_input.shape
			
			t1 = time.clock()
			fake = netG(gen_input)
			dt1 += time.clock()-t1

			if opt.verbosity > 1:
				print 'fake has shape', fake.shape

			if opt.n_cond_params > 0:
				fake = torch.cat((fake, cond_params), 1).requires_grad_(True)

			if j < opt.extraD:

				t2 = time.clock()
				optD.zero_grad()
				d_real = netD(real)
				d_fake = netD(fake)
				lossD = objective(d_real, d_fake) # calculate the loss function
				dt2 += time.clock()-t2

				t3 = time.clock()
				
				#improved training of wasserstein gans
				penalty = grad(d_real.sum(), real, create_graph=True)[0].view(-1,1).norm(2,1).pow(2).mean()

				# gradD = grad(lossD * opt.batch_size, fake, create_graph=True)[0] 
				# reguD = gradD.norm(2, 1).clamp(1).mean()
				
				dt3 += time.clock()-t3

				t4 = time.clock()
				(lossD + (opt.gamma/2) * penalty).backward()
				# (lossD + lamD * reguD).backward() # backward prop with some regularization
				optD.step()
				dt4 += time.clock()-t3
			else:
				t2 = time.clock()
				optG.zero_grad()
				lossG = - objective(netD(real), netD(fake))
				dt2 += time.clock()-t2

				t4 = time.clock()
				(lossG).backward()
				optG.step()
				dt4 += time.clock() - t4
		

		# save losses from the last iteration
		lossD_vals.append(-1*lossD.item())
		lossG_vals.append(lossG.item())

		if (i % int(opt.n_iterations/10))==0:
			# print 'Iteration', i, 'lossD =', np.round(lossD.item(), 6), 'lossG =', np.round(lossG.item(), 6), 'reguD =', np.round(reguD.item(), 6)
			print 'Iteration', i, 'lossD =', np.round(lossD.item(), 6), 'lossG =', np.round(lossG.item(), 6)
			if opt.sample_type=='2d_gaussian':
				mus = [0.0, 0.0]
				sigs = [1.0, 1.0]
				rhos = [-0.8, 0.0, 0.8]
				g_samples = []
				true_samples = []
				for rh in rhos:
					cov = np.array([[sigs[0]**2, rh*sigs[0]*sigs[1]],[rh*sigs[0]*sigs[1], sigs[1]**2]])
					true_samp = np.random.multivariate_normal(mus, cov, 1000)
					gen_input = sample_noise(1000, opt.latent_dim)
					cond_params = torch.from_numpy(np.column_stack((np.full(1000, mus[0]), np.full(1000, sigs[0]), np.full(1000, mus[1]), np.full(1000, sigs[1]), np.full(1000, rh)))).float()
					gen_sample = netG(torch.cat((gen_input, cond_params), 1))

					g_samples.append(gen_sample.detach().numpy())
					true_samples.append(true_samp)
				save_frames(g_samples, true_samples, rhos, frame_dir, i)

			elif opt.sample_type=='ring' or opt.sample_type=='grid':
				real = draw_true_samples(2000, opt, samp_type=opt.sample_type)
				gen_input = sample_noise(2000, opt.latent_dim)
				fake = netG(gen_input)
				plot(real, fake, frame_dir, i)


		if opt.verbosity > 1:
			print 'lossD'
			print -1*lossD.item()
			print 'lossG'
			print lossG.item()

		times[0, i] = dt0
		times[1, i] = dt1
		times[2, i] = dt2
		times[3, i] = dt3
		times[4, i] = dt4

	
	plot_loss_iterations(np.array(lossD_vals), np.array(lossG_vals), new_dir)

	print 'frame_dir:', frame_dir
	make_gif(frame_dir)

	plot_comp_resources(times, new_dir)	

	save_nn(netG, new_dir+'/netG')
	save_nn(netD, new_dir+'/netD')


	save_params(new_dir, opt)

	if len(comments)>0:
		with open(new_dir+'/comments.txt', 'w') as p:
			p.write(comments)
			p.close()


##### Execute program ########    
main()


