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

import matplotlib
matplotlib.use('tkAgg') 
import matplotlib.pyplot as plt
from helpers import *

Device = 'cpu'

nparam_dict = dict({'2d_gaussian':5, '1d_gaussian':2, 'bernoulli':2, 'ring':2, 'grid':2})
outparam_dict = dict({'2d_gaussian':2, '1d_gaussian':1, 'bernoulli':1, 'ring':2, 'grid':2})
lamD = 10.
base_dir = '/Users/richardfeder/Documents/caltech/gan_work/results/'
timestr = time.strftime("%Y%m%d-%H%M%S")
# parse initial parameters, set to default if parameters left unspecified

parser = argparse.ArgumentParser()
parser.add_argument('--n_iterations', type=int, default=1000, help='number of iterations in training')
parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
parser.add_argument('--n_hidden', type=int, default=512)
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--extraD', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--loss_func', type=str, default='MSE', help='type of loss function used during training')
parser.add_argument('--sample_type', type=str, default='1d_gaussian', help='type of distribution to learn')
parser.add_argument('--verbosity', type=int, default=0, help='Verbosity level')
parser.add_argument('--conditional_gan', type=int, default=1, help='1 to use conditional GAN, 0 otherwise')
opt = parser.parse_args()
print(opt)


comments = str(raw_input("Any comments? "))


n_cond_params = nparam_dict[opt.sample_type] if opt.conditional_gan else 0

cuda = True if torch.cuda.is_available() else False


if opt.verbosity > 0:
	print 'Number of conditional parameters:'
	print n_cond_params
	print 'Cuda: ', cuda


def draw_true_samples(nsamp, samp_type='2d_gaussian', *argv):

	'''For 2d_gaussian, *argv arguments are mu_0, mu_1, sig_1, sig_2, rho'''
	if samp_type=='2d_gaussian':
		mus = np.random.choice(np.linspace(-1, 1, 5), size=2)
		sigs = np.random.choice(np.linspace(0.01, 2, 5), size=2)
		rho = np.random.choice(np.linspace(-1, 1, 5))
		
		cov = np.array([[sigs[0]**2, rho*sigs[0]*sigs[1]],[rho*sigs[0]*sigs[1], sigs[1]**2]])
		
		conditional_params = np.column_stack((np.full(nsamp, mus[0]), np.full(nsamp, mus[1]), np.full(nsamp, sigs[0]), np.full(nsamp, sigs[1]), np.full(nsamp, rho)))
		
		s = torch.from_numpy(np.column_stack((np.random.multivariate_normal(mus, cov, nsamp), conditional_params)))
		conditional_params = torch.from_numpy(conditional_params)

	elif samp_type=='1d_gaussian':
		if len(argv)==0:
			mu = np.random.choice(np.linspace(-1, 1, 10))
			sig = np.random.choice(np.linspace(0.01, 2, 10))
		else:
			mu = argv[0]
			sig = argv[1]

		s = np.column_stack((np.random.normal(mu, sig, nsamp), np.full(nsamp, mu), np.full(nsamp, sig)))
		conditional_params = np.column_stack((np.full(nsamp, mu), np.full(nsamp, sig)))
		conditional_params = torch.from_numpy(conditional_params)


		s = torch.from_numpy(s)
	
	elif samp_type=='bernoulli':
		n = argv[0]
		p = argv[1]

		conditional_params = torch.cat((n,p), 0)
		s = torch.from_numpy(np.random.binomial(n, p, nsamp))

	elif samp_type == 'ring':
		k = argv[0]
		std = argv[1]
		m = means_circle(k)
		i = torch.zeros(nsamp).random_(m.size(0)).long()

		conditional_params = torch.cat((k, std), 0)
		s = torch.randn(nsamp, 2) * std + m[i]
	
	elif samp_type == 'grid':
		k = argv[0]
		std = argv[1]
		m = means_grid(k)
		i = torch.zeros(nsamp).random_(m.size(0)).long()
		conditional_params = torch.cat((k,std), 0)
		s = torch.randn(nsamp, 2) * std + m[i]

	return s.float(), conditional_params.float()



def sample_noise(bs, d):
	z = torch.randn(bs, d).float()
	return Variable(z.to(Device), requires_grad=True)


def main():
	'''Initialize the discriminator and generator from the Perceptron class, use Adam optimizer'''

	# take in (x,y) position along with conditional params, return softmax output
	netD = Perceptron([outparam_dict[opt.sample_type]+n_cond_params] + [opt.n_hidden] * opt.n_layers + [1]) 
	# take in latent vector and generate (x,y) position
	netG = Perceptron([opt.latent_dim+n_cond_params] + [opt.n_hidden] * opt.n_layers + [outparam_dict[opt.sample_type]])
	
	netD.to(Device)
	netG.to(Device)
	optD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
	optG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
	lossD_vals, lossG_vals = [],[]

	times = np.zeros((5, opt.n_iterations))


	# wasserstein gan objective for now
	objective = objective_wgan

	# ----------
	#  Training
	# ----------

	for i in xrange(opt.n_iterations):

		# training for discriminator with opt.extraD updates per one generator update

		dt0, dt1, dt2, dt3, dt4 = [0 for x in xrange(5)]

		for j in xrange(opt.extraD+1):
			
			t0 = time.clock()
			real, cond_params = draw_true_samples(opt.batch_size, samp_type=opt.sample_type)
			dt0 += time.clock()-t0

			if opt.verbosity > 1:
				print 'conditional parameters:', real[0, 1:]

			gen_input = sample_noise(opt.batch_size, opt.latent_dim)
			
			if cond_params is not None:

				gen_input = torch.cat((gen_input, cond_params), 1)
				if opt.verbosity > 1:
					print 'gen_input has shape', gen_input.shape
			
			t1 = time.clock()
			fake = netG(gen_input)
			dt1 += time.clock()-t1

			if opt.verbosity > 1:
				print 'fake has shape', fake.shape

			fake = torch.cat((fake, cond_params), 1)

			if j < opt.extraD:

				t2 = time.clock()
				optD.zero_grad()
				lossD = objective(netD(real), netD(fake)) # calculate the loss function
				dt2 += time.clock()-t2

				t3 = time.clock()
				gradD = grad(lossD * opt.batch_size, fake, create_graph=True)[0] 
				reguD = gradD.norm(2, 1).clamp(1).mean()
				dt3 += time.clock()-t3

				t4 = time.clock()
				(lossD + lamD * reguD).backward() # backward prop with some regularization
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
			print 'Iteration', i, 'lossD =', np.round(lossD.item(), 6), 'lossG =', np.round(lossG.item(), 6), 'reguD =', np.round(reguD.item(), 6)

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

    
	new_dir, frame_dir = create_directories(timestr)
	
	plot_loss_iterations(lossD_vals, lossG_vals, new_dir)

	plot_comp_resources(times, new_dir)	

	save_nn(netG, new_dir+'/netG')
	save_nn(netD, new_dir+'/netD')


	save_params(new_dir, opt)

	with open(new_dir+'/comments.txt', 'w') as p:
		p.write(comments)
		p.close()


##### Execute program ########    
main()


