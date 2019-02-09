import torch
from torch.autograd import Variable, grad
# from torch.nn.functional import binary_cross_entropy_with_logits as bce
import time
import numpy as np

import argparse
import torch.optim as optim

from helpers import *
from models import *

nparam_dict = dict({'2d_gaussian':5, '1d_gaussian':2, 'bernoulli':2, 'ring':2, 'grid':2})
outparam_dict = dict({'2d_gaussian':2, '1d_gaussian':1, 'bernoulli':1, 'ring':2, 'grid':2})

base_dir = '/Users/richardfeder/Documents/caltech/gan_work/results/'

Device = 'cpu'


timestr = time.strftime("%Y%m%d-%H%M%S")

# parse initial parameters, set to default if parameters left unspecified

parser = argparse.ArgumentParser()
parser.add_argument('--n_iterations', type=int, default=1000, help='number of iterations in training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--n_hidden', type=int, default=32)
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--extraD', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--loss_func', type=str, default='GAN', help='objective loss function used during training')
parser.add_argument('--architecture', type=str, default='GAN', help='GAN or ALI, more functionality here soon')
parser.add_argument('--sample_type', type=str, default='2d_gaussian', help='type of distribution to learn')
parser.add_argument('--verbosity', type=int, default=0, help='Verbosity level')
parser.add_argument('--conditional_gan', type=int, default=1, help='1 to use conditional GAN, 0 otherwise')
parser.add_argument('--gamma', type=float, default=1.0, help='Regularizing parameter for Roth GAN')
parser.add_argument('--sampling_method', type=str, default='Unif', help='Uniform (Unif) sampling or Latin Hypercube (LH) sampling on params during training')
parser.add_argument('--ngrid', type=int, default=8, help='number of partitions for each parameter when doing Latin Hypercube sampling')
parser.add_argument('--nmix', type=int, default=4, help='number of different param values to mix in a batch')
parser.add_argument('--activation', type=str, default='ReLU', help='type of activation function to use')
parser.add_argument('--weight_decay', type=float, default=2e-5, help='exponential decay factor of weights over run')
parser.add_argument('--lr_decay', type=float, default=0.99, help='exponential decay factor of weights over run')
parser.add_argument('--n_lr_decays', type=int, default=40, help='number of times to apply exponential decay to learning rate')


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


def main():

	# discriminator takes input x and z, outputs probability samples come from true or fake distribution
	print 'Initializing Discriminator'
	netD = Perceptron([outparam_dict[opt.sample_type]+opt.latent_dim+opt.n_cond_params] + [opt.n_hidden] * opt.n_layers + [1], opt.activation, sigmoid=True)
	
	# encoder takes in empirical data x, outputs estimate of latent vector z
	print 'Initializing Encoder'
	netE = Perceptron([outparam_dict[opt.sample_type]] + [opt.n_hidden] * opt.n_layers + [opt.latent_dim+opt.n_cond_params], opt.activation)

	# generator takes in latent noise, outputs sample x
	print 'Initializing Generator'
	netG = Perceptron([opt.latent_dim+opt.n_cond_params] + [opt.n_hidden] * opt.n_layers + [outparam_dict[opt.sample_type]], opt.activation)

	netD.to(Device)
	netG.to(Device)
	netE.to(Device)
	optD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
	optG = torch.optim.Adam([{'params' : netE.parameters()},{'params': netG.parameters()}], lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

	schedulerG = torch.optim.lr_scheduler.StepLR(optG, step_size=int(opt.n_iterations/opt.n_lr_decays), gamma=opt.lr_decay)
	schedulerD = torch.optim.lr_scheduler.StepLR(optD, step_size=int(opt.n_iterations/opt.n_lr_decays), gamma=opt.lr_decay)


	if opt.sampling_method == 'LH':
		print array_list[:2]
		samples = LHS(array_list[:nparam_dict[opt.sample_type]])
	else:
		samples = None

	criterion = nn.BCELoss()

	lossD_vals, lossG_vals = [],[]

	n_d_steps = 0

	# ----------
	#  Training
	# ----------

	for i in xrange(opt.n_iterations):

		if opt.n_cond_params > 0:
			real, cond_params = draw_true_samples(opt.batch_size, opt, samp_type=opt.sample_type, LH=samples)
		else:
			real = draw_true_samples(opt.batch_size, opt, samp_type=opt.sample_type)
		

		real = torch.from_numpy(real).float().requires_grad_(True)
		z_encoded = netE(real)

		gen_input = sample_noise(opt.batch_size, opt.latent_dim)

		if opt.n_cond_params > 0:
			gen_input = torch.cat((torch.from_numpy(cond_params).float(), gen_input), 1)

		fake = netG(gen_input)

		# real, fake has sample values, then conditional parameters, then latent z

		real = torch.cat((real, z_encoded), 1)
		fake = torch.cat((fake, gen_input), 1).requires_grad_(True)

		output_real = netD(real)
		output_fake = netD(fake)


		real_label = Variable(torch.ones(opt.batch_size))
		fake_label = Variable(torch.zeros(opt.batch_size))
		
		lossD = criterion(output_real, real_label) + criterion(output_fake, fake_label)
		lossG = criterion(output_fake, real_label) + criterion(output_real, fake_label)

		# print lossG.item()
		if lossG.item() < 1.5:
			n_d_steps += 1
			optD.zero_grad()
			lossD.backward(retain_graph=True)
			optD.step()

		optG.zero_grad()
		lossG.backward()
		optG.step()

		schedulerD.step()
		schedulerG.step()


		if (i % int(opt.n_iterations/10))==0:
			print 'Iteration', i, 'lossD =', np.round(lossD.item(), 6), 'lossG =', np.round(lossG.item(), 6)

		lossD_vals.append(-1*lossD.item())
		lossG_vals.append(lossG.item())

	plot_loss_iterations(np.array(lossD_vals), np.array(lossG_vals), new_dir)


	save_nn(netG, new_dir+'/netG')
	save_nn(netD, new_dir+'/netD')
	save_nn(netE, new_dir+'/netE')

	save_params(new_dir, opt)

	print 'number of discriminator steps:', n_d_steps

	if len(comments)>0:
		with open(new_dir+'/comments.txt', 'w') as p:
			p.write(comments)
			p.close()


main()



