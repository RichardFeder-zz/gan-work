from torch.autograd import Variable, grad
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import cPickle as pickle

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def restore_conv_nn_new(filepath, sizes):
    # get config parameters from run and use to construct Perceptron object
    filen = open(filepath+'/params.txt','r')
    pdict = pickle.load(filen)
    model = DC_Generator(pdict['ngpu'], 1, pdict['latent_dim'], pdict['ngf'], sizes)
    
    model.load_state_dict(torch.load(filepath+'/netG', map_location='cpu'))
    model.eval()
    
    return model, pdict

class Perceptron(torch.nn.Module):
    def __init__(self, sizes, activation, final=None, sigmoid=False):
        super(Perceptron, self).__init__() # what does this line do?
        layers = []
        self.sigmoid = sigmoid
        print 
        print 'Initializing Neural Net'
        print 'Activation: '+activation
        for i in xrange(len(sizes) - 1):
            print 'Layer', i, ':', sizes[i], sizes[i+1]
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i != (len(sizes) - 2):
                if activation=='ReLU':
                    layers.append(nn.ReLU())
                elif activation=='LeakyReLU':
                    layers.append(nn.LeakyReLU()) # leaky relu, alpha=0.01
                elif activation=='ELU':
                    layers.append(nn.ELU())
                elif activation=='PReLU':
                    layers.append(nn.PReLU())

        if final is not None:
            layers.append(final())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        output = self.net(x)
        if self.sigmoid:
            output = F.sigmoid(output)
        return output



class DC_Generator(nn.Module):
    def __init__(self, ngpu, nc, nz, ngf, sizes):
        super(DC_Generator, self).__init__()
        self.ngpu = ngpu
        layers = []

        for i, size in enumerate(sizes):
            print(i, size)
            if i==0:
                layers.append(nn.ConvTranspose2d(nz, ngf*int(size), 4, stride=1, padding=0, bias=False))
            else:
                layers.append(nn.ConvTranspose2d(outc, ngf*int(size), 2, stride=2, padding=0, bias=False))
          #      layers.append(nn.ConvTranspose2d(outc, ngf*int(size), 4, stride=2, padding=1, bias=False))
            outc = ngf*int(size) 
            layers.append(nn.BatchNorm2d(outc))
            layers.append(nn.ReLU(True))
        layers.append(nn.ConvTranspose2d(outc, nc, 4, 2, 1, bias=False))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class DC_Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf, sizes):
        super(DC_Discriminator, self).__init__()
        self.ngpu = ngpu

        layers = []
        for i, size in enumerate(np.flip(sizes, 0)):
            print(i, size)
            if i==0:
                layers.append(nn.Conv2d(nc, ndf*int(size), 4, 2, 1, bias=False))
            else:
                layers.append(nn.Conv2d(outc, ndf*int(size), 4, 2, 1, bias=False))
                layers.append(nn.BatchNorm2d(ndf*int(size)))
            outc = ndf*int(size)
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv2d(outc, 1, 4, 1, 0, bias=False))
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


