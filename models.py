from torch.autograd import Variable, grad
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


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
                layers.append(nn.ConvTranspose2d(nz, ngf*int(size), 4, 1, 0, bias=False))
            else:
                layers.append(nn.ConvTranspose2d(outc, ngf*int(size), 4, 2, 1, bias=False))
            outc = ngf*int(size) 
            layers.append(nn.BatchNorm2d(outc))
            layers.append(nn.ReLU(True))
        layers.append(nn.ConvTranspose2d(outc, nc, 4, 2, 1, bias=False))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

        # self.main = nn.Sequential(
        #     # input is Z, going into a convolution
        #     nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
        #     nn.BatchNorm2d(ngf * 8),
        #     nn.ReLU(True),
        #     # state size. (ngf*8) x 4 x 4
        #     nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 4),
        #     nn.ReLU(True),
        #     # state size. (ngf*4) x 8 x 8
        #     nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf * 2),
        #     nn.ReLU(True),
        #     # state size. (ngf*2) x 16 x 16
        #     nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ngf),
        #     nn.ReLU(True),
        #     # state size. (ngf) x 32 x 32
        #     nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
        #     nn.Tanh()
        #     # state size. (nc) x 64 x 64
        # )

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
        #sizes = [16, 8, 4, 2, 1] gets flipped to [1, 2, 4, 8, 16]
        for i, size in enumerate(np.flip(sizes)):
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

        # self.main = nn.Sequential(
        #     # input is (nc) x 64 x 64
        #     nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf) x 32 x 32
        #     nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*2) x 16 x 16
        #     nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*4) x 8 x 8
        #     nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*8) x 4 x 4
        #     nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
            
        #     nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        #     nn.Sigmoid()
        # )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


