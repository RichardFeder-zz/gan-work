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

class VAE_connected(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


''' This model taken from https://github.com/sksq96/pytorch-vae/blob/master/vae.py'''

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)
        return self.decoder(z), mu, logvar




class DC_Generator3D(nn.Module):
    def __init__(self, ngpu, nc, nz, ngf, sizes, endact='tanh'):
        super(DC_Generator3D, self).__init__()
        self.ngpu = ngpu
        layers = []

        
        kernel_sizes = [4, 4, 4, 4]
        strides = [1, 2, 2, 2]
        paddings = [0, 1, 1, 1]
        
        for i, size in enumerate(sizes):
            if i==0:
                layers.append(nn.ConvTranspose3d(nz, ngf*int(size), kernel_sizes[i], strides[i], paddings[i], bias=False) )
            else:
                layers.append(nn.ConvTranspose3d(outc, ngf*int(size), kernel_sizes[i], strides[i], paddings[i], bias=False) )
            #if i==0:
            #    layers.append(nn.ConvTranspose3d(nz, ngf*int(size), 4, stride=1, padding=0, bias=False))
            #else:
            #    layers.append(nn.ConvTranspose3d(outc, ngf*int(size), 4, stride=2, padding=1, bias=False))
                #layers.append(nn.ConvTranspose3d(outc, ngf*int(size), 2, stride=2, padding=0, bias=False))
            outc = ngf*int(size)
            layers.append(nn.BatchNorm3d(outc))
            layers.append(nn.ReLU(True))
        layers.append(nn.ConvTranspose3d(outc, nc, 4, stride=2, padding=1, bias=False))
        #layers.append(nn.ConvTranspose3d(outc, nc, 2, stride=2, padding=0, bias=False))
        if endact=='tanh': 
            layers.append(nn.Tanh())
        elif endact=='softplus':
            layers.append(nn.Softplus())
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class DC_Discriminator3D(nn.Module):
    def __init__(self, ngpu, nc, ndf, sizes):
        super(DC_Discriminator3D, self).__init__()
        self.ngpu = ngpu

        layers = []
        for i, size in enumerate(np.flip(sizes, 0)):
            if i==0:
                layers.append(nn.Conv3d(nc, ndf*int(size), 4, stride=2, padding=1, bias=False))
            else:
                layers.append(nn.Conv3d(outc, ndf*int(size), 4, stride=2, padding=1, bias=False))
                layers.append(nn.BatchNorm3d(ndf*int(size)))
            outc = ndf*int(size)
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        layers.append(nn.Conv3d(outc, 1, 4, stride=1, padding=0, bias=False))
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:  
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
