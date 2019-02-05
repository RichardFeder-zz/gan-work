from torch.autograd import Variable, grad
import torch.nn as nn
import torch

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

# will update this one later right now it is not set up 
class ConvNet(torch.nn.Module):
    def __init__(self, sizes, activation, final=None):
        super(ConvNet, self).__init__() # what does this line do?
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