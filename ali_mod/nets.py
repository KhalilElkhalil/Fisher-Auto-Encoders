import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import grad

import math

import flows

# VAE class 
class VAE(nn.Module):
    def __init__(self, feature_size, latent_size, conv=None, flow=None, exp_family=True, M=5, Fisher=True):
        super(VAE, self).__init__()
        self.latent_size = latent_size 

        if conv:
            self.enc, self.dec = get_conv_nets(latent_size = latent_size, **conv)
            self.enc1 = nn.Linear(conv['width']*8*4*4, latent_size)
            self.enc2 = nn.Linear(conv['width']*8*4*4, latent_size)

        else:
            # encoder
            self.enc = nn.Sequential(nn.Linear(feature_size, 512), nn.ReLU(True), 
                                     nn.Linear(512, 256), nn.ReLU(True))

            self.enc1 = nn.Linear(256, latent_size)
            self.enc2 = nn.Linear(256, latent_size)

            # decoder
            self.dec = nn.Sequential(nn.Linear(latent_size, 256), nn.ReLU(True), 
                                     nn.Linear(256, 512), nn.ReLU(True), nn.Linear(512, feature_size))
        
        # Exp. family prior/posterior 
        self.M = M
        self.exp_coef = nn.Parameter(torch.randn(M, latent_size).normal_(0, 0.01))
        
        # Fisher/KL VAE 
        self.Fisher = Fisher 
        
        # use exp_family model for prior
        self.exp_family = exp_family
        
        # exp. family natural parameter/ sufficient statistic
        self.natural_param = nn.Parameter(torch.randn(M*latent_size, 1).normal_(0, 0.01))

        # flow for more complicated latent
        self.flow = flow 

        # sufficient statistic 
        activation = nn.Softplus()
        self.sufficient_stat = nn.Sequential(\
                nn.Linear(  latent_size, M*latent_size), activation, \
                nn.Linear(M*latent_size, M*latent_size), activation,\
                nn.Linear(M*latent_size, M*latent_size), activation, \
                nn.Linear(M*latent_size, M*latent_size), activation, \
                nn.Linear(M*latent_size, M*latent_size))
        
    # Exp. family model     
    def dlnpz_exp(self, z, polynomial=True):
        '''
        --- returns both dz log p(z) and p(z)
        --- up to some multiplicative constant 
        '''
        if polynomial == True:
            c = self.exp_coef
            dlnpz = 0
            lnpz = 0

            for m in range(self.M):
                dlnpz += (m+1)*z**(m) * c[m,:].unsqueeze(0)
                lnpz += z**(m+1) * c[m,:].unsqueeze(0)

            pz = lnpz.sum(dim=1).exp()

            return dlnpz, pz
        else:
            Tz = self.sufficient_stat(z)
            eta = self.natural_param 
            lnpz = torch.mm(Tz, eta).sum()
            dlnpz = grad(lnpz, z, retain_graph=True)[0]
        
            return dlnpz, lnpz.exp()
            
        
    def encode(self, x):
        h1 = self.enc(x)
        mu_z = self.enc1(h1)
        logvar_z = self.enc2(h1)
        
        return mu_z, logvar_z 
    
    def decode(self, z):
        h1 = self.dec(z)
        x_hat = torch.sigmoid(h1)

        return x_hat
    
    def forward(self, x, detach=False):

        # encode 
        mu_z, logvar_z = self.encode(x) # input of the encoder 
        std_z = (0.5*logvar_z).exp() # std 
        q0 = torch.distributions.normal.Normal(mu_z, std_z) # dist. of epsilon N(0,1)

        # reparameterization trick 
        z = mu_z + std_z * torch.randn_like(std_z) # z ~ q(z|x)

        if self.flow:
            # sample from the new latent 
            # this is a transformation of the original Gaussian
            z = self.flow.sample(noise = z)
        
        # decode 
        x_hat = self.decode(z)
        
        if self.Fisher is True:

            if self.flow:
                dlnqzx = grad(self.flow.log_probs(z).sum(), x, create_graph=True)[0]
                dlnqzz = grad(self.flow.log_probs(z).sum(), z, create_graph=True)[0]
            else:
                dlnqzx = grad(q0.log_prob(z).sum(), x, create_graph=True)[0] # d/dx log q(z|x)
                dlnqzz = grad(q0.log_prob(z).sum(), z, create_graph=True)[0] # d/dz log q(z|x)

            stability = 0.5* dlnqzx.pow(2).sum() # stability term 
            pxz = torch.distributions.normal.Normal(x_hat, 1.0) # p(x|z)
            lnpxz = pxz.log_prob(x) # log p(x|z)
            dlnpxz = grad(lnpxz.sum(), z, retain_graph=True)[0] # d/dz log p(x|z)
            
            if self.exp_family is True:
                dlnpz, _ = self.dlnpz_exp(z) # Exp. family prior 
            else:
                dlnpz = -z # Gaussian prior 

            if detach:
                fisher_div = 0.5*(dlnqzz - dlnpz.detach() - dlnpxz).pow(2).sum() # Fisher div. with one sample from q(z|x)
            else:
                fisher_div = 0.5*(dlnqzz - dlnpz - dlnpxz).pow(2).sum() # Fisher div. with one sample from q(z|x)
            
            return x_hat, fisher_div, stability, mu_z.mean(0), std_z.mean(0)
        
        else:

            pz = torch.distributions.normal.Normal(0., 1.) # prior dist. 
            KL = q0.log_prob(z).sum() - pz.log_prob(z).sum() # KL[q(z|x) || p(z)]
            
            return x_hat, KL 
    
    # the VAE loss function 
    def loss(self, x, output):
        
        if self.Fisher is True:
            x_hat, fisher_div, stability = output 
            MSE = 0.5*(x-x_hat).pow(2).sum()
            loss = fisher_div + MSE + stability 
        else:
            x_hat, KL = output 
            MSE = 0.5*(x-x_hat).pow(2).sum()
            # BCE = F.binary_cross_entropy(x_hat, x.detach(), reduction='sum')
            loss = KL + MSE 

        return loss / x.shape[0], MSE / x.shape[0] 

# TODO remove some of the hard coding from here
class Upsample2d(nn.Module):
    def forward(self, x):
        return F.interpolate(x, scale_factor=2)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Unflatten(nn.Module):
    def __init__(self, im_size=2):
        super(Unflatten, self).__init__()
        self.im_size = im_size

    def forward(self, x):
        return x.view(x.size(0), -1, self.im_size, self.im_size)

def get_conv_nets(latent_size, width=128, in_channels=3, fs=3, act=nn.LeakyReLU(), n_layers=4, pooling = nn.AvgPool2d(2), sigmoid = False):

    padding = math.floor(fs/2)

    enc_modules = [nn.Conv2d(in_channels, width, fs, padding = padding), nn.ReLU(), nn.MaxPool2d(2)]
    dec_modules = [nn.Linear(latent_size, width *4*4*4), Unflatten()]

    for i in range(1, n_layers):

        enc_modules += [nn.Conv2d(width * 2 **(i - 1), width * 2 ** i, fs, padding = padding),
                act,
                pooling]

    for i in range(n_layers, 0, -1):

        dec_modules += [nn.Conv2d(width * 2 ** i , width * 2 ** i, fs, padding = padding),
            Upsample2d(),
            nn.Conv2d(width * 2 ** i, width * 2 ** (i - 1), fs, padding = padding),
            nn.BatchNorm2d(width * 2 ** (i - 1)),
            act]

    enc_modules.append(Flatten())
    dec_modules += [Upsample2d(),
            nn.Conv2d(width, in_channels, fs, padding = padding)]

    if sigmoid:
        dec_modules.append(nn.Sigmoid())

    conv_encoder = nn.Sequential( * enc_modules)
    conv_decoder = nn.Sequential( * dec_modules)

    return conv_encoder, conv_decoder
