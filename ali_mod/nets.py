import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import grad

import math

import flows

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

# VAE class 
class VAE(nn.Module):
    def __init__(self, feature_size, latent_size, conv=None, flow=None, exp_family=True, M=5, Fisher=True):
        super(VAE, self).__init__()
        self.latent_size = latent_size 

        if conv:
            ndf = conv['width']
            ngf = conv['width']
            nc = 3
            if Fisher:
                self.enc= nn.Sequential(
                    # input is (nc) x 64 x 64
                    nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf) x 32 x 32
                    nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                    #nn.BatchNorm2d(ndf * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*2) x 16 x 16
                    nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                    #nn.BatchNorm2d(ndf * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*4) x 8 x 8
                    nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                    #nn.BatchNorm2d(ndf * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*8) x 4 x 4
                    #nn.Conv2d(ndf * 8, latent_size, 4, 1, 0, bias=False),
                    View((-1, conv['width']*8*8*2)),                                 # B, 1024*4*4
                )
            else:
                self.enc= nn.Sequential(
                    # input is (nc) x 64 x 64
                    nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf) x 32 x 32
                    nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*2) x 16 x 16
                    nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*4) x 8 x 8
                    nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*8) x 4 x 4
                    #nn.Conv2d(ndf * 8, latent_size, 4, 1, 0, bias=False),
                    View((-1, conv['width']*8*8*2)),                                 # B, 1024*4*4
                )
            self.dec= nn.Sequential(
                # input is Z, going into a convolution
                View((-1, latent_size, 1, 1)),                                 # B, 1024*4*4
                nn.ConvTranspose2d( latent_size, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.ConvTranspose2d(    ngf,      3, 4, 2, 1, bias=False),
                nn.Tanh()
            )
            '''

            ndf = conv['width']
            nc = 3
            nz = latent_size
            ngf = conv['width']

            self.enc= nn.Sequential(
                # input is (nc) x 28 x 28
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 14 x 14
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 7 x 7
                nn.Conv2d(ndf * 2, ndf * 4, 3, 4, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 4 x 4
                nn.Conv2d(ndf * 4, 1024, 4, 1, 0, bias=False),
                # nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.2, inplace=True),
                # nn.Sigmoid()
                View((-1, conv['width']*8)),                                 # B, 1024*4*4
            )        
            self.dec= nn.Sequential(
                #nn.Linear(latent_size, conv['width']*4*8*8),
                View((-1, nz, 1, 1)),                               # B, 1024,  8,  8
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )
            self.enc1 = nn.Linear(conv['width']*8, latent_size)
            self.enc2 = nn.Linear(conv['width']*8, latent_size)
            '''
            self.enc1 = nn.Linear(conv['width']*8*8*2, latent_size)
            self.enc2 = nn.Linear(conv['width']*8*8*2, latent_size)

            #self.enc1 = nn.Linear(conv['width']*8, latent_size)
            #self.enc2 = nn.Linear(conv['width']*8, latent_size)
            self.enc, self.dec = get_conv_nets(latent_size = latent_size, **conv)
            self.enc1 = nn.Linear(conv['width']*8**3, latent_size)
            self.enc2 = nn.Linear(conv['width']*8**3, latent_size)

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
        x_hat = self.dec(z)
        #x_hat = F.tanh(h1)

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
            
            return x_hat, fisher_div, stability
        
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
    def __init__(self, im_size=8):
        super(Unflatten, self).__init__()
        self.im_size = im_size

    def forward(self, x):
        return x.view(x.size(0), -1, self.im_size, self.im_size)

def get_conv_nets(latent_size, width=64, in_channels=3, fs=5, act_enc=nn.LeakyReLU(), act_dec=nn.ReLU(), n_layers=4, pooling=nn.AvgPool2d(2), tanh = True):

    padding = math.floor(fs/2)

    enc_modules = [nn.Conv2d(in_channels, width, fs, padding = padding), act_enc, pooling]
    dec_modules = [nn.Linear(latent_size, width *8*8*8), Unflatten()]

    for i in range(1, n_layers):

        if i == n_layers - 1:
            enc_modules += [nn.Conv2d(width * 2 **(i - 1), width * 2 ** i, fs, padding = padding),
                    act_enc]
        else:
            enc_modules += [nn.Conv2d(width * 2 **(i - 1), width * 2 ** i, fs, padding = padding),
                    #nn.BatchNorm2d(width * 2 ** i), 
                    act_enc,
                    pooling]

    for i in range(n_layers-1, 0, -1):

        dec_modules += [Upsample2d(),
            nn.Conv2d(width * 2 ** i, width * 2 ** (i - 1), fs, padding = padding),
            nn.BatchNorm2d(width * 2 ** (i - 1)),
            act_dec]

    enc_modules.append(Flatten())
    dec_modules += [nn.Conv2d(width, in_channels, fs, padding = padding)]

    if tanh:
        dec_modules.append(nn.Tanh())

    conv_encoder = nn.Sequential( * enc_modules)
    conv_decoder = nn.Sequential( * dec_modules)

    print(conv_encoder)
    print(conv_decoder)

    return conv_encoder, conv_decoder
