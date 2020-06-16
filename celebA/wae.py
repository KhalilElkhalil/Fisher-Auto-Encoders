# code from https://github.com/neale/Wasserstein-Autoencoder/blob/master/wae_celeba.py
#
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision import transforms, datasets
import wae_utils
import fid_api 
import datasets


cuda = True
cnt = 0
lr = 1e-4
out_dir = "out_aae3"
batch_size = 128

nc = 3 # number of channels
nz = 64 # size of latent vector
ngf = 64 # decoder (generator) filter factor
ndf = 64 # encoder filter factor
h_dim = 128 # discriminator hidden size
lam = 1 # regulization coefficient


transform = transforms.Compose([
        transforms.CenterCrop(160),
        transforms.Scale(64),
        transforms.ToTensor(),
         #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

#dataset = datasets.ImageFolder('/data0/images/celeba', transform)
#data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20, drop_last=True)



class Upsample2d(nn.Module):
    def forward(self, x):
        return F.interpolate(x, scale_factor=2)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        conv = {'width' : 64}
        _, self.dec = get_conv_nets(latent_size = nz, **conv)

    def forward(self, x):
        nz = 64
        x = self.dec(x.view(-1,nz))
        #for layer in self.main:
        #    x = layer(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        conv = {'width' : 64}

        self.enc, _ = get_conv_nets(latent_size = nz, **conv)

        self.enc1 = nn.Linear(conv['width']*8**3, nz)

    def forward(self, x):
        #for layer in self.main:
        #    x = layer(x)
        h1 = self.enc(x)
        z = self.enc1(h1)

        return z

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = [
            nn.Linear(nz, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        ]
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)
    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x

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
                    nn.BatchNorm2d(width * 2 ** i), 
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

Q = Encoder()
P = Decoder()
D = Discriminator()


if os.path.exists("Q_latest.pth"):
    Q = torch.load("Q_latest.pth")
if os.path.exists("P_latest.pth"):
    P =torch.load("P_latest.pth")
if os.path.exists("D_latest.pth"):
    D = torch.load("D_latest.pth")

if cuda:
    Q = Q.cuda()
    P = P.cuda()
    D = D.cuda()

def reset_grad():
    Q.zero_grad()
    P.zero_grad()
    D.zero_grad()

Q_solver = optim.Adam(Q.parameters(), lr=lr)
P_solver = optim.Adam(P.parameters(), lr=lr)
D_solver = optim.Adam(D.parameters(), lr=lr*0.1)

def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

best_FID = 1000000

torch.manual_seed(1)
torch.cuda.manual_seed(1)
data_path = '/home/ali/ext1/Data/'
train_loader, valid_loader, test_loader = datasets.get_celeba(batch_size = batch_size, dataset_directory = data_path)

for epoch in range(100):

    for batch_idx, batch_item in enumerate(train_loader):
        #X = sample_X(mb_size)
        """ Reconstruction phase """
        X = Variable(batch_item[0])
        if cuda:
            X = X.cuda()

        z_sample = Q(X)

        X_sample = P(z_sample)
        recon_loss = F.mse_loss(X_sample, X)

        recon_loss.backward()
        P_solver.step()
        Q_solver.step()
        reset_grad()

        """ Regularization phase """
        # Discriminator
        for _ in range(5):
            z_real = Variable(torch.randn(batch_size, nz))
            if cuda:
                z_real = z_real.cuda()

            z_fake = Q(X).view(batch_size,-1)

            D_real = D(z_real)
            D_fake = D(z_fake)

            #D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))
            D_loss = -(torch.mean(D_real) - torch.mean(D_fake))

            D_loss.backward()
            D_solver.step()

            # Weight clipping
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

            reset_grad()

        # Generator
        z_fake = Q(X).view(batch_size,-1)
        D_fake = D(z_fake)

        #G_loss = -torch.mean(torch.log(D_fake))
        G_loss = -torch.mean(D_fake)

        G_loss.backward()
        Q_solver.step()
        reset_grad()

        if batch_idx % 100 == 0:
            print('Epoch{} Iter-{}; D_loss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'.format(epoch, batch_idx, D_loss.item(), G_loss.item(), recon_loss.item()))

        # Print and plot every now and then
        if batch_idx == 0:

            z_real = z_real.unsqueeze(2).unsqueeze(3) # add 2 dimensions
            if cnt % 2 == 0:
                samples = P(z_real) # Generated
            else:
                samples = P(z_real) # Generated
                #samples = X_sample # Reconstruction
            #samples = X_sample
            # compute FID score
            fid_api.initialize_fid(valid_loader, sample_size=batch_size)
            score_fisher = fid_api.fid_images(samples)
            print('FID: {}'.format(score_fisher))

            if score_fisher < best_FID:
                best_FID = score_fisher.cloney()
                torch.save(Q,"Q_best_{:.1f}.pth".format(score_fisher))
                torch.save(P,"P_latest_{:.1f}.pth".format(score_fisher))
                torch.save(D,"D_latest_{:.1f}.pth".format(score_fisher))

            if cuda:
                samples = samples.cpu()
            show(make_grid(samples[:64].detach(), padding=0, normalize=True))

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            plt.savefig('{}/{}.png'
                        .format(out_dir,str(cnt).zfill(3)), bbox_inches='tight')
            cnt += 1
            plt.close('all')
