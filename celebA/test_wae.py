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
cuda = False
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


Q = torch.load("results/wae_github_zay_valid/Q_best_102.2.pth")
P = torch.load("results/wae_github_zay_valid/P_latest_102.2.pth")
D = torch.load("results/wae_github_zay_valid/D_latest_102.2.pth")

if cuda:
    Q = Q.cuda()
    P = P.cuda()
    D = D.cuda()
else:
    Q = Q.cpu()
    P = P.cpu()
    D = D.cpu()

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

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

corruption = 'square'
if corruption == 'gauss':
    sigma2 = torch.linspace(0.01,2,10)
elif corruption == 'binary' or corruption == 'square':
    sigma2 = torch.linspace(0.01,1,10)

all_mse = [0]*sigma2.shape[0]
all_fid = []
calc_FID = False
import datasets

data_path = '/home/ali/ext1/Data/'
_, _, test_loader = datasets.get_celeba(batch_size = batch_size, dataset_directory = data_path, dataloader_workers=1)
save_dict = {}
for noise_idx in range(sigma2.shape[0]):
    #X = sample_X(mb_size)
    """ Reconstruction phase """
    mse_loss = []
    for batch_idx, batch_item in enumerate(test_loader):
        with torch.no_grad():
            s = sigma2[noise_idx]
            '''
            print('NOISE IS FIXED, CHANGE IF NECESSARY')
            s = sigma2[-2]
            '''
            X = Variable(batch_item[0])
            if corruption == 'gauss':
                X_corrupt = Variable(batch_item[0] + s * torch.randn_like(batch_item[0]))
            elif corruption == 'binary':
                mask = np.random.choice(2, (batch_item[0].shape[0], batch_item[0].shape[-1], batch_item[0].shape[-2]), replace=True, p=[s, 1-s])
                mask = np.expand_dims(mask, axis=1)
                mask = np.repeat(mask,repeats=3, axis=1)
                mask = torch.tensor(mask)
                X_corrupt = Variable(batch_item[0] * mask)
            elif corruption == 'square':
                mask = torch.zeros((batch_item[0].shape[0], batch_item[0].shape[1], int(batch_item[0].shape[2] * s), int(batch_item[0].shape[3] * s)))
                corrupt_x = batch_item[0].clone()
                corrupt_x[:,:,:mask.shape[2], :mask.shape[3]] = mask
                X_corrupt = Variable(corrupt_x)

            if cuda:
                X = X.cuda()
                X_corrupt = X_corrupt.cuda()

            z_sample = Q(X)
            z_corrupt_sample = Q(X_corrupt)

            X_sample = P(z_sample)
            X_sample_corrupt = P(z_corrupt_sample)
            mse_loss.append(F.mse_loss(X_sample_corrupt, X).item())

            X_corrupt_norm_plot = 2* (X_corrupt - X_corrupt.min() )/ (X_corrupt.max() - X_corrupt.min()) - 1


            if batch_idx == 0 and noise_idx == 8:
                grid_ims= []

                n_row = 8 
                for i in range(int(n_row/2)):
                    grid_ims.append(X[i*n_row:(i+1)*n_row])
                    grid_ims.append(X_sample[i*n_row:(i+1)*n_row])
                merged= torch.cat(grid_ims)
                clean_grid = make_grid(merged.cpu().detach(), padding=0, normalize=True, nrow=n_row)

                grid_ims_c = []
                for i in range(int(n_row/2)):
                    grid_ims_c.append(X[i*n_row:(i+1)*n_row])
                    grid_ims_c.append(X_sample_corrupt[i*n_row:(i+1)*n_row])
                merged_corrupt = torch.cat(grid_ims_c)

                corrupt_grid = make_grid(merged_corrupt.cpu().detach(), padding=0, normalize=True, nrow=n_row)

                save_dict['corrupt'] = corrupt_grid.clone()
                save_dict['clean'] = clean_grid.clone()
                n_samples = 1000
                with open('wae_{}_noise_clean_dep.pkl'.format(corruption), 'wb') as f:
                    import pickle
                    pickle.dump(save_dict, f)

                if calc_FID:
                    for _ in range(10):
                
                        z_real = Variable(torch.randn(n_samples, nz))
                        generated = P(z_real)

                        fid_api.initialize_fid(test_loader, sample_size=n_samples)
                        score_fisher = fid_api.fid_images(generated)
                        all_fid.append(score_fisher)
                        print(score_fisher)
                        generated_grid = make_grid(generated[0:100].cpu().detach(), padding=0, normalize=True, nrow=10)
                    save_dict['fisher_scores'] = all_fid
                    save_dict['generated'] = generated_grid.clone()
                    with open('wae_figs_fid.pkl', 'wb') as f:
                        import pickle
                        pickle.dump(save_dict, f)

    all_mse[noise_idx] =  mse_loss
    print(mse_loss)

save_dict['mse'] = all_mse

with open('wae_{}_dep.pkl'.format(corruption),'wb') as f:
    import pickle
    pickle.dump(save_dict, f)


