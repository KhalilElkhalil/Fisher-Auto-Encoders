from __future__ import division
from __future__ import print_function
import torch
import torch.utils.data
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import random
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from scipy.stats import kde
from torchvision.utils import make_grid
from torch.autograd import grad
import numpy as np
import numpy.linalg as la
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import math
import svgd
import datasets
import fid_api
# from torchsummary import summary
# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# set up device 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
# Root directory for dataset
dataroot = '/home/ali/ext1/Data/celeba'

# Number of workers for dataloader
workers = 2
# Batch size during training
batch_size = 100
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Learning rate for optimizers
lr = 0.0005

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = torchvision.datasets.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

# Create the dataloader
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

data_path = '/home/ali/ext1/Data/'
train_loader, test_loader = datasets.get_celeba(batch_size = batch_size, dataset_directory = data_path)
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
# VAE class for celeba 
class VAE(nn.Module):
    def __init__(self, nz):
        super(VAE, self).__init__()
        self.have_cuda = False
        self.nz = nz
        self.Fisher = False
        self.latent_size = nz
        self.encoder = nn.Sequential(

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

        )

        self.decoder = nn.Sequential(

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

        self.fc1 = nn.Linear(1024, 512)

        self.fc21 = nn.Linear(512, nz)

        self.fc22 = nn.Linear(512, nz)
        self.encoder= nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),            # B, 1024,  4,  4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024*4*4)),                                 # B, 1024*4*4
        )
        self.decoder= nn.Sequential(
            nn.Linear(nz, 1024*8*8),                           # B, 1024*8*8
            View((-1, 1024, 8, 8)),                               # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),   # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 1),                       # B,   nc, 64, 64
        )

        self.fc1 = nn.Linear(1024*4*4, 512)

        self.fc21 = nn.Linear(1024*4*4, nz)

        self.fc22 = nn.Linear(1024*4*4, nz)

    def encode(self, x):

        conv = self.encoder(x);

        #h1 = self.fc1(conv.view(-1, 1024))
        h1 = conv

        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        #x_hat = self.decoder(z.reshape(z.shape[0], z.shape[1], 1, 1))
        x_hat = self.decoder(z)

        return x_hat


    def forward(self, x):

        mu, logvar = self.encode(x)

        std = (0.5*logvar).exp()

        z = mu + std*torch.randn_like(std)

        qzx = -0.5*logvar.sum() - 0.5*((z-mu)/std).pow(2).sum()

        dx_logq = grad(qzx, x, create_graph=True)[0]

        stability = 0.5 * dx_logq.pow(2).sum()

        dz_logq = -(z-mu) / std.pow(2)

        x_hat = self.decode(z)

        logpxz = -0.5*(x-x_hat).pow(2).sum()

        dz_logpxz = grad(logpxz, z, retain_graph=True)[0]

        Fisher_div = 0.5*(dz_logq + z - dz_logpxz).pow(2).sum()

        if self.Fisher:
            return x_hat, Fisher_div, stability
        else:
            KL = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return x_hat, KL


num_epochs = 300
lr = 0.0005
vae_celeba = VAE(nz=64).to(device)
optimizer = optim.Adam(vae_celeba.parameters(), lr=lr)


def vae_loss_fisher(x, x_hat, Fisher_div, stability):

# def vae_loss(x, x_hat, KL):

    MSE =  0.5*(x-x_hat).pow(2).sum()

    # loss = KL + MSE 

    loss = Fisher_div + MSE + stability

    return loss

def vae_loss_kl(x, x_hat, KL):

    MSE =  0.5*(x-x_hat).pow(2).sum()

    loss = KL + MSE 
    return loss

def sample_images(local_vae, epoch, img_size, mu=0, std=1, file_prefix=''):
    x_hat_fisher = svgd.generate_images_nosvgd(model=local_vae, img_size= img_size, num_samples=100, mu=mu, std=std)
    plt.figure()
    show(make_grid(x_hat_fisher[0:64], padding=0))
    plt.title('Generated data Fisher = {}'.format(local_vae.Fisher))
    plt.savefig('{}generated_samples_khalil_epoch{}_fisher{}.png'.format(file_prefix,epoch,local_vae.Fisher))
    plt.close('all')
    return x_hat_fisher
# train the VAE on celeba 
best_loss = 1e10
for epoch in tqdm(range(1, num_epochs + 1)):

    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data_shape = data.shape

        data = Variable(data , requires_grad=True).to(device)

        optimizer.zero_grad()

        if vae_celeba.Fisher:

            x_hat, Fisher_div, stability = vae_celeba.forward(data)

            loss = vae_loss_fisher(data, x_hat, Fisher_div, stability)
        else:

            x_hat, KL = vae_celeba.forward(data)

            loss = vae_loss_kl(data, x_hat, KL)

        # backward pass

        loss.backward()

        optimizer.step()

        # update loss 
        train_loss += loss.item()

    print('Epoch : ', epoch, ' ====> Loss VAE: {:.4f}'.format(train_loss / len(train_loader)))

    x_hat_fisher = sample_images(vae_celeba, epoch, img_size = data_shape)

    # compute FID score
    fid_api.initialize_fid(test_loader, sample_size=1000)
    score_fisher = fid_api.fid_images(x_hat_fisher)
    print(score_fisher)


    # save model 
    if train_loss < best_loss:
        torch.save(vae_celeba, 'ckpt_celeba_FVAE_test_fisher{}.pth'.format(vae_celeba.Fisher))
        best_loss = train_loss
