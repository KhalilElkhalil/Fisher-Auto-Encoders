import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision
import torchvision.transforms as transforms
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
import copy 
import time
import os

import hamiltorch

import fid_api

from nets import VAE

import svgd

import flows

# set up device 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = '/home/ali/Documents/Data/'

# plotting images 
def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

# load data
batch_size = 128
data_name  = 'celeba'

train_loader = None
test_loader  = None
valid_loader  = None


# optimizer
def make_optimizer(optimizer_name, model, flow = None, **kwargs):

    if flow:
        params = [{'params': flow.parameters(), 'lr': kwargs['lr'] / kwargs['flow_scale']},
                {'params': list(set(model.parameters()) - set(flow.parameters())), 'lr': kwargs['lr']}]
    else:
        params = [{'params': model.parameters(), 'lr': kwargs['lr']}]

    if optimizer_name=='Adam':
        optimizer = optim.Adam(params, betas=[0.9, 0.999])
    elif optimizer_name=='SGD':
        optimizer = optim.SGD(model.parameters(),lr=kwargs['lr'],momentum=kwargs['momentum'], weight_decay=kwargs['weight_decay'])
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(),lr=kwargs['lr'], momentum=0.9)
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer

# scheduler
def make_scheduler(scheduler_name, optimizer, **kwargs):
    if scheduler_name=='MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=kwargs['milestones'],gamma=kwargs['factor'])
    elif scheduler_name=='exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=.9998)
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler

# training parameters
optimizer_name = 'Adam'
scheduler_name = 'MultiStepLR'
num_epochs = 20
lr = 1e-4
device = torch.device(device)
flow = False
n_prior_update = 5
flow_scale = 1

if data_name == 'celeba':
    conv = {'width' : 64}
else:
    conv = None

print('lr: {}'.format(lr))
print('epochs: {}'.format(num_epochs))
print('batch_size: {}'.format(batch_size))
print('n_prior_update: {}'.format(n_prior_update))
print('flow_scale: {}'.format(flow_scale))
print(conv)

# VAE
latent_size = 64 
if flow:
    flow_width  = 256
    flow_layers = 8
    modules = []
    mask = torch.arange(0, latent_size) % 2
    mask = mask.to("cuda:0" if torch.cuda.is_available() else "cpu")

    for _ in range(flow_layers):
        modules += [
            flows.LUInvertibleMM(latent_size),
            flows.CouplingLayer(
                latent_size, flow_width, mask, 0,
                s_act='tanh', t_act='relu')
        ]
        mask = 1 - mask

    flow_net = flows.FlowSequential(*modules).to("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    flow_net = None


# sample from the prior then decode to generate new images 
def sample_images(local_vae, file_prefix, flow, epoch, mu, std, img_size,savepath):
    if flow:
        x_hat_fisher = svgd.generate_images_nosvgd(model=local_vae, img_size= img_size, num_samples=100,mu=mu,std=std)
        plt.figure()
        show(make_grid(x_hat_fisher[0:64], padding=0, normalize=True))
        plt.title('Generated data (exp. prior) flow={}'.format(flow))
        plt.savefig('{}generated_samples_epoch{}.png'.format(file_prefix,epoch))
        plt.close('all')
    else:
        x_hat_fisher = svgd.generate_images(model=local_vae.cpu(), img_size = img_size, num_samples=100, n_iter=50000, stepsize=1e-4)
        plt.figure()
        show(make_grid(x_hat_fisher[0:64], padding=0, normalize=True))
        plt.title('Generated data (exp. prior) flow={}'.format(flow))
        plt.savefig(os.path.join(savepath, '{}generated_samples_svgd={}_epoch{}.png'.format(file_prefix,local_vae.exp_family,epoch)))
        plt.close('all')
        local_vae.to(device)

    return x_hat_fisher

def sample_hmcmc(local_vae, file_prefix, flow, epoch):

    hamiltorch.set_random_seed(123)
    params_init = torch.ones(latent_size + 1)
    params_init[0] = 0.
    step_size = 0.3093
    num_samples = 500
    L = 25
    omega=100
    threshold = 1e-3
    softabs_const=10**6

    params_hmc = hamiltorch.sample(log_prob_func=model.dlnpz_exp, params_init=params_init, num_samples=num_samples,
                                   step_size=step_size, num_steps_per_sample=L)

    samples = local_vae.decode(params_hmc)

    plt.title('Generated data (exp. prior) flow={}'.format(flow))
    plt.savefig('{}generated_samples_hmc={}_epoch{}.png'.format(file_prefix,local_vae.exp_family,epoch))
    plt.close('all')
    return samples


torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import datasets
if data_name == 'mnist':
    train_loader, test_loader = datasets.get_mnist(batch_size = batch_size, dataset_directory = data_path)

elif data_name == 'celeba':
    data_path = '/home/ali/ext1/Data/'
    train_loader, valid_loader, test_loader = datasets.get_celeba(batch_size = batch_size, dataset_directory = data_path)

# Train the VAE
loader = train_loader
data_shape = 0 

sigma2 = torch.linspace(0.01, 2, 10)
all_fid = [0]*sigma2.shape[0]
all_mse = [0]*sigma2.shape[0]
Fisher = True
if Fisher:
    savepath = 'results/fisher_noise'
else:
    savepath = 'results/kl_noise'

if not os.path.exists(savepath):
    os.makedirs(savepath)

for noise_idx in range(sigma2.shape[0]):
    best_FID = 100000

    s = sigma2[noise_idx] 

    local_vae = VAE(feature_size=784, latent_size=latent_size, M=4, conv=conv, flow=flow_net, Fisher=Fisher, exp_family=False).to(device)
    optimizer = make_optimizer(optimizer_name, local_vae, flow = flow_net, lr=lr, weight_decay=0, momentum=0.01, flow_scale=flow_scale)
    scheduler = make_scheduler(scheduler_name, optimizer, milestones=[25, 50, 70, 90], factor=0.99)

    file_prefix = 'fisherl={}_latent={}_sigma={}'.format(local_vae.Fisher,latent_size,s)

    for epoch in tqdm(range(num_epochs+1)):
        loss_epoch = 0
        mse_epoch = 0 
        for iter_, (data, _) in enumerate(loader):

            data_shape = data.shape

            # zero grad
            optimizer.zero_grad()

            data = data + torch.sqrt(s) * torch.randn_like(data)
            # forward pass
            if data_name == 'mnist':
                data = Variable(data.reshape(data.shape[0], 784), requires_grad=True).to(device)
            else:
                data = Variable(data, requires_grad=True).to(device)


            if iter_ % n_prior_update :
                output = local_vae.forward(data,detach=False)
            else:
                output = local_vae.forward(data)

            loss, mse = local_vae.loss(data, output)
            mu  =0  
            std =1 
            loss_epoch += loss.item()
            mse_epoch += mse.item()

            plt.figure()
            show(make_grid(output[0][0:64, :].view(-1,data_shape[1],data_shape[2],data_shape[3]).cpu().detach(), padding=0, normalize=True))
            plt.savefig(os.path.join(savepath, '{}train_samps_poly.png'.format(file_prefix)))
            plt.close('all')


            # backward pass
            loss.backward()

            # update parameters
            optimizer.step()
        scheduler.step()

        # print loss at the end of every epoch
        print('Epoch : ', epoch, ' | Loss VAE: {:.4f} | Loss MSE: {:.4f}'.format(loss_epoch / len(loader), mse_epoch / len(loader)), ' | lr : ', optimizer.param_groups[0]['lr'])
        if epoch % 1 == 0 :
            x_hat_fisher = sample_images(local_vae, file_prefix, flow, epoch, mu, std, img_size = data_shape, savepath=savepath)

            # compute FID score
            fid_api.initialize_fid(valid_loader, sample_size=200)
            score_fisher = fid_api.fid_images(x_hat_fisher)
            print('FID : {}'.format(score_fisher))

            # compute MSE
            '''
            mse_total = 0 
            for batch_v, _ in valid_loader:
                data = Variable(batch_v, requires_grad=True).to(device)
                output = local_vae.forward(data)
                mse_total += F.mse_loss(output[0], batch_v.to('cuda:0'))

                print(' Valid MSE : {} '.format(mse_total / len(valid_loader)))
            '''

            if score_fisher < best_FID:
                print(score_fisher)
                best_FID = score_fisher.copy()
                all_fid[noise_idx] = (s, best_FID)
                #all_mse[noise_idx] = (s, mse_total)
                model_path = os.path.join(savepath, 'model_best_noise{}_fisher{}.pth'.format(noise_idx, local_vae.Fisher))
                torch.save(local_vae.state_dict(), model_path)

            # save the model
            #torch.save(local_vae.state_dict(), 'model_epoch_{}_{}_fisher{}_noise{}.pth'.format(epoch,data_name,local_vae.Fisher,s))

print(all_fid)
print(all_mse)
import pickle
with open(os.path.join(savepath, 'fid_fisher{}.pkl'.format(local_vae.Fisher)), 'wb') as f:
    pickle.dump(all_fid, f)
#with open(os.path.join(savepath, 'mse_fisher{}.pkl'.format(local_vae.Fisher)), 'wb') as f:
#    pickle.dump(all_mse, f)


