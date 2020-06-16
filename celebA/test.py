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

import fid_api

from nets import VAE

import svgd

import flows

import datasets

# set up device 
device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'

data_path = '/home/ali/Documents/Data/'

# plotting images 
def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

# load data
batch_size = 128
data_name  = 'celeba'

device = torch.device(device)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
n_prior_update = 10
flow_scale = 10

if data_name == 'celeba':
    conv = {'width' : 64}
else:
    conv = None

# VAE
latent_size = 64

TEST_TYPE = 'vae'
corruption = 'square'
calc_fid = False

if TEST_TYPE == 'vae':
    net_path = 'results/kl_gauss_zay_valid/model_epoch_56_celeba_fisherFalse_fid124.4.pth'
    Fisher = False
    exp_family = False
    M = 8 
    flow = False
elif TEST_TYPE == 'fisher-exp':
    
    net_path = 'model_epoch_89_celeba_fisherTrue_fid117.1.pth'
    Fisher = True
    exp_family = True
    M = 4
    flow = False
elif TEST_TYPE == 'fisher-flow':

    net_path = 'results/fisher_exp_flow_zay_valid/model_epoch_146_celeba_fisherTrue_fid119.8.pth'
    Fisher = True
    exp_family = True
    M = 4
    flow = True
elif TEST_TYPE == 'fisher-gauss':
    net_path = 'results/fisher_gauss_zay_valid/model_epoch_57_celeba_fisherTrue_fid123.6.pth'
    Fisher = True
    exp_family = False
    M = 8
    flow = False 

if flow:
    flow_width = 128
    flow_layers =  8
    modules = []
    mask = torch.arange(0, latent_size) % 2
    mask = mask.to(device)

    for _ in range(flow_layers):
        modules += [
            flows.LUInvertibleMM(latent_size),
            flows.CouplingLayer(
                latent_size, flow_width, mask, 0,
                s_act='tanh', t_act='relu')
        ]
        mask = 1 - mask

    flow_net = flows.FlowSequential(*modules).to(device)
else:
    flow_net = None



local_vae = VAE(feature_size=784, latent_size=latent_size, M=M, conv=conv, flow=flow_net, Fisher=Fisher, exp_family=exp_family).to(device)
local_vae.load_state_dict(torch.load(net_path))

file_prefix = 'TESTING_flow={}_latent={}_exp={}_fisher={}'.format(flow,latent_size,local_vae.exp_family,local_vae.Fisher)

# sample from the prior then decode to generate new images 
def sample_images(local_vae, file_prefix, flow, mu, std, num_samples, img_size):
    if flow:
        x_hat_fisher = svgd.generate_images_nosvgd(model=local_vae, img_size= img_size, num_samples=num_samples, mu=mu, std=std)
        plt.figure()
        show(make_grid(x_hat_fisher[0:64], padding=0, normalize=True))
        plt.title('Generated data (exp. prior) flow={}'.format(flow))
        plt.savefig('{}generated_samples.png'.format(file_prefix))
        plt.close('all')
    else:
        x_hat_fisher = svgd.generate_images(model=local_vae.cpu(), img_size = img_size, num_samples=num_samples, n_iter=50000, stepsize=1e-4)
        plt.figure()
        show(make_grid(x_hat_fisher[0:64], padding=0, normalize=True))
        plt.title('Generated data (exp. prior) flow={}'.format(flow))
        plt.savefig('{}generated_samples_svgd={}.png'.format(file_prefix,local_vae.exp_family))
        plt.close('all')
        local_vae.to(device)

    return x_hat_fisher

if corruption == 'gauss':
    sigma2 = torch.linspace(0.01,2,10)
elif corruption == 'binary' or corruption == 'square':
    sigma2 = torch.linspace(0.01,1,10)

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

data_shape = 0 
loss_epoch = 0
mse_epoch = 0 

mse_losses = [0]*sigma2.shape[0]

save_dict = {}

for noise_idx in range(sigma2.shape[0]):
    mse_idx = []
    for iter_, (data, _) in enumerate(test_loader):
        with torch.no_grad():
            s =  sigma2[noise_idx]
            '''
            print('NOISE IS FIXED, EDIT IF NECESSARY')
            s =  sigma2[-2]
            '''

            data_shape = data.shape

            in_data = data.clone()

            # save output plots
            clean_data = Variable(in_data, requires_grad=True).to(device)
            if corruption == 'gaussian':
                corrupt_data = Variable(in_data + s * torch.randn_like(data), requires_grad=True).to(device)
            elif corruption == 'binary':
                mask = np.random.choice(2, (data.shape[0], data.shape[-1], data.shape[-2]), replace=True, p=[s, 1-s])
                mask = np.expand_dims(mask, axis=1)
                mask = np.repeat(mask,repeats=3, axis=1)
                #mask = mask.reshape(data.shape[0], 3, 64, 64)
                mask = torch.tensor(mask)
                corrupt_data = Variable(in_data * mask)
            elif corruption == 'square':
                mask = torch.zeros((data.shape[0], data.shape[1], int(data.shape[2] * s), int(data.shape[3] * s)))
                corrupt_data = in_data.clone()
                corrupt_data[:,:,:mask.shape[2], :mask.shape[3]] = mask
                corrupt_data = Variable(corrupt_data)


            output = local_vae.forward(clean_data, test=True)
            corrupt_output = local_vae.forward(corrupt_data, test=True)


            if iter_ == 0 and noise_idx == 8: 
                n_row = 8
                grid_ims = []
                for i in range(int(n_row/2)):
                    grid_ims.append(in_data[i*n_row:(i+1)*n_row])
                    grid_ims.append(output[0][i*n_row:(i+1)*n_row])

                merged = torch.cat(grid_ims)
                corrupt_data_norm_plot = 2* (corrupt_data - corrupt_data.min() )/ (corrupt_data.max() -corrupt_data.min()) - 1

                grid_ims_c = []
                for i in range(int(n_row/2)):
                    grid_ims_c.append(in_data[i*n_row:(i+1)*n_row])
                    grid_ims_c.append(corrupt_output[0][i*n_row:(i+1)*n_row])
                merged_corrupt = torch.cat(grid_ims_c)

                clean_grid = make_grid(merged.cpu().detach(), padding=0, normalize=True, nrow=n_row)
                corrupt_grid = make_grid(merged_corrupt.cpu().detach(), padding=0, normalize=True, nrow=n_row)
                save_dict['corrupt'] = corrupt_grid.clone()
                save_dict['clean'] = clean_grid.clone()

                with open('{}_{}_noise_clean_dep.pkl'.format(TEST_TYPE,corruption),'wb') as f:
                    import pickle
                    pickle.dump(save_dict, f)

                if calc_fid:
                    # calc FID
                    num_samples = 1000
                    mu = 0 
                    std = 1
                    fisher_scores = []
                    for i in range(10):
                        x_hat_fisher = sample_images(local_vae, file_prefix, flow, mu, std, img_size = data_shape, num_samples = num_samples)
                        generated_grid = make_grid(x_hat_fisher[0:100].cpu().detach(), padding=0, normalize=True, nrow=10)
                        fid_api.initialize_fid(test_loader, sample_size=num_samples)
                        score_fisher = fid_api.fid_images(x_hat_fisher)
                        print('score fisher {}'.format(score_fisher))
                        fisher_scores.append(score_fisher)

                    save_dict['generated'] = generated_grid.clone()
                    save_dict['fisher_scores'] = fisher_scores
                    with open('{}_figs_fid.pkl'.format(TEST_TYPE),'wb') as f:
                        import pickle
                        pickle.dump(save_dict, f)

            mse_idx.append(F.mse_loss(corrupt_output[0], in_data).item())
    #print('MSE: {}'.format(F.mse_loss(output[0], data).item()))
    mse_losses[noise_idx] = mse_idx
    print('Corrupt MSE idx {}: {}'.format(noise_idx, mse_idx))
save_dict['mse_losses'] = mse_losses
with open('{}_{}_dep.pkl'.format(TEST_TYPE,corruption),'wb') as f:
    import pickle
    pickle.dump(save_dict, f)
