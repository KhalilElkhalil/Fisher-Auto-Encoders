import torch
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.spatial.distance import pdist, squareform
# SVGD sampler to sample from p(x) known up to some constant. 
class SVGD_model():

    def __init__(self):
        pass

    def SVGD_kernel(self, x, h=-1):
        init_dist = pdist(x)
        pairwise_dists = squareform(init_dist)
        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = h ** 2 / np.log(x.shape[0] + 1)

        kernal_xj_xi = np.exp(- pairwise_dists ** 2 / h)
        d_kernal_xi = np.zeros(x.shape)
        for i_index in range(x.shape[0]):
            d_kernal_xi[i_index] = np.matmul(kernal_xj_xi[i_index], x[i_index] - x) * 2 / h

        return kernal_xj_xi, d_kernal_xi

    def update(self, x0, dlnprob, n_iter=5000, stepsize=1e-3, bandwidth=-1, alpha=0.9, debug=False):
        # Check input
        if x0 is None or dlnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')
        
        x = np.copy(x0)

        # adagrad with momentum
        eps_factor = 1e-8
        historical_grad_square = 0
        for iter in range(n_iter):
            if debug and (iter + 1) % 1000 == 0:
                print('iter ' + str(iter + 1))

            kernal_xj_xi, d_kernal_xi = self.SVGD_kernel(x, h=-1)
            current_grad = (np.matmul(kernal_xj_xi, dlnprob(x)) + d_kernal_xi) / x.shape[0]
            if iter == 0:
                historical_grad_square += current_grad ** 2
            else:
                historical_grad_square = alpha * historical_grad_square + (1 - alpha) * (current_grad ** 2)
            adj_grad = current_grad / np.sqrt(historical_grad_square + eps_factor)
            x += stepsize * adj_grad

        return x

# generate images from random input using SVGD 
def generate_images(model, num_samples, img_size,  n_iter=16000, stepsize=1e-4):
    img_size[0] = num_samples
    if model.Fisher is True:
        # SVGD samppling 
        def dlnp(z):
            z = torch.tensor(z, dtype=torch.float, requires_grad=True)
            dlnp, _ = model.dlnpz_exp(z)
            return dlnp.detach().numpy()

        svgd_sampler = SVGD_model()
        z0 = np.random.rand(num_samples, model.latent_size)

        print(n_iter)
        print(stepsize)
        # z0 = 2*torch.rand(num_samples, model.latent_size) - 1
        # z0 = z0.to(device)
        samples = svgd_sampler.update(x0=z0, dlnprob=dlnp, n_iter=n_iter, stepsize=stepsize)

        print(samples.std(0))
        
        # decode samples 
        x_hat = model.decode(torch.tensor(samples, dtype=torch.float))
        x_hat = x_hat.detach()
        x_hat = x_hat.reshape(img_size)
        
    else:
        z = torch.randn(num_samples, model.latent_size)
        x_hat = model.decode(z)
        x_hat = x_hat.detach()
        x_hat = x_hat.reshape(img_size)
        
    return x_hat

# generate images from random input using SVGD
def generate_images(model, num_samples, img_size, n_iter=16000, stepsize=1e-4):

    if model.Fisher is True and model.exp_family is True:
        # SVGD samppling
        def dlnp(z):
            '''
            enforce samples to be confined in [-1, 1]^d
            d : latent size
            '''
            #t0 = z < -1
            #t1 = z > 1
            #if (t0==True).sum() > 0 or (t1==True).sum() > 0:
            #    dlnp = np.zeros_like(z)
            #else:
            z = torch.tensor(z, dtype=torch.float, requires_grad=True)
            dlnp, _ = model.dlnpz_exp(z)
            dlnp = dlnp.detach().numpy()

            return dlnp

        svgd_sampler = SVGD_model()
        z0 = np.random.rand(num_samples, model.latent_size)
        # z0 = 2*torch.rand(num_samples, model.latent_size) - 1
        # z0 = z0.to(device)
        samples = svgd_sampler.update(x0=z0, dlnprob=dlnp, n_iter=n_iter, stepsize=stepsize)

        # decode samples
        x_hat = model.decode(torch.tensor(samples, dtype=torch.float))
        x_hat = x_hat.detach()
        x_hat = x_hat.reshape(-1, img_size[1], img_size[2], img_size[3])

    else:
        z = torch.randn(num_samples, model.latent_size)
        x_hat = model.decode(z)
        x_hat = x_hat.detach()
        x_hat = x_hat.reshape(-1, img_size[1], img_size[2], img_size[3])

    return x_hat



def generate_images_nosvgd(model, num_samples,img_size):
    if model.Fisher is True:
        z0 = torch.randn((num_samples, model.latent_size)).to('cuda:0')
        # z0 = 2*torch.rand(num_samples, model.latent_size) - 1
        # z0 = z0.to(device)
        z = model.flow.sample(noise= z0)
        
        # decode samples 
        x_hat = model.decode(z)
        x_hat = x_hat.cpu().detach()
        x_hat = x_hat.reshape(-1, img_size[1], img_size[2], img_size[3])
        
    else:
        z = torch.randn(num_samples, model.latent_size).to('cuda:0')
        x_hat = model.decode(z)
        x_hat = x_hat.detach()
        x_hat = x_hat.reshape(-1, img_size[1], img_size[2], img_size[3])
        
    return x_hat
