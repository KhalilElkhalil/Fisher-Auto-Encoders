import torch
import numpy as np
from scipy.stats import kde
from torchvision.utils import make_grid
from torch.autograd import grad
import numpy as np 
import numpy.linalg as la 
import matplotlib.pyplot as plt 
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import math
import copy 
import time 


# FID score 
from fid import fid_score
import fid.tools as tools
from fid.inception import InceptionV3

base_fid_statistics = None
inception_model = None

def initialize_fid(train_loader, sample_size=1000):
    global base_fid_statistics, inception_model
    if inception_model is None:
        inception_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]])
    inception_model = tools.cuda(inception_model)

    if base_fid_statistics is None:
        train_images = []
        for images, _ in train_loader:
            train_images += list(images.numpy())
            if len(train_images) > sample_size:
                train_images = train_images[:sample_size]
                break

        train_images = np.array(train_images)
        base_fid_statistics = fid_score.calculate_activation_statistics(
            train_images, inception_model, cuda=tools.is_cuda_available(),
            dims=2048)
        inception_model.cpu() 


def fid(generated_images, noise=None):
    score = fid_images(generated_images)
    return score


def fid_images(generated_images):
    global base_fid_statistics, inception_model
    inception_model = tools.cuda(inception_model)
    m1, s1 = fid_score.calculate_activation_statistics(
        generated_images.data.cpu().numpy(), inception_model, cuda=tools.is_cuda_available(),
        dims=2048)
    inception_model.cpu()
    m2, s2 = base_fid_statistics
    ret = fid_score.calculate_frechet_distance(m1, s1, m2, s2)
    return ret

