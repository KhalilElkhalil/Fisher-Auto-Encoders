import os
import torch
import PIL.Image as Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision import datasets
from torchvision.utils import save_image as save_img


def save_model(net, optim, epoch, path):
    state_dict = net.state_dict()
    torch.save({
        'epoch': epoch + 1,
        'state_dict': state_dict,
        'optimizer': optim.state_dict(),
        }, path)


def load_model(net, optim, path):
    print ("==> restoring checkpoint")
    ckpt = torch.load(path)
    epoch = ckpt['epoch']
    net.load_state_dict(ckpt['state_dict'])
    optim.load_state_dict(ckpt['optimizer'])
    print ("==> loaded checkpoint '{}' (epoch {})".format(path, epoch))
    return net, optim, epoch


def get_transform(args):
    if args.dataset == 'celeba':
        crop_size = 108
        re_size = 64
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size,
                offset_width:offset_width + crop_size]
        preprocess = transforms.Compose(
                [transforms.ToTensor(),
                    transforms.Lambda(crop),
                    transforms.ToPILImage(),
                    transforms.Scale(size=(re_size, re_size), interpolation=Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    return preprocess


def dataset_iterator(args):
    transform = get_transform(args)
    if args.dataset == 'mnist':
        train_gen, dev_gen, test_gen = mnist.load(args.batch_size, args.batch_size)
    elif args.dataset == 'cifar10':
        data_dir = '/data0/images/cifar-10-batches-py/'
        train_gen, dev_gen = cifar10.load(args.batch_size, data_dir)
        test_gen = None
    elif args.dataset == 'celeba':
        data_dir = '/data0/images/celeba'
        data = datasets.ImageFolder(data_dir, transform=transform)
        train_loader = torch.utils.data.DataLoader(data,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=4)
        test_loader = torch.utils.data.DataLoader(data,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=4)
        return train_loader, test_loader

    return (train_gen, dev_gen, test_gen)


def load_data(args):
    trainset = MNIST(root='./data/',
            train=True,
            transform=transforms.ToTensor(),
            download=True)

    testset = MNIST(root='./data/',
            train=False,
            transform=transforms.ToTensor(),
            download=True)

    train_loader = DataLoader(dataset=trainset,
            batch_size=args.batch_size,
            shuffle=True)

    test_loader = DataLoader(dataset=testset,
            batch_size=args.batch_size,
            shuffle=False)
    return (trainset, testset), (train_loader, test_loader)


def sample_noise(args):
    n = args.batch_size
    dist = 'normal'
    scale = 1.
    if dist == 'unifrom':
        noise = np.random.uniform(-1, 1, [n, args.dim]).astype(np.float32)
    elif dist in ('normal', 'sphere'):
        mean = np.zeros(args.dim)
        cov = np.identity(args.dim)
        noise = np.random.multivariate_normal(mean, cov, n).astype(np.float32)
        if dist == 'sphere':
            noise = noise/np.sqrt(np.sum(noise**2, axis=1))[:, np.newaxis]
    noise = torch.from_numpy(noise) * scale

    return noise


def pretrain_loss(encoded, noise):
    mean_g = torch.mean(noise, dim=0, keepdim=True)
    mean_e = torch.mean(encoded, dim=0, keepdim=True)
    mean_loss = F.mse_loss(mean_e, mean_g, size_average=True)

    cov_g = torch.matmul((noise-mean_g).transpose(0, 1), noise-mean_g)
    cov_g /= 1000 - 1
    cov_e = torch.matmul((encoded-mean_e).transpose(0, 1), encoded-mean_e)
    cov_e /= 1000 - 1
    cov_loss = F.mse_loss(cov_e, cov_g)
    return mean_loss + cov_loss


def ae_loss(args, real, sample):
    if args.loss == 'l2':
        loss = F.mse_loss(sample, real, size_average=False)
        loss = 0.2 * torch.sqrt(1e-08 + loss).mean()
    if args.loss == 'l2sq':
        loss = F.mse_loss(sample, real, size_average=True)
        loss = 0.05 * loss
    if args.loss == 'l1':
        abs = torch.abs(sample - real)
        loss = sqr.view(-1, abs.size(-1)).sum(0)
        loss = 0.02 * loss.mean()
    return loss


def gan_loss2(args, sample, encoded, netD):
    logits_e = netD(encoded)
    loss = args.l * (torch.log(logits_e)).mean()
    return loss


def gan_loss(args, sample, encoded, netD):
    loss = nn.BCEWithLogitsLoss()
    logits_e = netD(encoded)
    logits_g = netD(sample)

    loss_e = loss(logits_e, torch.zeros_like(logits_e))
    loss_g = loss(logits_g, torch.ones_like(logits_g))
    loss_e_trick = loss(logits_e, torch.ones_like(logits_e))
    loss_adversary = args.l * (loss_e + loss_g)
    # Non saturating loss trick
    loss_match = loss_e_trick
    return loss_match, (loss_adversary, logits_g, logits_e)


def update_lambda(args, loss_match, loss_rec):
    new_lambda = 0.5 * args.l + 0.5 * loss_rec / loss_match
    args.l = new_lambda
    return args


def save_image(netG, epoch, iter, orthogonal=True):
    imgs = []
    if orthogonal:
        fixed_noise = torch.ones(100, 8)
    else:
        fixed_noise = torch.randn(100, 8)
    noisev = Variable(fixed_noise).cuda()
    samples, samples_logits = netG(noisev.view(100, 8, 1, 1))
    samples = samples.view(100, 28, 28)
    for sample in samples:
        imgs.append(sample.view(1, 1, 28, 28))
    recons_image = torch.cat(imgs, dim=0)

    if not os.path.isdir('./data/reconst_images'):
        os.makedirs('data/reconst_images')
    save_img(recons_image.data,
        './data/reconst_images/wae_{}_{}.png'.format(epoch+1, iter), nrow=10)
