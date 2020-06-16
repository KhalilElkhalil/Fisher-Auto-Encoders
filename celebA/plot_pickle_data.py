import pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

#file_list = ['vae_figs_fid','fisher-exp_figs_fid','wae_figs_fid']
file_list = ['fisher-exp','vae','wae']
imgs_list = ['fisher-exp_noise_clean','vae_noise_clean','wae_noise_clean']
file_list = ['vae', 'fisher-exp','wae']
imgs_list = ['vae_noise_clean', 'fisher-exp_noise_clean','wae_noise_clean']
binary_file_list = [ 'fisher-exp_binary_dep', 'vae_binary_dep','wae_binary_dep']
binary_imgs_list = ['vae_binary_noise_clean_dep', 'fisher-exp_binary_noise_clean_dep','wae_binary_noise_clean_dep']
square_file_list = [ 'fisher-exp_square_dep', 'vae_square_dep','wae_square_dep']
square_imgs_list = ['vae_square_noise_clean_dep', 'fisher-exp_square_noise_clean_dep','wae_square_noise_clean_dep']
imgs_dictionaries = []

title_list= ['VAE', 'Fisher AE','WAE-GAN']
plot_title_list= [ 'Fisher AE','VAE','WAE-GAN']
def get_dicts(file_list):
    dictionaries = []
    for idx, fname in enumerate(file_list):
        print(fname)
        with open(fname + '.pkl', 'rb') as f:
            x = pickle.load(f)
            dictionaries.append(x)
        '''
        with open(imgs_list[idx] + '.pkl', 'rb') as f:
            x = pickle.load(f)
            imgs_dictionaries.append(x)
        '''
    return dictionaries
binary_dictionaries = get_dicts(binary_file_list)
binary_im_dictionaries = get_dicts(binary_imgs_list)

square_dictionaries = get_dicts(square_file_list)
square_im_dictionaries = get_dicts(square_imgs_list)

def plot_images():
    plt.figure(figsize=(15, 15), dpi=300)
    for i in range(len(file_list)):
        d = dictionaries[i] 
        im_d = imgs_dictionaries[i]
        try:
            fid_list = np.array(d['fisher_scores'])
            print(fid_list.mean())
            print(fid_list.std())
        except:
            pass
        plt.subplot(3,3,i+1)
        plt.title(title_list[i], fontsize=20)
        if i == 0:
            plt.xticks([])
            plt.yticks([])
            plt.ylabel('Noisy reconstruction', fontsize=20)
        else:
            plt.axis('off')
        show(im_d['corrupt'])
        plt.subplot(3,3,i+1 + 3)
        if i == 0:
            plt.xticks([])
            plt.yticks([])
            plt.ylabel('Test reconstruction', fontsize=20)
        else:
            plt.axis('off')
        show(im_d['clean'])
        plt.subplot(3,3,i+1 + 6)
        show(d['generated'])
        if i == 0:
            plt.xticks([])
            plt.yticks([])
            plt.ylabel('Random samples', fontsize=20)
        else:
            plt.axis('off')
    plt.tight_layout()
    plt.savefig('agg_celeba.png')
    plt.close('all')

def plot_images_single_row(img_dictionaries, title_list, fname):
    plt.figure(figsize=(15, 5), dpi=300)
    for i in range(len(title_list)):
        im_d = img_dictionaries[i]
        plt.subplot(1,3,i+1)
        plt.title(title_list[i], fontsize=20)
        if i == 0:
            plt.xticks([])
            plt.yticks([])
            plt.ylabel('Noisy reconstruction', fontsize=20)
        else:
            plt.axis('off')
        show(im_d['corrupt'])
    plt.tight_layout()
    plt.savefig(fname)
    plt.close('all')

def plot_lines(file_list, dictionaries, title_list, x, xlabel, fname):

    plt.figure(figsize=(8, 5))
    plt.style.use('seaborn-darkgrid')
    for i in range(len(file_list)):
        try:
            d = dictionaries[i]
            try:
                losses = d['mse_losses']
            except:
                try:
                    losses = d['mse']
                except:
                    pass

            plots = []
            errors = []
            for noise_idx in range(len(losses)-1):
                idx_loss = np.array(losses[noise_idx])
                plots.append(idx_loss.mean())
                errors.append(idx_loss.std())
            #plt.errorbar(x, plots, yerr=errors, label=file_list[i])
            errors = np.array(errors)
            plots = np.array(plots)
            plt.plot(x, plots,label=title_list[i])
            plt.fill_between(x, plots-errors, plots+errors, alpha=0.2)
        except FileNotFoundError:
            pass
    plt.legend()

    plt.xlabel(xlabel, fontsize=19)
    plt.ylabel('MSE', fontsize=19)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(fname)
    
plot_lines(square_file_list, square_dictionaries, plot_title_list, np.linspace(0.01,0.9,9), xlabel=r'$\nu$', fname='mse_square_plot_err.pdf')
plot_images_single_row(square_im_dictionaries, title_list, fname='agg_square_celeba.png')
#plot_lines(binary_file_list, binary_dictionaries, plot_title_list, np.linspace(0.01,0.9,9), xlabel=r'$\nu$', fname='mse_binary_plot_err.pdf')
#plot_images_single_row(binary_im_dictionaries, title_list)
