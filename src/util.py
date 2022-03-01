import numpy as np
import torch
from torch import autograd
import wandb
from dotenv import load_dotenv
import os
import subprocess
import shutil
import matplotlib.pyplot as plt
from torch import nn
import tifffile

# check for existing models and folders
def check_existence(tag, overwrite):
    """Checks if model exists, then asks for user input. Returns True for overwrite, False for load.

    :param tag: [description]
    :type tag: [type]
    :raises SystemExit: [description]
    :raises AssertionError: [description]
    :return: True for overwrite, False for load
    :rtype: [type]
    """
    if overwrite:
        return True
    root = f'runs/{tag}'
    check_D = os.path.exists(f'{root}/Disc.pt')
    check_G = os.path.exists(f'{root}/Gen.pt')
    if check_G or check_D:
        print(f'Models already exist for tag {tag}.')
        x = input("To overwrite existing model enter 'o', to load existing model enter 'l' or to cancel enter 'c'.\n")
        if x=='o':
            print("Overwriting")
            return True
        if x=='l':
            print("Loading previous model")
            return False
        elif x=='c':
            raise SystemExit
        else:
            raise AssertionError("Incorrect argument entered.")
    return True

# set-up util
def initialise_folders(tag, overwrite):
    """[summary]

    :param tag: [description]
    :type tag: [type]
    """
    if overwrite:
        try:
            os.mkdir(f'runs')
        except:
            pass
        try:
            os.mkdir(f'runs/{tag}')
        except:
            pass

def wandb_init(name, offline):
    """[summary]

    :param name: [description]
    :type name: [type]
    :param offline: [description]
    :type offline: [type]
    """
    if offline:
        mode = 'disabled'
    else:
        mode = None
    load_dotenv(os.path.join(os.getcwd(), '.env'))
    API_KEY = os.getenv('WANDB_API_KEY')
    ENTITY = os.getenv('WANDB_ENTITY')
    PROJECT = os.getenv('WANDB_PROJECT')
    if API_KEY is None or ENTITY is None or PROJECT is None:
        raise AssertionError('.env file arguments missing. Make sure WANDB_API_KEY, WANDB_ENTITY and WANDB_PROJECT are present.')
    print("Logging into W and B using API key {}".format(API_KEY))
    process = subprocess.run(["wandb", "login", API_KEY], capture_output=True)
    print("stderr:", process.stderr)

    
    print('initing')
    wandb.init(entity=ENTITY, name=name, project=PROJECT, mode=mode)

    wandb_config = {
        'active': True,
        'api_key': API_KEY,
        'entity': ENTITY,
        'project': PROJECT,
        # 'watch_called': False,
        'no_cuda': False,
        # 'seed': 42,
        'log_interval': 1000,

    }
    # wandb.watch_called = wandb_config['watch_called']
    wandb.config.no_cuda = wandb_config['no_cuda']
    # wandb.config.seed = wandb_config['seed']
    wandb.config.log_interval = wandb_config['log_interval']

def wandb_save_models(fn):
    """[summary]

    :param pth: [description]
    :type pth: [type]
    :param fn: [description]
    :type fn: filename
    """
    shutil.copy(fn, os.path.join(wandb.run.dir, fn))
    wandb.save(fn)

# training util
def preprocess(data_path, oh=True):
    """[summary]

    :param imgs: [description]
    :type imgs: [type]
    :return: [description]
    :rtype: [type]
    """
    
    if oh:
        img_ohs = []
        for p in ['top', 'bot']:
            img = plt.imread(f'{data_path}{p}.png')[:, :, 0]
            phases = np.unique(img)
            if len(phases) > 10:
                raise AssertionError('Image not one hot encoded.')
            x, y = img.shape
            img_oh = torch.zeros(len(phases), x, y)
            for i, ph in enumerate(phases):
                img_oh[i][img == ph] = 1
            img_ohs.append(img_oh)
        return img_ohs, len(phases)
    else:
        img = torch.tensor(plt.imread(data_path)[:, :, 0])
        nphases=2
        x, y = img.shape
        img_oh = torch.zeros(nphases, x, y)
        img_oh[0] = img
        img_oh[1] = 1 - img
        return img_oh, nphases

def calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device, gp_lambda, nc):
    """[summary]

    :param netD: [description]
    :type netD: [type]
    :param real_data: [description]
    :type real_data: [type]
    :param fake_data: [description]
    :type fake_data: [type]
    :param batch_size: [description]
    :type batch_size: [type]
    :param l: [description]
    :type l: [type]
    :param device: [description]
    :type device: [type]
    :param gp_lambda: [description]
    :type gp_lambda: [type]
    :param nc: [description]
    :type nc: [type]
    :return: [description]
    :rtype: [type]
    """
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(
        real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, nc, l, l)
    alpha = alpha.to(device)

    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(
                                  disc_interpolates.size()).to(device),
                              create_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty

def batch_real(img, l, bs):
    """[summary]
    :param training_imgs: [description]
    :type training_imgs: [type]
    :return: [description]
    :rtype: [type]
    """
    flip, idx = torch.randint(2, (2,))
    img = img[idx]
    if flip:
        img = torch.flip(img, (-1,))
    n_ph, x_max, y_max = img.shape
    data = torch.zeros((bs, n_ph, l, l))
    for i in range(bs):
        y = torch.randint(y_max - l, (1,))
        data[i] = img[:, :, y:y+l]
    # data[data==0] += torch.rand_like(data[data==0])*0.1
    # data[data==1] -= torch.rand_like(data[data==1])*0.1
    
    return data

# Evaluation util
def post_process(img):
    """Turns a n phase image (bs, n, imsize, imsize) into a plottable euler image (bs, 3, imsize, imsize, imsize)

    :param img: a tensor of the n phase img
    :type img: torch.Tensor
    :return:
    :rtype:
    """
    img = img.detach().cpu()
    img = torch.argmax(img, dim=1).unsqueeze(-1).numpy()

    return img * 255

def generate(c, netG, lz):
    """Generate an instance from generator, save to .tif

    :param c: Config object class
    :type c: Config
    :param netG: Generator instance
    :type netG: Generator
    :return: Post-processed generated instance
    :rtype: torch.Tensor
    """
    tag, ngpu, nz, pth = c.tag, c.ngpu, c.nz, c.path
    real = preprocess(c.data_path)[0]
    real = [post_process(img.unsqueeze(0)) for img in real]
    out_pth = f"runs/{tag}/out.tif"
    if torch.cuda.device_count() > 1 and c.ngpu > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
    device = torch.device("cuda:0" if(
            torch.cuda.is_available() and ngpu > 0) else "cpu")
    if (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu))).to(device)
    netG.load_state_dict(torch.load(f"{pth}/Gen.pt"))
    netG.eval()
    # netG.eval()
    imgs = []
    noise = torch.randn(10, nz, 1, lz)

    raw = netG(noise)
    img = post_process(raw)
    img = np.array(img, dtype=np.uint8)
    a, b, c, d = img.shape
    img = img.reshape(-1, c)[:,:3093]
    print(img.shape, real[0].shape)
    for r in real:
        r = r[0,:,:,0]
        idx = np.random.randint(0,9)*128
        img[idx:idx+128] = r
        print(idx/128)
    img = np.stack([img for i in range(3)], -1)
    print(img.shape)
    print(pth)
    plt.imsave(f'{pth}/output.png', img)
    
    return img

def opt_generate(c, netG, netD, lz):
    tag, ngpu, nz, pth = c.tag, c.ngpu, c.nz, c.path

    out_pth = f"runs/{tag}/"
    if torch.cuda.device_count() > 1 and c.ngpu > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
    device = torch.device("cuda:0" if(
            torch.cuda.is_available() and ngpu > 0) else "cpu")
    if (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu))).to(device)
    netG.load_state_dict(torch.load(f"{pth}/Gen.pt"))
    netG.cuda()
    netG.eval()
    netD.load_state_dict(torch.load(f"{pth}/Disc.pt"))
    netD.cuda()
    netD.eval()
    noise = [torch.nn.Parameter(torch.randn(1, nz, 1, lz, requires_grad = True, device=device))]
    opt = torch.optim.SGD(params=noise, lr=0.01)
    imgs = []
    iters=50
    store = iters//10
    for i in range(iters):
        raw = netG(noise[0])
        loss = -netD(raw).mean()
        loss.backward()
        opt.step()
        print(i, loss)
        if i%store==0:
            imgs.append(post_process(raw)[0])
    gb = np.concatenate(imgs, axis=0)
    tif = np.array(gb, dtype=np.uint8)
    tifffile.imwrite(f'{out_pth}out.tif', tif, imagej=True)
    print(out_pth)
    return tif
def progress(i, iters, n, num_epochs, timed):
    """[summary]

    :param i: [description]
    :type i: [type]
    :param iters: [description]
    :type iters: [type]
    :param n: [description]
    :type n: [type]
    :param num_epochs: [description]
    :type num_epochs: [type]
    :param timed: [description]
    :type timed: [type]
    """
    progress = 'iteration {} of {}, epoch {} of {}'.format(
        i, iters, n, num_epochs)
    print(f"Progress: {progress}, Time per iter: {timed}")

def plot_img(img, iter, epoch, path, offline=True):
    """[summary]

    :param img: [description]
    :type img: [type]
    :param slcs: [description], defaults to 4
    :type slcs: int, optional
    """
    if not offline:
        wandb.log({"raw slices": [wandb.Image(i[0]) for i in img]})
    img = post_process(img)
    if not offline:
        wandb.log({"slices": [wandb.Image(i) for i in img]})
    else:
        fig, axs = plt.subplots(1, img.shape[0])
        for ax, im in zip(axs, img):
            ax.imshow(im)
        plt.savefig(f'{path}/{epoch}_{iter}_slices.png')