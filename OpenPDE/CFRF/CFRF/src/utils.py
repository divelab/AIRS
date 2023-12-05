import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# save image, and save animation

def save_img(eval_data, args, save_animation=False, test=False):

    if test:
        save_path = os.path.join(args.working_dir, "test_eval")
    else:
        save_path = os.path.join(args.working_dir, "valid_eval")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pred = eval_data['pred']
    label = eval_data['label']
    if eval_data.get('HR_residual') is not None:
        HR_residual = eval_data['HR_residual']

    torch.save(eval_data, os.path.join(save_path, f'eval.pt'))

    fig, (ax1, ax2) = plt.subplots(1,2)
    if pred.shape[1] < 99:
        return
    for i in range(50,100,10):

        ax1.set_title("HR reconstructed")
        ax2.set_title("HR ground truth")

        if pred.dim() == 5:
            ax1.imshow(pred[0, i, :, :, 0], cmap='twilight')
            ax2.imshow(label[0, i, :, :, 0], cmap='twilight')
        else:
            ax1.imshow(pred[0][i], cmap='twilight')
            ax2.imshow(label[0][i], cmap='twilight')

        plt.savefig(os.path.join(save_path, f'pred_{i}.png'))


def plot_animation_two(LRdata=None, HRdata=None, frames=200, save_name=''):
    fig, (ax1, ax2) = plt.subplots(1,2)
    ims = []
    for i in range(frames):
        ax1.set_title("LR")
        ax2.set_title("HR")

        im1 = ax1.imshow(LRdata[0,i], animated=True, cmap='twilight')
        im2 = ax2.imshow(HRdata[0,i], animated=True, cmap='twilight')
        if i == 0:
            ax1.imshow(LRdata[0,i], cmap='twilight')
            ax2.imshow(HRdata[0,i], cmap='twilight')
        ims.append([im1, im2])
    
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1500)
    ani.save(f'./{save_name}_animation.gif', writer='imagemagick', fps=60)
    plt.show()


def pdeLossSWE(data=None, dx=None, dy=None, dt=None, g=1, L=5):
    # eq1: h_t + (hu)_x + (hv)_y = 0
    # eq2: (hu)_t + (hu^2 + \frac{1}{2}gh^2)_x + (huv)_y = 0
    # eq3: (hv)_t + (huv)_x + (hv^2 + \frac{1}{2}gh^2)_y = 0

    data = data.clone()
    data.requires_grad_(True)
    nx = data.size(-3)
    ny = data.size(-2)
    
    H = data[..., 0]
    U = data[..., 1]
    V = data[..., 2]

    momentumx = H * U
    momentumy = H * V
    
    # Wavenumbers
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=data.device),
                     torch.arange(start=-k_max, end=0, step=1, device=data.device)), 0).\
        reshape(N, 1).repeat(1, N).reshape(1,1,N,N)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=data.device),
                     torch.arange(start=-k_max, end=0, step=1, device=data.device)), 0).\
        reshape(1, N).repeat(N, 1).reshape(1,1,N,N)
    

    # for eq1
    h_t = (H[:, 2:, :, :] - H[:, :-2, :, :]) / (2 * dt)
    
    momentumx_h = torch.fft.fft2(momentumx[:, 1:-1], dim=[2, 3])
    momentumy_h = torch.fft.fft2(momentumy[:, 1:-1], dim=[2, 3])
    
    momentumx_x_h = 2. * math.pi / L * 1j * k_x * momentumx_h
    momentumy_y_h = 2. * math.pi / L * 1j * k_y * momentumy_h
    
    momentumx_x = torch.fft.irfft2(momentumx_x_h[..., :, :k_max + 1], dim=[2, 3])
    momentumy_y = torch.fft.irfft2(momentumy_y_h[..., :, :k_max + 1], dim=[2, 3])
    
    # for eq2
    hu_t = (momentumx[:, 2:, :, :] - momentumx[:, :-2, :, :]) / (2 * dt)

    ## (hu^2 + \frac{1}{2}gh^2)
    term1 = H*U*U + 0.5*g*H*H

    huv = momentumx * V
    
    term1_h = torch.fft.fft2(term1[:, 1:-1], dim=[2, 3])
    huv_h = torch.fft.fft2(huv[:, 1:-1], dim=[2, 3])
    
    term1_x_h = 2. * math.pi / L * 1j * k_x * term1_h
    huv_y_h = 2. * math.pi / L * 1j * k_y * huv_h
    
    term1_x = torch.fft.irfft2(term1_x_h[..., :, :k_max + 1], dim=[2, 3])
    huv_y = torch.fft.irfft2(huv_y_h[..., :, :k_max + 1], dim=[2, 3])
    
    # for eq3
    hv_t = (momentumy[:, 2:, :, :] - momentumy[:, :-2, :, :]) / (2 * dt)

    ## (hv^2 + \frac{1}{2}gh^2)
    term2 = H*V*V + 0.5*g*H*H
    
    term2_h = torch.fft.fft2(term2[:, 1:-1], dim=[2, 3])
    
    term2_y_h = 2. * math.pi / L * 1j * k_y * term2_h
    huv_x_h = 2. * math.pi / L * 1j * k_x * huv_h
    
    term2_y = torch.fft.irfft2(term2_y_h[..., :, :k_max + 1], dim=[2, 3])
    huv_x = torch.fft.irfft2(huv_x_h[..., :, :k_max + 1], dim=[2, 3])   
    
    
    # sum
    eq1 = h_t + momentumx_x + momentumy_y
    eq2 = hu_t + term1_x + huv_y
    eq3 = hv_t + term2_y + huv_x

    # residual = eq1 + eq2 + eq3
    # residual_loss = (residual**2).mean()
    residual_loss = (eq1**2).mean() + (eq2**2).mean() + (eq3**2).mean()
    
    return residual_loss

def pdeLossDiffReact(data=None, dx=None, dy=None, dt=None, L=2):


    data = data.clone()
    data.requires_grad_(True)
    nx = data.size(-3)
    ny = data.size(-2)
    
    Du = 1e-3
    Dv = 5e-3
    k = 5e-3
    
    
    U = data[..., 0]
    V = data[..., 1]
    
    Ru = U[:, 1:-1] - U[:, 1:-1]**3 - k - V[:, 1:-1]
    Rv = U[:, 1:-1] - V[:, 1:-1]
    
    # Wavenumbers
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=data.device),
                     torch.arange(start=-k_max, end=0, step=1, device=data.device)), 0).\
        reshape(N, 1).repeat(1, N).reshape(1,1,N,N)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=data.device),
                     torch.arange(start=-k_max, end=0, step=1, device=data.device)), 0).\
        reshape(1, N).repeat(N, 1).reshape(1,1,N,N)
    

    u_t = (U[:, 2:, :, :] - U[:, :-2, :, :]) / (2 * dt)
    v_t = (V[:, 2:, :, :] - V[:, :-2, :, :]) / (2 * dt)
    
    u_h = torch.fft.fft2(U[:, 1:-1], dim=[2, 3])
    v_h = torch.fft.fft2(V[:, 1:-1], dim=[2, 3])
    
    u_xx_h = -4*(math.pi**2)/(L**2)*(k_x ** 2) * u_h
    u_yy_h = -4*(math.pi**2)/(L**2)*(k_y ** 2) * u_h
    v_xx_h = -4*(math.pi**2)/(L**2)*(k_x ** 2) * v_h
    v_yy_h = -4*(math.pi**2)/(L**2)*(k_y ** 2) * v_h
    
    u_xx = torch.fft.irfft2(u_xx_h[..., :, :k_max + 1], dim=[2, 3])
    u_yy = torch.fft.irfft2(u_yy_h[..., :, :k_max + 1], dim=[2, 3])
    v_xx = torch.fft.irfft2(v_xx_h[..., :, :k_max + 1], dim=[2, 3])
    v_yy = torch.fft.irfft2(v_yy_h[..., :, :k_max + 1], dim=[2, 3])

    
    # sum
    eq1 = u_t - Du*u_xx - Du*u_yy - Ru
    eq2 = v_t - Dv*v_xx - Dv*v_yy - Rv

    residual_loss = (eq1**2).mean() + (eq2**2).mean()
#     residual_loss = ((eq1 + eq2)**2).mean()
    
    return residual_loss


# PDE loss below for iCFD is adapted from https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution/blob/main_v1/train_ddpm/functions/losses.py

def pdeLoss(w, re=1000.0, dt=1/32, s=64, L=1):
    # w [b t h w]
    batchsize = w.size(0)
    w = w.clone()
    w.requires_grad_(True)
    nx = w.size(2)
    ny = w.size(3)
    device = w.device

    w_h = torch.fft.fft2(w[:, 1:-1], dim=[2, 3])
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(N, 1).repeat(1, N).reshape(1,1,N,N)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(1, N).repeat(N, 1).reshape(1,1,N,N)
    # Negative Laplacian in Fourier space
    lap = 4*(math.pi**2)/(L**2)*(k_x ** 2 + k_y ** 2)
    lap[..., 0, 0] = 1.0
    psi_h = w_h / lap

    u_h = 2. * math.pi / L * 1j * k_y * psi_h
    v_h = 2. * math.pi / L * -1j * k_x * psi_h
    wx_h = 2. * math.pi / L * 1j * k_x * w_h
    wy_h = 2. * math.pi / L * 1j * k_y * w_h
    wlap_h = -lap * w_h

    u = torch.fft.irfft2(u_h[..., :, :k_max + 1], dim=[2, 3])
    v = torch.fft.irfft2(v_h[..., :, :k_max + 1], dim=[2, 3])
    wx = torch.fft.irfft2(wx_h[..., :, :k_max + 1], dim=[2, 3])
    wy = torch.fft.irfft2(wy_h[..., :, :k_max + 1], dim=[2, 3])
    wlap = torch.fft.irfft2(wlap_h[..., :, :k_max + 1], dim=[2, 3])
    advection = u*wx + v*wy

    wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)
    
    #Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
    t = torch.linspace(0, L, nx + 1, device=device)
    t = t[0:-1]

    X,Y = torch.meshgrid(t, t, indexing='ij')
    f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))

    residual = wt + (advection - (1.0 / re) * wlap) - f
    residual_loss = (residual**2).mean()

    return residual_loss


def divergenceLoss(w, re=1000.0, dt=1/32, s=64, L=1):
    # w [b t h w]
    batchsize = w.size(0)
    w = w.clone()
    w.requires_grad_(True)
    nx = w.size(2)
    ny = w.size(3)
    device = w.device

    w_h = torch.fft.fft2(w[:, 1:-1], dim=[2, 3])
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(N, 1).repeat(1, N).reshape(1,1,N,N)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(1, N).repeat(N, 1).reshape(1,1,N,N)
    # Negative Laplacian in Fourier space
    lap = 4*(math.pi**2)/(L**2)*(k_x ** 2 + k_y ** 2)
    lap[..., 0, 0] = 1.0
    psi_h = w_h / lap

    u_h = 2. * math.pi / L * 1j * k_y * psi_h
    v_h = 2. * math.pi / L * -1j * k_x * psi_h

    ux_h = 2. * math.pi / L * k_x * u_h
    vy_h = 2. * math.pi / L * k_y * v_h

    ux = torch.fft.irfft2(ux_h[..., :, :k_max + 1], dim=[2, 3])
    vy = torch.fft.irfft2(vy_h[..., :, :k_max + 1], dim=[2, 3])

    divergence = ux + vy

    return (divergence ** 2).mean()

def tv_loss(img, tv_weight):

    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2)).sqrt()
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2)).sqrt()
    loss = tv_weight * (h_variance + w_variance)
    return loss

def downsample(HR, ratio=None):
    
    LR = HR[..., ::ratio, ::ratio]
    
    return LR 

def create_mask(inputs, offset=0):
    mask = torch.zeros_like(inputs)
    nx = inputs.shape[-2]
    ny = inputs.shape[-1]
    mask[..., offset:(nx-offset), offset:(ny-offset)] = 1

    return mask

# below code are adapted from https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class ContentLoss(nn.Module):

    def __init__(self):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        # self.target = target.detach()

    def forward(self, input):
        # self.loss = F.mse_loss(input, target)
        return input


# cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
# cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']

def get_model_and_losses(cnn, normalization_mean, normalization_std,
                        content_layers=content_layers_default):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    # content_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    # model = nn.Sequential(normalization)
    model = nn.Sequential()

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            content_loss = ContentLoss()
            model.add_module("content_loss_{}".format(i), content_loss)
            # content_losses.append(content_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss):
            break

    model = model[:(i + 1)]

    return model