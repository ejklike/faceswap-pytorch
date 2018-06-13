import os
import shutil

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import load, save

# https://github.com/pytorch/examples

# https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
# https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/23

# https://github.com/GunhoChoi/Kind-PyTorch-Tutorial/blob/master/10_InfoGAN_Least_Squares_Loss/InfoGAN_Least_Squares_Loss.ipynb
# https://github.com/GunhoChoi/Kind-PyTorch-Tutorial/tree/master/07_Denoising_Autoencoder
# https://github.com/ShayanPersonal/stacked-autoencoder-pytorch/blob/master/model.py
# https://pytorch.org/docs/master/nn.html



def get_model(model_name, model_class, **kwargs):
    print('build {}...'.format(model_name))
    model = model_class(**kwargs)
    model.load()
    return model


def get_optimizer(lr, optimizer_path, parameters):
    optimizer = t.optim.Adam(
        parameters, lr=lr,  betas=(0.5, 0.999))
    if os.path.isfile(optimizer_path):
        optimizer.load_state_dict(t.load(optimizer_path))
    return optimizer


def save_optimizer(optimizer_path, optimizer):
    t.save(optimizer.state_dict(), optimizer_path)


def _first_order(x, axis=1):
    _, _, w, h = x.shape
    if axis == 2:
        left = x[:, :, 0:w-1, :]
        right = x[:, :, 1:w, :]
        return t.abs(left - right)
    elif axis == 3:
        upper = x[:, :, :, 0:h-1]
        lower = x[:, :, :, 1:h]
        return t.abs(upper - lower)
    else:
        return None


def define_lossG(fake_img, real_img, output_neg, device=None):
        # MAE loss
        l1_loss = nn.L1Loss()(fake_img, real_img)

        # Edge loss (similar with total variation loss)
        edge_loss_w = t.mean(t.abs(
            _first_order(fake_img, axis=2) - _first_order(real_img, axis=2)))
        edge_loss_h = t.mean(t.abs(
            _first_order(fake_img, axis=3) - _first_order(real_img, axis=3)))

        # LS-GAN loss
        loss_G_fake = nn.MSELoss()(output_neg, t.ones(output_neg.shape).to(device))

        lossG = l1_loss + 0.5 * loss_G_fake + 0.01 * (edge_loss_w + edge_loss_h)
        return lossG


def define_lossD(output_pos, output_neg, device=None):
        # LS-GAN loss
        # MSE for real
        loss_D_real = nn.MSELoss()(output_pos, t.ones(output_pos.shape).to(device))
        # MSE for fake
        loss_D_fake = nn.MSELoss()(output_neg, t.zeros(output_neg.shape).to(device))

        return loss_D_real + loss_D_fake


class BasicModule(nn.Module):
    def __init__(self, path):
        super(BasicModule, self).__init__()
        self.path = path

    def load(self):
        if os.path.isfile(self.path):
            ckpt = t.load(self.path)
            self.load_state_dict(ckpt['state_dict'])
            print("=> loaded checkpoint '{}'".format(self.path))
            if ckpt['epoch'] is not None:
                print('   (prev_epoch: {})'.format(ckpt['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(self.path))

    def save(self, epoch=None):
        ckpt = {
            'state_dict': self.state_dict(),
            'epoch': epoch
        }
        t.save(ckpt, self.path)
        print("=> saved checkpoint '{}'".format(self.path))


def upscale(in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels * 4, kernel_size, padding=1),
        nn.LeakyReLU(0.1),
        nn.PixelShuffle(2))


class BasicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class S1ENC(BasicModule):
    def __init__(self, init_dim=64, code_dim=1024, path=None):
        assert path is not None
        super(S1ENC, self).__init__(path)

        self.conv1 = self.conv(3, init_dim)
        self.conv2 = self.conv(init_dim, init_dim * 2)
        self.conv3 = self.conv(init_dim * 2, init_dim * 4)
        self.conv4 = self.conv(init_dim * 4, init_dim * 8)

        # # use two linear layers to squeeze image
        # it helps to reduce blur... Dont know why;;
        self.linear1 = nn.Linear(init_dim*8 * 4*4, code_dim)
        self.linear2 = nn.Linear(code_dim, 1024 * 4*4)

        self.upscale = upscale(1024, 512)

    def conv(self, in_channels, out_channels, kernel_size=5):
        # if stride == 2:
        #   padding = size - 1 - (size - k) // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=2),
            nn.LeakyReLU(0.1))

    def forward(self, x): # (3, 64, 64)
        b, c, w, h = x.shape
        x = self.conv1(x) # (128, 32, 32)
        x = self.conv2(x) # (256, 16, 16)
        x = self.conv3(x) # (512, 8, 8)
        x = self.conv4(x) # (1024, 4, 4)
        x = x.view(b, -1) # (1024*4*4)
        x = self.linear1(x) # (cdim)
        x = self.linear2(x) # (1024*4*4)
        x = x.view(-1, 1024, 4, 4)
        x = self.upscale(x) # (512, 8, 8)
        return x


class S1DEC(BasicModule):
    def __init__(self, path=None):
        assert path is not None
        super(S1DEC, self).__init__(path)

        self.upscale1 = upscale(512, 256)
        self.upscale2 = upscale(256, 128)
        self.upscale3 = upscale(128, 64)

        # self.res_block1 = BasicResBlock(64, 64, kernel_size=3)
        # self.res_block2 = BasicResBlock(64, 64, kernel_size=3)

        # if stride == 1:
        #   padding = (k - 1) // 2
        self.conv = nn.Conv2d(64, 3, kernel_size=5, padding=2)

    def forward(self, x):    # (512, 8, 8)
        x = self.upscale1(x) # (256, 16, 16)
        x = self.upscale2(x) # (128, 32, 32)
        x = self.upscale3(x) # (64, 64, 64)
        # x = self.res_block1(x)
        # x = self.res_block2(x)
        x = self.conv(x)     # (3, 64, 64)
        x = F.sigmoid(x)
        return x

class S1DISC(BasicModule):
    def __init__(self, path=None):
        assert path is not None
        super(S1DISC, self).__init__(path)
        init_dim = 64
        self.encode_img = nn.Sequential(
            nn.Conv2d(6, init_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (init_dim) x 32 x 32
            nn.Conv2d(init_dim, init_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (init_dim*2) x 16 x 16
            nn.Conv2d(init_dim*2, init_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (init_dim*4) x 8 x 8
            nn.Conv2d(init_dim*4, init_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(init_dim * 8),
            # state size (init_dim * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, image):
        img_embedding = self.encode_img(image)

        return img_embedding