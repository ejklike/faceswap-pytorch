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

class FaceEncoder(BasicModule):
    def __init__(self, init_dim=32, path=None):
        assert path is not None
        super(FaceEncoder, self).__init__(path)

        self.init_dim = init_dim

        self.conv1 = self.conv(3, self.init_dim)        
        self.conv2 = self.conv(self.init_dim, self.init_dim * 2)
        self.conv3 = self.conv(self.init_dim * 2, self.init_dim * 4)
        self.conv4 = self.conv(self.init_dim * 4, self.init_dim * 8)

        self.code_dim = self.init_dim * 8
        # self.code_dim = self.init_dim * 4

        # use two linear layers to reduce #parameters
        self.linear1 = nn.Linear(self.code_dim*64*64, self.code_dim)
        self.linear2 = nn.Linear(self.code_dim, self.code_dim*4*4)

        self.upscale = upscale(self.code_dim, self.code_dim//2)

    def conv(self, in_channels, out_channels, kernel_size=5):
        # if stride == 2:
        #   padding = size - 1 - (size - k) // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=34),
            nn.LeakyReLU(0.1))

    def forward(self, x):
        b, c, w, h = x.shape
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(b, -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.view(-1, self.code_dim, 4, 4)
        x = self.upscale(x) # (128, 8, 8)
        return x


class BasicResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)

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



class FaceDecoder(BasicModule):
    def __init__(self, code_dim=256, path=None):
        assert path is not None
        super(FaceDecoder, self).__init__(path)

        self.code_dim = code_dim

        self.upscale1 = upscale(self.code_dim//2, self.code_dim//4)
        self.upscale2 = upscale(self.code_dim//4, self.code_dim//8)
        self.upscale3 = upscale(self.code_dim//8, self.code_dim//16)

        self.res_block1 = BasicResBlock(self.code_dim//16, self.code_dim//16, kernel_size=3)
        self.res_block2 = BasicResBlock(self.code_dim//16, self.code_dim//16, kernel_size=3)

        # if stride == 1:
        #   padding = (k - 1) // 2
        self.conv = nn.Conv2d(self.code_dim//16, 3, kernel_size=5, padding=2)

    def forward(self, x):    # (512, 8, 8)
        x = self.upscale1(x) # (256, 16, 16)
        x = self.upscale2(x) # (128, 32, 32)
        x = self.upscale3(x) # (64, 64, 64)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.conv(x)     # (3, 64, 64)
        x = F.sigmoid(x)
        return x