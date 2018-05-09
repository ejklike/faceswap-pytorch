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
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        if name is None:
            prefix = './model/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name


def upscale(in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels * 4, kernel_size, padding=1),
        nn.LeakyReLU(0.1),
        nn.PixelShuffle(2))


class FaceEncoder(nn.Module):
    def __init__(self, init_dim=64, model_dir='./model/'):
        super(FaceEncoder, self).__init__()
        self.encoder_pkl_name = os.path.join(model_dir, 'encoder.pkl')

        self.init_dim = init_dim

        self.conv1 = self.conv(3, self.init_dim)        
        self.conv2 = self.conv(self.init_dim, self.init_dim * 2)
        self.conv3 = self.conv(self.init_dim * 2, self.init_dim * 4)
        self.conv4 = self.conv(self.init_dim * 4, self.init_dim * 8)

        self.code_dim = self.init_dim * 8

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

    def save_checkpoint(self, epoch):
        encoder_state = {
            'state_dict': self.state_dict(),
        }
        save(encoder_state, self.encoder_pkl_name)
        print("=> saved checkpoint '{}'"
              .format(self.encoder_pkl_name))

    def load_checkpoint(self):
        if os.path.isfile(self.encoder_pkl_name):
            enc_checkpoint = load(self.encoder_pkl_name)

            self.load_state_dict(enc_checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(self.encoder_pkl_name))
        else:
            print("=> no checkpoint found at '{}'"
                  .format(self.encoder_pkl_name))



class FaceDecoder(nn.Module):
    def __init__(self, code_dim=128, model_dir='./model/'):
        super(FaceDecoder, self).__init__()
        self.decoder_pkl_name = os.path.join(model_dir, 'decoder.pkl')

        self.code_dim = code_dim

        self.upscale1 = upscale(self.code_dim//2, self.code_dim//4)
        self.upscale2 = upscale(self.code_dim//4, self.code_dim//8)
        self.upscale3 = upscale(self.code_dim//8, self.code_dim//16)
        # if stride == 1:
        #   padding = (k - 1) // 2
        self.conv = nn.Conv2d(self.code_dim//16, 3, kernel_size=5, padding=2)

    def res_block(self, in_channels, out_channels):
        pass

    def forward(self, x):    # (512, 8, 8)
        x = self.upscale1(x) # (256, 16, 16)
        x = self.upscale2(x) # (128, 32, 32)
        x = self.upscale3(x) # (64, 64, 64)
        x = self.conv(x)     # (3, 64, 64)
        x = F.sigmoid(x)
        return x

    def save_checkpoint(self, epoch):
        decoder_state = {
            'epoch': epoch,
            'state_dict': self.state_dict(),
        }
        save(decoder_state, self.decoder_pkl_name)
        print("=> saved checkpoint '{}' (epoch {})"
              .format(self.decoder_pkl_name, epoch))

    def load_checkpoint(self):
        if os.path.isfile(self.decoder_pkl_name):
            dec_checkpoint = load(self.decoder_pkl_name)

            start_epoch = dec_checkpoint['epoch']
            self.load_state_dict(dec_checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.decoder_pkl_name, start_epoch))
        else:
            print("=> no checkpoint found at '{}'"
                  .format(self.decoder_pkl_name))


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))

    # def save_checkpoint(self, epoch):
    #     encoder_state = {
    #         'state_dict': self.encoder.state_dict(),
    #     }
    #     decoder_state = {
    #         'epoch': epoch,
    #         'state_dict': self.decoder.state_dict(),
    #     }
    #     save(encoder_state, self.encoder_pkl_name)
    #     save(decoder_state, self.decoder_pkl_name)
    #     print("=> saved checkpoint '{}, {}' (epoch {})"
    #           .format(self.encoder_pkl_name, self.decoder_pkl_name, epoch))

    # def load_checkpoint(self):
    #     if os.path.isfile(self.encoder_pkl_name) and os.path.isfile(self.decoder_pkl_name):
    #         enc_checkpoint = load(self.encoder_pkl_name)
    #         dec_checkpoint = load(self.decoder_pkl_name)

    #         start_epoch = dec_checkpoint['epoch']
    #         self.encoder.load_state_dict(enc_checkpoint['state_dict'])
    #         self.decoder.load_state_dict(dec_checkpoint['state_dict'])
    #         print("=> loaded checkpoint '{}, {}' (epoch {})"
    #               .format(self.encoder_pkl_name, self.decoder_pkl_name, start_epoch))
    #     else:
    #         print("=> no checkpoint found at '{}, {}'"
    #               .format(self.encoder_pkl_name, self.decoder_pkl_name))


    # def save(self):
    #     save(self.encoder, self.encoder_pkl_name)
    #     save(self.decoder, self.decoder_pkl_name)
    #     print("--------model saved--------\n")

    # def load(self):
    #     try:
    #         self.encoder = load(self.encoder_pkl_name)
    #         self.decoder = load(self.decoder_pkl_name)
    #         print("\n--------model restored--------\n")
    #     except:
    #         print("\n--------model not restored--------\n")
    #         pass