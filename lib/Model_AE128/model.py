import torch as t
import torch.nn as nn
import torch.nn.functional as F

from lib.model import *


# def upscale(in_channels, out_channels, kernel_size=3):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels * 4, kernel_size, padding=1, bias=False),
#         nn.BatchNorm2d(out_channels * 4),
#         nn.LeakyReLU(0.1),
#         nn.PixelShuffle(2))


# class Encoder128(BasicModule):
#     def __init__(self, init_dim=64, code_dim=1024, path=None):
#         assert path is not None
#         super(Encoder128, self).__init__(path)

#         self.encode_img = nn.Sequential(
#             nn.Conv2d(3, 32, 5, stride=1, padding=2),
#             self.conv(32, init_dim, batch_norm=False),
#             self.conv(init_dim, init_dim * 2),
#             self.conv(init_dim * 2, init_dim * 4),
#             self.conv(init_dim * 4, init_dim * 8),
#             self.conv(init_dim * 8, init_dim * 16),
#         )
#         # # use two linear layers to squeeze image
#         # it helps to reduce blur... Dont know why;;
#         self.linear1 = nn.Linear(init_dim*16 * 4*4, code_dim)
#         self.linear2 = nn.Linear(code_dim, 1024 * 4*4)

#         self.upscale = upscale(1024, 512)

#     def conv(self, in_channels, out_channels, kernel_size=5, batch_norm=True):
#         # if stride == 2:
#         #   padding = size - 1 - (size - k) // 2
#         modules = []
#         modules.append(
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=2, bias=False))
#         if batch_norm:
#             modules.append(nn.BatchNorm2d(out_channels))
#         modules.append(nn.LeakyReLU(0.1))
#         return nn.Sequential(*modules)

#     def forward(self, x): # (3, 128, 128)
#         b, c, w, h = x.shape
#         x = self.encode_img(x)
#         x = x.view(b, -1) # (1024*4*4)
#         x = self.linear1(x) # (cdim)
#         x = self.linear2(x) # (1024*4*4)
#         x = x.view(-1, 1024, 4, 4)
#         x = self.upscale(x) # (512, 8, 8)
#         return x


# class Decoder128(BasicModule):
#     def __init__(self, path=None):
#         assert path is not None
#         super(Decoder128, self).__init__(path)

#         self.upscale1 = upscale(512, 256)
#         self.upscale2 = upscale(256, 128)
#         self.upscale3 = upscale(128, 64)

#         self.decoded_img = nn.Sequential(
#             upscale(512, 256),
#             upscale(256, 128),
#             upscale(128, 64),
#             ResBlock(64),
#         )

#         self.out64 = nn.Sequential(
#             nn.Conv2d(64, 64, 3, 1, 1),
#             nn.LeakyReLU(0.1),
#             nn.Conv2d(64, 3, 5, 1, 2),
#             nn.Sigmoid()
#         )

#         self.out128 = nn.Sequential(
#             upscale(64, 32),
#             ResBlock(32),
#             ResBlock(32),
#             nn.Conv2d(32, 3, 5, 1, 2),
#             nn.Sigmoid()
#         )

#     def forward(self, x):    # (512, 8, 8)
#         x = self.decoded_img(x)
#         # out64 = self.out64(x)
#         out128 = self.out128(x)
#         return out128
