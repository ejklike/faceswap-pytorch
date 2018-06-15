import torch as t
import torch.nn as nn
import torch.nn.functional as F

from lib.model import *


def upscale(in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels * 4, kernel_size, padding=1),
        nn.LeakyReLU(0.1),
        nn.PixelShuffle(2))


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