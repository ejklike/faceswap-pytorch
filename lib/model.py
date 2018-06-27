import os

import torch as t
import torch.nn as nn
# import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, nin, nout, kernel_size=4, stride=2, padding=1, bias=True):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=nin, bias=False)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

def conv3x3(in_planes, out_planes, bias=True, sep=False):
    "3x3 convolution with SAME padding, leakyReLU"
    # if stride == 1:
    #   padding = (k - 1) // 2
    conv = SeparableConv2d if sep else nn.Conv2d
    return nn.Conv2d(in_planes, out_planes, bias=bias,
                     kernel_size=3, stride=1, padding=1)

def strided_conv3x3(in_planes, out_planes, bias=True, sep=False):
    "3x3 convolution with SAME padding, leakyReLU"
    conv = SeparableConv2d if sep else nn.Conv2d
    return nn.Conv2d(in_planes, out_planes, bias=bias,
                     kernel_size=3, stride=2, padding=1)

def strided_conv4x4(in_planes, out_planes, bias=True, sep=False):
    "4x4 strided convolution with SAME padding, leakyReLU"
    # e.g., input (64x64) ==> output (32x32) img (stride=2) 
    conv = SeparableConv2d if sep else nn.Conv2d
    return conv(in_planes, out_planes, bias=bias,
                kernel_size=4, stride=2, padding=1)

def strided_conv5x5(in_planes, out_planes, bias=True, sep=False):
    "5x5 convolution with SAME padding, LeakyReLU"
    # e.g., input (64x64) ==> output (32x32) img (stride=2) 
    conv = SeparableConv2d if sep else nn.Conv2d
    return conv(in_planes, out_planes, bias=bias,
                kernel_size=5, stride=2, padding=2)

def up_block3x3(in_channels, out_channels, sep=False):
    # if stride == 1:
    #   padding = (k - 1) // 2
    return nn.Sequential(
        conv3x3(in_channels, out_channels * 4, sep=sep),
        nn.LeakyReLU(0.1),
        nn.PixelShuffle(2))

# ---------------------------------------------------------

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

# ---------------------------------------------------------


class Encoder64(BasicModule):
    def __init__(self, init_dim=64, code_dim=1024, path=None):
        assert path is not None
        super(Encoder64, self).__init__(path)

        self.encode_img = nn.Sequential(
            strided_conv5x5(3, init_dim),
            nn.LeakyReLU(0.1),
            strided_conv5x5(init_dim, init_dim * 2),
            nn.LeakyReLU(0.1),
            strided_conv5x5(init_dim * 2, init_dim * 4),
            nn.LeakyReLU(0.1),
            strided_conv5x5(init_dim * 4, init_dim * 8),
            nn.LeakyReLU(0.1)
        )

        # # use two linear layers to squeeze image
        # it helps to reduce blur... Dont know why;;
        self.linear = nn.Sequential(
            nn.Linear(init_dim*8 * 4*4, code_dim),
            nn.Linear(code_dim, 1024*4*4)
        )

        self.upscale = up_block3x3(1024, 512)

    def forward(self, x): # (3, 64, 64)
        b, c, w, h = x.shape
        x = self.encode_img(x)
        x = x.view(b, -1) # (1024*4*4)
        x = self.linear(x) # (1024*4*4)
        x = x.view(-1, 1024, 4, 4)
        x = self.upscale(x) # (512, 8, 8)
        return x


class Decoder64(BasicModule):
    def __init__(self, path=None):
        assert path is not None
        super(Decoder64, self).__init__(path)

        self.decode_img = nn.Sequential(
            up_block3x3(512, 256),
            up_block3x3(256, 128),
            up_block3x3(128, 64),
            nn.Conv2d(64, 3, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):    # (512, 8, 8)
        return self.decode_img(x) # (3, 64, 64)


class Decoder64RGBA(BasicModule):
    def __init__(self, path=None):
        assert path is not None
        super(Decoder64, self).__init__(path)

        self.decode_img = nn.Sequential(
            up_block3x3(512, 256),
            up_block3x3(256, 128),
            up_block3x3(128, 64)
        )
        self.rgb = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
        self.alpha = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):    # (512, 8, 8)
        tmp = self.decode_img(x)
        return self.rgb(tmp)


class Discriminator64(BasicModule):
    def __init__(self, path=None):
        assert path is not None
        super(Discriminator64, self).__init__(path)

        self.logit = nn.Sequential(    # (3, 64, 64)
            strided_conv4x4(3, 64),    # (64, 32, 32)
            nn.LeakyReLU(0.2),
            strided_conv4x4(64, 128),  # (128, 16, 16)
            nn.LeakyReLU(0.2),
            strided_conv4x4(128, 256), # (256, 8, 8)
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, kernel_size=4, padding=1), # (1, 8, 8)
            nn.Sigmoid()
        )

    def forward(self, x):    # (512, 8, 8)
        return self.logit(x) # (3, 64, 64)

# ---------------------------------------------------------

class Encoder128(BasicModule):
    def __init__(self, init_dim=128, code_dim=2048, path=None):
        assert path is not None
        super(Encoder128, self).__init__(path)

        self.encode_img = nn.Sequential(
            strided_conv5x5(3, init_dim),
            nn.LeakyReLU(0.1),
            strided_conv5x5(init_dim, init_dim * 2, sep=True),
            nn.LeakyReLU(0.1),
            strided_conv5x5(init_dim * 2, init_dim * 4),
            nn.LeakyReLU(0.1),
            strided_conv5x5(init_dim * 4, init_dim * 8, sep=True),
            nn.LeakyReLU(0.1)
        )
        # # use two linear layers to squeeze image
        # it helps to reduce blur... Dont know why;;
        self.linear = nn.Sequential(
            nn.Linear(init_dim*8 * 8*8, code_dim),
            nn.Linear(code_dim, 1024*8*8)
        )

        self.upscale = up_block3x3(1024, 512)

    def forward(self, x): # (3, 128, 128)
        b, c, w, h = x.shape
        x = self.encode_img(x)
        x = x.view(b, -1) # (2048*8*8)
        x = self.linear(x) # (1024*8*8)
        x = x.view(-1, 1024, 8, 8)
        x = self.upscale(x) # (512, 16, 16)
        return x


class Decoder128(BasicModule):
    def __init__(self, path=None):
        assert path is not None
        super(Decoder128, self).__init__(path)

        self.decode_img = nn.Sequential(
            up_block3x3(512, 384),
            up_block3x3(384, 256-32),
            up_block3x3(256-32, 128),
            nn.Conv2d(128, 3, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):    # (512, 16, 16)
        return self.decode_img(x) # (3, 128, 128)


class Discriminator128(BasicModule):
    def __init__(self, path=None):
        assert path is not None
        super(Discriminator128, self).__init__(path)

        self.logit = nn.Sequential(    # (3, 128, 128)
            strided_conv4x4(3, 64),    # (64, 64, 64) # 3 or 6
            nn.LeakyReLU(0.2),
            strided_conv4x4(64, 128),  # (128, 32, 32)
            nn.LeakyReLU(0.2),
            strided_conv4x4(128, 256), # (256, 16, 16)
            nn.LeakyReLU(0.2),
            strided_conv4x4(256, 512), # (512, 8, 8)
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, padding=1), # (1, 8, 8)
            nn.Sigmoid()
        )

    def forward(self, x):    # (512, 8, 8)
        return self.logit(x) # (3, 64, 64)

# ---------------------------------------------------------


# # Upsale the spatial size by a factor of 2
# def upscale(in_planes, out_planes):
#     block = nn.Sequential(
#         nn.Upsample(scale_factor=2, mode='nearest'),
#         conv3x3(in_planes, out_planes),
#         nn.BatchNorm2d(out_planes),
#         nn.ReLU(True))
#     return block


# class S1ENC(BasicModule):
#     def __init__(self, init_dim=64, code_dim=1024, path=None):
#         assert path is not None
#         super(S1ENC, self).__init__(path)

#         self.encoder = nn.Sequential( # 3, 64, 64
#             # conv3x3(3, init_dim),          # ngf, 64, 64
#             nn.Conv2d(3, init_dim, 4, 2, 1, bias=False), # ngf*2, 32, 32
#             nn.ReLU(True),
#             nn.Conv2d(init_dim, init_dim * 2, 4, 2, 1, bias=False), # ngf*2, 16, 16
#             nn.BatchNorm2d(init_dim * 2),
#             nn.ReLU(True),
#             nn.Conv2d(init_dim * 2, init_dim * 4, 4, 2, 1, bias=False), # 4ngf, 8, 8
#             nn.BatchNorm2d(init_dim * 4),
#             nn.ReLU(True),
#             nn.Conv2d(init_dim * 4, init_dim * 8, 4, 2, 1, bias=False), # 8ngf, 4, 4
#             nn.BatchNorm2d(init_dim * 8),
#             nn.ReLU(True)
#         )

#         # # use two linear layers to squeeze image
#         # it helps to reduce blur... Dont know why;;
#         self.linear1 = nn.Linear(init_dim*8 * 4*4, code_dim)
#         self.linear2 = nn.Linear(code_dim, 1024 * 4*4)

#         self.upscale = upscale(1024, 512)

#     def forward(self, x): # (3, 64, 64)
#         b, c, w, h = x.shape
#         x = self.encoder(x)
#         x = x.view(b, -1) # (1024*4*4)
#         x = self.linear1(x) # (cdim)
#         x = self.linear2(x) # (1024*4*4)
#         x = x.view(-1, 1024, 4, 4)
#         x = self.upscale(x) # (512, 8, 8)
#         return x





# class S1DISC(BasicModule):
#     def __init__(self, path=None):
#         assert path is not None
#         super(S1DISC, self).__init__(path)
#         init_dim = 64
#         # self.encode_img = nn.Sequential(
#         #     nn.Conv2d(6, init_dim, 4, 2, 1, bias=False),
#         #     nn.LeakyReLU(0.2, inplace=True),
#         #     # state size. (init_dim) x 32 x 32
#         #     nn.Conv2d(init_dim, init_dim * 2, 4, 2, 1, bias=False),
#         #     # nn.BatchNorm2d(init_dim * 2),
#         #     nn.LeakyReLU(0.2, inplace=True),
#         #     # state size (init_dim*2) x 16 x 16
#         #     nn.Conv2d(init_dim*2, init_dim * 4, 4, 2, 1, bias=False),
#         #     # nn.BatchNorm2d(init_dim * 4),
#         #     nn.LeakyReLU(0.2, inplace=True),
#         #     # state size (init_dim*4) x 8 x 8
#         #     nn.Conv2d(init_dim*4, 1, 4, 1, 0, bias=False),
#         #     # nn.BatchNorm2d(1),
#         #     # state size (init_dim * 8) x 4 x 4)
#         #     nn.Sigmoid()
#         # )
#         self.encode_img = nn.Sequential(
#             nn.Conv2d(6, init_dim, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (init_dim) x 32 x 32
#             nn.Conv2d(init_dim, init_dim * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(init_dim * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size (init_dim*2) x 16 x 16
#             nn.Conv2d(init_dim*2, init_dim * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(init_dim * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size (init_dim*4) x 8 x 8
#             nn.Conv2d(init_dim*4, init_dim * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(init_dim * 8),
#             # state size (init_dim * 8) x 4 x 4)
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(init_dim * 8, 1, kernel_size=4, stride=4),
#             nn.Sigmoid()
#         )

#     def forward(self, image):
#         logit = self.encode_img(image)
#         return logit


# ### ------ StageII -------

# class ResBlock(nn.Module):
#     def __init__(self, channel_num):
#         super(ResBlock, self).__init__()
#         self.block = nn.Sequential(
#             conv3x3(channel_num, channel_num, bias=False),
#             nn.BatchNorm2d(channel_num),
#             nn.ReLU(True),
#             conv3x3(channel_num, channel_num, bias=False),
#             nn.BatchNorm2d(channel_num))
#         # self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         residual = x
#         out = self.block(x)
#         out += residual
#         # out = self.relu(out)
#         return out


# class S2GEN(BasicModule):
#     def __init__(self, **kwargs):
#         super(S2GEN, self).__init__(**kwargs)
#         ngf = 256 # 128, 196, 256

#         def _make_layer(block, channel_num):
#             layers = []
#             for i in range(2):
#                 layers.append(block(channel_num))
#             return nn.Sequential(*layers)

#         # --> 4ngf x 16 x 16
#         self.encoder = nn.Sequential( # 3, 64, 64
#             conv3x3(3, ngf, bias=False),          # ngf, 64, 64
#             nn.ReLU(True),
#             nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False), # ngf*2, 32, 32
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False), # 4ngf, 16, 16
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True))
#         self.residual = _make_layer(ResBlock, ngf * 4)
#         # --> 2ngf x 32 x 32
#         self.upsample1 = upscale(ngf * 4, ngf * 2)
#         # --> ngf x 64 x 64
#         self.upsample2 = upscale(ngf * 2, ngf)
#         # --> ngf // 2 x 128 x 128
#         self.upsample3 = upscale(ngf, ngf // 2)
#         # --> ngf // 4 x 256 x 256
#         # self.upsample4 = upBlock(ngf // 2, ngf // 4)
#         # --> 3 x 256 x 256
#         self.img = nn.Sequential(
#             conv3x3(ngf // 2, 3, bias=False),
#             nn.Sigmoid())

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.residual(x)

#         x = self.upsample1(x)
#         x = self.upsample2(x)
#         x = self.upsample3(x)
#         # x = self.upsample4(x)

#         x = self.img(x)
#         return x


# class S2DISC(BasicModule):
#     def __init__(self, **kwargs):
#         super(S2DISC, self).__init__(**kwargs)
#         ndf = 96 # 64, 96
#         self.encode_img = nn.Sequential(
#             nn.Conv2d(6, ndf, 4, 2, 1, bias=False),  # 128 * 128 * ndf
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),  # 64 * 64 * ndf * 2
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),  # 32 * 32 * ndf * 4
#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),  # 16 * 16 * ndf * 8
#             nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 16),
#             nn.LeakyReLU(0.2, inplace=True),  # 8 * 8 * ndf * 16
#             # conv3x3(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
#             # nn.BatchNorm2d(ndf * 32),
#             # nn.LeakyReLU(0.2, inplace=True),  # 4 * 4 * ndf * 32
#             conv3x3(ndf * 16, ndf * 8, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),   # 4 * 4 * ndf * 16
#             conv3x3(ndf * 8, ndf * 4, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),   # 4 * 4 * ndf * 8
#             nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=4),
#             nn.Sigmoid()
#         )

#     def forward(self, image):
#         logit = self.encode_img(image)
#         # print(logit.shape)
#         return logit



