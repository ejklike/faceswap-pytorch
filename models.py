import torch.nn as nn
import torch.nn.functional as F

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)


class Conv2d_same(nn.Module):
    def __init__(self, out_channel, kernel_size, stride=2):
        super(Conv2d_same, self).__init__()
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, input):
        _, in_channel, w, h = input.shape
        padding = {
            1: (self.kernel_size - 1)//2,
            2: w - 1 - (w - self.kernel_size) // 2
        }[self.stride]
        return nn.Conv2d(
            in_channel, self.out_channel, self.kernel_size,
            stride=self.stride, padding=padding)(input)


##############

# class torch.nn.Conv2d(in_channels, out_channels, kernel_size, 
#                       stride=1, padding=0, dilation=1, groups=1, bias=True)

    # def res_block(self, f):
    #     def block(x):
    #         input_tensor = x
    #         x = Conv2D(f, kernel_size=3, use_bias=False, padding="same")(x)
    #         x = LeakyReLU(alpha=0.2)(x)
    #         x = Conv2D(f, kernel_size=3, use_bias=False, padding="same")(x)
    #         x = add([x, input_tensor])
    #         x = LeakyReLU(alpha=0.2)(x)
    #         return x
    #     return block

def upscale(in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels * 4, kernel_size, padding=1),
        nn.LeakyReLU(0.1),
        nn.PixelShuffle(2))

# class FaceEncoder(nn.Module):
#     def __init__(self):
#         super(FaceEncoder, self).__init__()
#         self.conv1 = self.conv(INPUT_CHANNEL, 128)
#         self.conv2 = self.conv(128, 256)
#         self.conv3 = self.conv(256, 512)
#         self.conv4 = self.conv(512, 1024)

#         self.linear1 = nn.Linear(1024*64*64, 1024)
#         self.linear2 = nn.Linear(1024, 1024*4*4)

#         self.upscale = upscale(1024, 512)

#     def conv(self, in_channels, out_channels, kernel_size=5):
#         # if stride == 2:
#         #   padding = size - 1 - (size - k) // 2
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=34),
#             nn.LeakyReLU(0.1))

#     def forward(self, x):
#         b, c, w, h = x.shape
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = x.view(b, -1)
#         x = self.linear1(x)
#         x = self.linear2(x)
#         x = x.view(-1, 1024, 4, 4)
#         x = self.upscale(x) # (512, 8, 8)
#         return x


# class FaceDecoder(nn.Module):
#     def __init__(self):
#         super(FaceDecoder, self).__init__()
#         self.upscale1 = upscale(512, 256)
#         self.upscale2 = upscale(256, 128)
#         self.upscale3 = upscale(128, 64)
#         # if stride == 1:
#         #   padding = (k - 1) // 2
#         self.conv = nn.Conv2d(64, INPUT_CHANNEL, kernel_size=5, padding=2)

#     def res_block(self, in_channels, out_channels):
#         pass

#     def forward(self, x):    # (512, 8, 8)
#         x = self.upscale1(x) # (256, 16, 16)
#         x = self.upscale2(x) # (128, 32, 32)
#         x = self.upscale3(x) # (64, 64, 64)
#         x = self.conv(x)     # (3, 64, 64)
#         x = F.sigmoid(x)
#         return x

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))


class FaceEncoder(nn.Module):
    def __init__(self):
        super(FaceEncoder, self).__init__()
        self.init_dim = 16

        self.conv1 = self.conv(3, self.init_dim)
        self.conv2 = self.conv(self.init_dim, self.init_dim * 2)
        self.conv3 = self.conv(self.init_dim * 2, self.init_dim * 4)
        self.conv4 = self.conv(self.init_dim * 4, self.init_dim * 8)

        self.code_dim = self.init_dim * 8

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


class FaceDecoder(nn.Module):
    def __init__(self):
        super(FaceDecoder, self).__init__()

        self.code_dim = 128

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