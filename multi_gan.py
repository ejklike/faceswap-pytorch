import os
import argparse
from random import shuffle

import torch
from torch.autograd import Variable
from torchvision import transforms

from models import *
from lib.utils import mkdir, get_face_ids
from lib.data_loader import FaceImages, ToTensor
from lib.save_fig import save_fig

# Training settings
parser = argparse.ArgumentParser(
    description='PyTorch FACESWAP Example')

parser.add_argument('-d', '--data-dir', dest='data_dir', default='./data',
                    help="input data directory")
parser.add_argument('-o', '--output-dir', dest='output_dir', default='./output/gan',
                    help="output data directory")
parser.add_argument('-m', '--model-dir', dest='model_dir', default='./model/gan',
                    help="model pth directory")

parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--init-dim', dest='init_dim', type=int, default=32,
                    help="the number of initial channel (default: 32)")
parser.add_argument('--code-dim', dest='code_dim', type=int, default=512,
                    help="the number of channel in encoded tensor (default: 512)")
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--epochs', type=int, default=100000000, metavar='N',
                    help='number of epochs to train (default: 100000000)')
parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                    help='learning rate (default: 5e-5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()

# Torch Seed
torch.manual_seed(args.seed)

# CUDA/CUDNN setting
torch.backends.cudnn.benchmark = True
use_cuda = args.no_cuda is False and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# MODEL/OUTPUT DIR
model_dir = mkdir(args.model_dir)
output_dir = mkdir(args.output_dir)

# ENCODER
encoder_args = dict(
    path=os.path.join(model_dir, 'encoder.pth'),
    init_dim=args.init_dim,
    code_dim=args.code_dim)
encoder = get_model('encoder', FaceEncoder, device='cuda', **encoder_args)
print('')

## 
# DISCRIMINATOR
disc_args = dict(
    path=os.path.join(model_dir, 'discriminator.pth'))
discriminator = get_model('discriminator', FaceDiscriminator, device='cuda', **disc_args)
print('')

# FACE IDs for training
face_ids = get_face_ids(args.data_dir)
print('Face_id: {} (total: {})\n'.format(', '.join(face_ids), len(face_ids)))

# DATALOADERS for each face_id
print('make dataloaders...', end='')
data_loader = dict()
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
for face_id in face_ids:
    dataset = FaceImages(
        data_dir=os.path.join(args.data_dir, face_id), 
        transform=transforms.Compose([ToTensor()]))
    data_loader[face_id] = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
print('done!')

# path dicts to load/save model/optimizer while training
decoder_path = dict()
optimizer_path = dict()
for face_id in face_ids:
    decoder_path[face_id] = os.path.join(
        model_dir, 'decoder{}.pth'.format(face_id))
    optimizer_path[face_id] = os.path.join(
        model_dir, 'optimizer{}.pth'.format(face_id))

# LOSSES
criterion_basic = BasicLoss().to(device)
## 
criterion_gan = LSGANLoss().to(device)

def train(epoch, face_id, dataloader, decoder, optimizer, draw_img=False, loop=10):
    encoder.train()
    decoder.train()
    ##
    discriminator.train()
    for loop_idx in range(1, loop + 1):
        for batch_idx, (warped, target) in enumerate(dataloader):
            # forward
            warped, target = warped.to(device), target.to(device)
            rgb, mask = decoder(encoder(warped))
            loss = criterion_basic(rgb, target)
            
            output = rgb * mask + warped * (1-mask)
            dreal, dfake = discriminator(target), discriminator(output)
            loss += criterion_gan(output, dreal, dfake, device=device)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('\rTrain Epoch: {} (face_id: {}, loop: {}/{}) [{}/{} ({:.0f}%)], Loss: {:.6f}'.format(
                    epoch, face_id, loop_idx, loop, batch_idx * len(warped), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss.item()), end='')

    if draw_img:
        output_dir = mkdir(os.path.join(args.output_dir, face_id))
        img_list = [warped, rgb, mask, output, target]
        save_fig(output_dir, epoch * loop, img_list, size=8)

print('\nstart training...\n')
for epoch in range(1, args.epochs + 1):
    inner_loop = 100
    is_save = epoch % args.log_interval == 0
    shuffle(face_ids)
    for face_id in face_ids:
        decoder_args = dict(path=decoder_path[face_id])
        decoder = get_model('decoder_' + face_id, FaceDecoder, device='cuda', **decoder_args)

        parameters = list(encoder.parameters()) + list(decoder.parameters())
        ##
        parameters += list(discriminator.parameters())

        optimizer = get_optimizer(args.lr, optimizer_path[face_id], parameters)

        train(epoch, face_id, data_loader[face_id], decoder, optimizer, 
            draw_img=is_save, loop=inner_loop)

        print('')
        decoder.save(epoch * inner_loop)
        print('')

        save_optimizer(optimizer_path[face_id], optimizer)
        del decoder, parameters, optimizer

    print('')
    encoder.save(epoch * inner_loop)
    ##
    discriminator.save(epoch * inner_loop)