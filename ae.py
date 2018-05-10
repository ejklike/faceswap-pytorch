import os
import argparse

import torch
from torchvision import transforms

from models import *
from lib.utils import mkdir
from lib.data_loader import FaceImages, ToTensor
from lib.save_fig import save_fig


# Training settings
parser = argparse.ArgumentParser(description='PyTorch FACESWAP Example')
parser.add_argument('--face-id', dest='face_id', default='1',
                    help="face id (default: 1)")
parser.add_argument('--init-dim', dest='init_dim', default=32,
                    help="the number of initial channel (default: 32)")
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=100000000, metavar='N',
                    help='number of epochs to train (default: 100000000)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 5e-5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before saving training status')
args = parser.parse_args()

torch.manual_seed(args.seed)

use_cuda = args.no_cuda is False and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

data_dir = mkdir('./data/{}'.format(args.face_id))
model_dir = mkdir('./model/{}'.format(args.face_id))
output_dir = mkdir('./output/{}'.format(args.face_id))

# Load data
print('load data...', end='')
transform = transforms.Compose([ToTensor()])
dataset = FaceImages(data_dir, transform=transform)
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
trn_data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
print('finished!')

# Define model
print('define model...', end='')
encoder_path = os.path.join(model_dir, 'encoder.pth')
decoder_path = os.path.join(model_dir, 'decoder.pth')
encoder = FaceEncoder(init_dim=args.init_dim, path=encoder_path).to(device)
decoder = FaceDecoder(code_dim=args.init_dim * 8, path=decoder_path).to(device)
print('---'*10)
print(encoder)
print('---'*10)
print(decoder)
print('---'*10)
print('(load model...)')
encoder.load()
decoder.load()
print('finished!')

# Loss and Optimizer
criterion = nn.L1Loss().cuda()
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), 
    lr=args.lr,  betas=(0.5, 0.999))

def train(epoch, draw_img=False):
    encoder.train()
    decoder.train()
    for batch_idx, (warped, target) in enumerate(trn_data_loader):
        # forward
        warped, target = warped.to(device), target.to(device)
        output = output = decoder(encoder(warped))
        loss = criterion(output, target)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if draw_img is True and batch_idx == 0:
            save_fig(output_dir, epoch, warped, output, target, size=8)
        if batch_idx % args.log_interval == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}'.format(
                epoch, batch_idx * len(warped), len(trn_data_loader.dataset),
                100. * batch_idx / len(trn_data_loader), loss.item()), end='')

for epoch in range(1, args.epochs + 1):
    is_save = epoch % args.log_interval == 0
    train(epoch, draw_img=is_save)
    if is_save:
        print('')
        encoder.save(epoch)
        decoder.save(epoch)