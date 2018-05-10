import os
import argparse

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

parser.add_argument('-i', '--input_dir', dest='input_dir', default='./data',
                    help="input data directory")
parser.add_argument('--init-dim', dest='init_dim', type=int, default=32,
                    help="the number of initial channel (default: 32)")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100000000, metavar='N',
                    help='number of epochs to train (default: 100000000)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 5e-5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True

use_cuda = args.no_cuda is False and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

print('build encoder...', end='')
model_dir = mkdir('./model/multi')
encoder_path = os.path.join(model_dir, 'encoder.pth')
encoder = FaceEncoder(init_dim=args.init_dim, path=encoder_path)
encoder = encoder.to(device)
encoder.load()
print('finished!')

def get_dataloader(face_id):
    data_dir = './data/{}'.format(face_id)
    transform = transforms.Compose([ToTensor()])
    dataset = FaceImages(data_dir, transform=transform)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

def get_decoder_and_optimizer(face_id):
    # decoder
    decoder_path = os.path.join(
        model_dir, 'decoder{}.pth'.format(face_id))
    decoder = FaceDecoder(
        code_dim=args.init_dim * 8, path=decoder_path)
    decoder = decoder.to(device)
    decoder.load()
    # optimizer
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(
        parameters, lr=args.lr,  betas=(0.5, 0.999))
    return decoder, optimizer

print('make dataloaders, decoders, and optimizers...')
dataloader = dict()
decoder = dict()
optimizer = dict()

face_ids = get_face_ids(args.input_dir)
for face_id in face_ids:
    print('Face_id: {}'.format(face_id), end=' ')
    dataloader[face_id] = get_dataloader(face_id)
    decoder[face_id], optimizer[face_id] = \
        get_decoder_and_optimizer(face_id)

# Define loss function
criterion = nn.L1Loss().cuda()
output_dir = mkdir('./output/multi')

def train(epoch, face_id, draw_img=False, loop=10):
    encoder.train()
    decoder[face_id].train()
    for loop_idx in range(1, loop + 1):
        for batch_idx, (warped, target) in enumerate(dataloader[face_id]):
            # forward
            warped, target = warped.to(device), target.to(device)
            output = decoder[face_id](encoder(warped))
            loss = criterion(output, target)
            # backward
            optimizer[face_id].zero_grad()
            loss.backward()
            optimizer[face_id].step()
            if batch_idx % args.log_interval == 0:
                print('\rTrain Epoch: {} (face_id: {}, loop: {}/{}) [{}/{} ({:.0f}%)], Loss: {:.6f}'.format(
                    epoch, face_id, loop_idx, loop, batch_idx * len(warped), len(dataloader[face_id].dataset),
                    100. * batch_idx / len(dataloader[face_id]), loss.item()), end='')
    if draw_img:
        output_dir = mkdir('./output/multi/{}'.format(face_id))
        save_fig(output_dir, epoch, warped, output, target, size=8)


for epoch in range(1, args.epochs + 1):
    inner_loop = 10
    is_save = epoch % args.log_interval == 0
    for fid in face_ids:
        train(epoch, fid, draw_img=is_save, loop=inner_loop)

    if is_save:
        print('')
        encoder.save(epoch)
        for fid in face_ids:
            decoder[fid].save(epoch)