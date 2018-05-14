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
parser.add_argument('-o', '--output-dir', dest='output_dir', default='./output/multi',
                    help="output data directory")
parser.add_argument('-m', '--model-dir', dest='model_dir', default='./model/multi',
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

torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True

use_cuda = args.no_cuda is False and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

print('build encoder...', end='')
model_dir = mkdir(args.model_dir)
encoder_path = os.path.join(model_dir, 'encoder.pth')
encoder = FaceEncoder(
    init_dim=args.init_dim, code_dim=args.code_dim, path=encoder_path)
encoder = encoder.to(device)
encoder.load()
print('finished!')

def get_dataloader(face_id):
    data_dir = os.path.join(args.data_dir, face_id)
    transform = transforms.Compose([ToTensor()])
    dataset = FaceImages(data_dir, transform=transform)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

def get_decoder(decoder_path):
    decoder = FaceDecoder(path=decoder_path)
    decoder = decoder.to(device)
    decoder.load()
    return decoder

print('make dataloaders, decoders, and optimizers...')
data_loader = dict()
decoder_path = dict()
optimizer_path = dict()

face_ids = get_face_ids(args.data_dir)
for face_id in face_ids:
    print('Face_id: {}'.format(face_id), end=', ')
    data_loader[face_id] = get_dataloader(face_id)
    decoder_path[face_id] = os.path.join(
        model_dir, 'decoder{}.pth'.format(face_id))
    optimizer_path[face_id] = os.path.join(
        model_dir, 'optimizer{}.pth'.format(face_id))

# Define loss function
# criterion = nn.L1Loss().to(device)
criterion = GLoss().to(device)
output_dir = mkdir(args.output_dir)

def train(epoch, face_id, dataloader, decoder, optimizer, draw_img=False, loop=10):
    encoder.train()
    decoder.train()
    for loop_idx in range(1, loop + 1):
        for batch_idx, (warped, target) in enumerate(dataloader):
            # forward
            warped, target = warped.to(device), target.to(device)
            output = decoder(encoder(warped))
            loss = criterion(output, target)
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
        save_fig(output_dir, epoch, warped, output, target, size=8)

print('\nstart training...\n')
for epoch in range(1, args.epochs + 1):
    inner_loop = 100
    is_save = epoch % args.log_interval == 0
    shuffle(face_ids)
    for face_id in face_ids:
        decoder = get_decoder(decoder_path[face_id])
        parameters = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = get_optimizer(args.lr, optimizer_path[face_id], parameters)

        train(epoch, face_id, data_loader[face_id], decoder, optimizer,
            draw_img=is_save, loop=inner_loop)

        print('')
        decoder.save(epoch)
        print('')

        save_optimizer(optimizer_path[face_id], optimizer)
        del decoder, parameters, optimizer

    print('')
    encoder.save(epoch)