import os
import argparse
from random import shuffle

import torch
from torch.autograd import Variable
from torchvision import transforms

from models import *
from lib.utils import mkdir, get_face_ids
from lib.image_loader import FaceImages, ToTensor
from lib.save_fig import imwrite

# Training settings
parser = argparse.ArgumentParser(description='PyTorch FACESWAP Example')

parser.add_argument('-d', '--data-dir', 
                    dest='data_dir', 
                    default='./data',
                    help="input data directory")
parser.add_argument('-n', '--model-name', 
                    dest='model_name', 
                    default='./output/model_name',
                    help="model name (which will become output dir name)")

parser.add_argument('-b', '--batch-size', 
                    dest='batch_size', 
                    type=int, 
                    default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--init-dim', 
                    dest='init_dim', 
                    type=int, 
                    default=128,
                    help="the number of initial channel (default: 32)")
parser.add_argument('--code-dim', 
                    dest='code_dim', 
                    type=int, 
                    default=1024,
                    help="the number of channel in encoded tensor (default: 1024)")
parser.add_argument('--log-interval', 
                    type=int, 
                    default=1,
                    help='how many batches to wait before logging training status')

parser.add_argument('--epochs', 
                    type=int, 
                    default=100000000, 
                    help='number of epochs to train (default: 100000000)')
parser.add_argument('--inner-loop', 
                    type=int, 
                    default=100, 
                    help='number of loop in an epoch for each face (default: 100)')
parser.add_argument('--lr',
                    type=float, 
                    default=5e-5, 
                    help='learning rate (default: 5e-5)')
parser.add_argument('--no-cuda', 
                    action='store_true', 
                    default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', 
                    type=int, 
                    default=1,
                    help='random seed (default: 1)')

parser.add_argument('--fix-enc',
                    action='store_true', 
                    default=False,
                    help='fix encoder and train decoder only')

args = parser.parse_args()

# Torch Seed
torch.manual_seed(args.seed)

# CUDA/CUDNN setting
torch.backends.cudnn.benchmark = True
use_cuda = args.no_cuda is False and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# MODEL/OUTPUT DIR
output_dir = mkdir(os.path.join('./output', args.model_name))

# ENCODER
encoder_args = dict(
    path=os.path.join(output_dir, 'encoder.pth'),
    init_dim=args.init_dim,
    code_dim=args.code_dim)
encoder = get_model('encoder', FaceEncoder, device=device, **encoder_args)
###
if args.fix_enc:
    print('encoder will not be trained!')
    for param in encoder.parameters():
        param.requires_grad = False
###
print('')

# FACE IDs for training
face_ids = get_face_ids(args.data_dir)
print('Face_id: {} (total: {})'.format(', '.join(face_ids), len(face_ids)))
print('')

# DATALOADERS for each face_id
print('make dataloaders...', end='')
dataset = dict()
dataloader_args = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
for face_id in face_ids:
    dataset[face_id] = FaceImages(
        data_dir=os.path.join(args.data_dir, face_id), 
        transform=transforms.Compose([ToTensor()]))
print('done!')

# path dicts to load/save model/optimizer while training
decoder_path = dict()
optimizer_path = dict()
for face_id in face_ids:
    decoder_path[face_id] = os.path.join(
        output_dir, 'decoder{}.pth'.format(face_id))
    optimizer_path[face_id] = os.path.join(
        output_dir, 'optimizer{}.pth'.format(face_id))

# LOSSES
criterion = BasicLoss().to(device)

def train(epoch, face_id, decoder, optimizer, draw_img=False, loop=10):
    encoder.train()
    decoder.train()
    for loop_idx in range(1, loop + 1):
        dataset[face_id].distort_and_shuffle_images()
        dataloader = torch.utils.data.DataLoader(
            dataset[face_id], batch_size=args.batch_size, shuffle=True, **dataloader_args)
        for batch_idx, (warped, target) in enumerate(dataloader):
            # forward
            warped, target = warped.to(device), target.to(device)
            output = decoder(encoder(warped))

            # loss
            loss = criterion(output, target)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('\rTrain Epoch: {} (face_id: {}, loop: {}/{}) [{}/{} ({:.0f}%)], Loss: {:.6f}'
                      .format(epoch, face_id, loop_idx, loop, batch_idx * len(warped), 
                              len(dataloader.dataset), 100. * batch_idx / len(dataloader), 
                              loss.item()), end='')

    if draw_img:
        this_output_dir = mkdir(os.path.join(output_dir, face_id))
        fname = '{}/epoch_{}.png'.format(this_output_dir, epoch * loop)
        img_list = [warped, output, target]
        imwrite(img_list, fname, size=8)


def test(epoch, face_id, decoder, draw_img=False):
    encoder.eval()
    decoder.eval()
    dataset[face_id].distort_and_shuffle_images()
    dataloader = torch.utils.data.DataLoader(
            dataset[face_id], 
            batch_size=args.batch_size, shuffle=True, **dataloader_args)
    for batch_idx, (warped, target) in enumerate(dataloader):
        if batch_idx > 0:
            break
        
        warped, target = warped.to(device), target.to(device)
        output = decoder(encoder(warped))

        if draw_img:
            this_output_dir = mkdir(os.path.join(output_dir, face_id))
            fname = '{}/epoch_{}_test.png'.format(this_output_dir, epoch)
            img_list = [warped, output, target]
            imwrite(img_list, fname, size=8)


print('\nstart training...\n')
for epoch in range(1, args.epochs + 1):
    inner_loop = args.inner_loop
    is_save = epoch % args.log_interval == 0
    shuffle(face_ids)
    for face_id in face_ids:
        decoder_args = dict(path=decoder_path[face_id])
        decoder = get_model('decoder_' + face_id, FaceDecoder, device=device, **decoder_args)
        
        ###
        if args.fix_enc:
            parameters = decoder.parameters()
        else:
            parameters = list(encoder.parameters()) + list(decoder.parameters())
        # 
        ###
        optimizer = get_optimizer(args.lr, optimizer_path[face_id], parameters)
        
        train(epoch, face_id, decoder, optimizer, 
            draw_img=is_save, loop=inner_loop)
        test(epoch * inner_loop, face_id, decoder, draw_img=is_save)

        print('')
        ###
        if not args.fix_enc:
            encoder.save(epoch * inner_loop)
        ###
        decoder.save(epoch * inner_loop)
        save_optimizer(optimizer_path[face_id], optimizer)
        print('')
        
        del decoder, parameters, optimizer
    