import os
import argparse
from random import shuffle

import torch
from torchvision import transforms

from models import *
from lib.utils import mkdir, get_face_ids
from lib.image_loader import FaceImages, ToTensor
from lib.save_fig import imwrite
from lib.logger import Logger

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
encoder = get_model('encoder', S1ENC, **encoder_args).to(device)
if args.fix_enc:
    print('encoder will not be trained!')
    for param in encoder.parameters():
        param.requires_grad = False
print('')

# FACE IDs for training
face_ids = get_face_ids(args.data_dir)
print('Face_id: {} (total: {})'.format(', '.join(face_ids), len(face_ids)))
print('')

# DATALOADERS for each face_id
print('make dataloaders...', end='')
dataset = dict()
dataloader_args = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
random_augment_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,
}
random_warp_args = {
    'coverage': 256, # 600, #160 #180 #200 #256
    'warp_scale': 5,
}
for face_id in face_ids:
    dataset[face_id] = FaceImages(
        data_dir=os.path.join(args.data_dir, face_id), 
        transform=transforms.Compose([ToTensor()]),
        random_augment_args=random_augment_args,
        random_warp_args=random_warp_args)
print('done!')

# path dicts to load/save model/optimizer while training
dec_path = dict()
disc_path = dict()
optG_path = dict()
optD_path = dict()
for face_id in face_ids:
    dec_path[face_id] = os.path.join(
        output_dir, 'decoder{}.pth'.format(face_id))
    disc_path[face_id] = os.path.join(
        output_dir, 'discriminator{}.pth'.format(face_id))
    optG_path[face_id] = os.path.join(
        output_dir, 'optG{}.pth'.format(face_id))
    optD_path[face_id] = os.path.join(
        output_dir, 'optD{}.pth'.format(face_id))

logger = dict()
for face_id in face_ids:
    logger[face_id] = Logger(mkdir(os.path.join(output_dir, 'log', face_id)))

def train(epoch, face_id, decoder, discriminator, optG, optD, draw_img=False, loop=10):
    encoder.train()
    decoder.train()
    discriminator.train()
    for loop_idx in range(1, loop + 1):
        dataset[face_id].load_data(augment=True, warp=True, shuffle=True, to=64)
        dataloader = torch.utils.data.DataLoader(
            dataset[face_id], batch_size=args.batch_size, shuffle=True, **dataloader_args)
        lossG_sum = lossD_sum = 0.
        for batch_idx, (warped, target) in enumerate(dataloader):
            warped, target = warped.to(device), target.to(device)
            output = decoder(encoder(warped))

            # DISCRIMINATOR
            optD.zero_grad()
            output_pos = discriminator(t.cat([target, warped], 1))
            output_neg = discriminator(t.cat([output.detach(), warped], 1))
            lossD = define_lossD(output_pos, output_neg, device=device)
            lossD.backward()
            optD.step()

            # GENERATOR
            optG.zero_grad()
            output_neg = discriminator(t.cat([output, warped], 1))
            lossG = define_lossG(output, target, output_neg, device=device)
            lossG.backward()
            optG.step()

            if batch_idx % args.log_interval == 0:
                print('\rEpoch: {}(face {}; loop {}/{}) LossG/D: {:.6f}/{:.6f} [{}/{} ({:.0f}%)]'
                      .format(epoch, face_id, loop_idx, loop, 
                              lossG.item(), lossD.item(), batch_idx * len(warped), 
                               len(dataloader.dataset), 100. * batch_idx / len(dataloader)), end='')
            lossG_sum += lossG.item() * len(warped)
            lossD_sum += lossD.item() * len(warped)
        net_epoch = loop * (epoch - 1) + loop_idx
        logger[face_id].scalar_summary(
            'lossG', lossG_sum / len(dataloader.dataset), net_epoch)
        logger[face_id].scalar_summary(
            'lossD', lossD_sum / len(dataloader.dataset), net_epoch)

    if draw_img:
        this_output_dir = mkdir(os.path.join(output_dir, face_id))
        fname = '{}/epoch_{}.png'.format(this_output_dir, epoch * loop)
        img_list = [warped, output, target]
        imwrite(img_list, fname, size=8)

    dataset[face_id].clear_data()


print('\nstart training...\n')
# summary_writer = FileWriter(output_dir)
for epoch in range(1, args.epochs + 1):
    is_save = epoch % args.log_interval == 0
    shuffle(face_ids)
    for face_id in face_ids:
        # load decoder
        decoder_args = dict(path=dec_path[face_id])
        decoder = get_model(
            'decoder_' + face_id, S1DEC, **decoder_args).to(device)

        # load discriminator
        discriminator_args = dict(path=disc_path[face_id])
        discriminator = get_model(
            'discriminator_' + face_id, S1DISC, **discriminator_args).to(device)

        # define optimizer
        paramG = list(decoder.parameters())
        if not args.fix_enc:
            paramG += list(encoder.parameters())
        optG = get_optimizer(args.lr, optG_path[face_id], paramG)
        optD = get_optimizer(args.lr, optD_path[face_id], discriminator.parameters())

        # train
        train(epoch, face_id, decoder, discriminator, optG, optD, 
              draw_img=is_save, loop=args.inner_loop)

        print('')
        if not args.fix_enc:
            encoder.save(epoch * args.inner_loop)
        decoder.save(epoch * args.inner_loop)
        discriminator.save(epoch * args.inner_loop)
        save_optimizer(optG_path[face_id], optG)
        save_optimizer(optD_path[face_id], optD)
        print('')

        del decoder, discriminator, paramG, optG, optD