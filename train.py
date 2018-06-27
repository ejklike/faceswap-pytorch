import os
import argparse
from random import shuffle

from torchvision import transforms

from lib.utils import mkdir, get_face_ids
from lib.image_loader import MultiResFaceImages, ToTensor
from lib.loader import PluginLoader
from lib.logger import Logger

# Training settings
parser = argparse.ArgumentParser(description='PyTorch FACESWAP Example')

parser.add_argument('-d', '--data-dir', 
                    dest='data_dir', 
                    default='./data',
                    help="input data directory")

parser.add_argument('-m', '--model-name', 
                    dest='model_name', 
                    choices=PluginLoader.get_available_models(),
                    default=PluginLoader.get_default_model(),
                    help="select a model to train")

parser.add_argument('-o', '--output-dir', 
                    dest='output_dir', 
                    default='output_dir',
                    help="output dir name (which will become output dir name)")

parser.add_argument('-b', '--batch-size', 
                    dest='batch_size', 
                    type=int, 
                    default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', 
                    type=int, 
                    default=100, 
                    help='number of epochs to train (default: 100)')
parser.add_argument('--sub-epoch', 
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
parser.add_argument('--mask-loss',
                    action='store_true', 
                    default=False,
                    help='mask loss for segmented image')

args = parser.parse_args()

# FACE IDs for training
face_ids = get_face_ids(args.data_dir)
print('Face_id: {} (total: {})'.format(', '.join(face_ids), len(face_ids)))
print('')

# DATALOADERS for each face_id
print('make dataloaders...', end='')
dataset = dict()
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
    dataset[face_id] = MultiResFaceImages(
        data_dir=os.path.join(args.data_dir, face_id), 
        transform=transforms.Compose([ToTensor()]),
        random_augment_args=random_augment_args,
        random_warp_args=random_warp_args)
print('done!')

# OUTPUT DIR
output_dir = mkdir(os.path.join('./output', args.output_dir))

# TRAINER
Trainer = PluginLoader.get_trainer(args.model_name)
trainer = Trainer(
    output_dir=output_dir, 
    sub_epoch=args.sub_epoch, 
    no_cuda=args.no_cuda,
    seed=args.seed,
    lrG=args.lr,
    lrD=args.lr,
    batch_size=args.batch_size,
    mask_loss=args.mask_loss)

# logger to record loss
logger = dict()
for face_id in face_ids:
    logger[face_id] = Logger(mkdir(os.path.join(output_dir, 'log', face_id)))

print('\nstart training...\n')
for epoch in range(1, args.epochs + 1):
    shuffle(face_ids)
    for face_id in face_ids:
        trainer.train(face_id, epoch, dataset[face_id], logger[face_id])