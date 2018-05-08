import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from models import *
from data_loader import FaceImages, ToTensor

# Training settings
parser = argparse.ArgumentParser(description='PyTorch FACESWAP Example')

# parser.add_argument('-i', '--input_dir', dest='input_dir', default='./data',
#                     help="input data directory")
parser.add_argument('--face-id', dest='face_id', default='1',
                    help="face id (default: 1)")

parser.add_argument('--batch-size', type=int, default=500, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
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
args = parser.parse_args()

torch.manual_seed(args.seed)

use_cuda = args.no_cuda is False and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load data
print('load data...', end='')
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
transform = transforms.Compose([ToTensor()])
data_loader = torch.utils.data.DataLoader(
    FaceImages('./data/faces/{}'.format(args.face_id), transform=transform), 
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    FaceImages('./data/faces/{}'.format(args.face_id), transform=transform), 
    batch_size=args.test_batch_size, shuffle=False, **kwargs)
print('finished!')

# Define model
print('define model...', end='')
encoder = FaceEncoder().to(device)
decoder = FaceDecoder().to(device)

model = AutoEncoder(encoder, decoder)
parameters = model.parameters()
print('finished!')

# Define optimizer
optimizer = torch.optim.Adam(parameters, lr=args.lr,  betas=(0.5, 0.999))

# Define loss function
criterion = F.l1_loss
# criterion = F.mse_loss

try:
    encoder_pkl_name = './model/encoder_face_{}.pkl'.format(args.face_id)
    decoder_pkl_name = './model/decoder_face_{}.pkl'.format(args.face_id)
    encoder = torch.load(encoder_pkl_name)
    decoder = torch.load(decoder_pkl_name)
    print("\n--------model restored--------\n")
except:
    print("\n--------model not restored--------\n")
    pass

def train(epoch):
    model.train()
    for batch_idx, (warped, target) in enumerate(data_loader):
        # warped, target = warped.to(device), target.to(device)
        warped, target = target.clone().to(device), target.to(device)
        output = model(warped)
        loss = criterion(output, target)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}'.format(
                epoch, batch_idx * len(warped), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()), end='')


def save_fig(epoch, input_data, output_data, target_data, size=8):
    # initialize figure
    scale = 20
    fig, axis = plt.subplots(3, size, figsize=(5 * scale, 2 * scale))

    input_data = input_data.data.cpu().numpy()
    output_data = output_data.data.cpu().numpy()
    target_data = target_data.data.cpu().numpy()

    def change_axis_and_colorspace(img):
        img = np.reshape(img, (3, 64, 64)).transpose((1, 2, 0))
        img = img[:, :, ::-1] # channel permutation
        return img

    for i in range(size):
        axis[0][i].imshow(change_axis_and_colorspace(input_data[i]))
        axis[0][i].set_xticks(()); axis[0][i].set_yticks(())

        axis[1][i].clear()
        axis[1][i].imshow(change_axis_and_colorspace(output_data[i]))
        axis[1][i].set_xticks(()); axis[1][i].set_yticks(())

        axis[2][i].clear()
        axis[2][i].imshow(change_axis_and_colorspace(target_data[i]))
        axis[2][i].set_xticks(()); axis[1][i].set_yticks(())
    
    axis[0][0].set_ylabel('input_data')
    axis[1][0].set_ylabel('output_data')
    axis[2][0].set_ylabel('target_data')

    to_folder = './output/{}'.format(args.face_id)
    if not os.path.exists(to_folder):
        os.mkdir(to_folder)
    fig.tight_layout()
    plt.savefig('{}/epoch_{:00000}.png'.format(to_folder, epoch))
    plt.close()

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for warped, target in test_loader:
            warped, target = target.clone().to(device), target.to(device)
            output = model(warped)
            test_loss += criterion(output, target, size_average=False).item() # sum up batch loss
        save_fig(epoch, warped, output, target, size=8)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))
    return test_loss


for epoch in range(1, args.epochs + 1):
    train(epoch)
    
    best_test_loss = 1e+5
    if epoch % args.log_interval == 0:
        test_loss = test(epoch)
        print(best_test_loss, test_loss)
        if test_loss - best_test_loss <= 0:
            best_test_loss = test_loss
            torch.save(encoder, encoder_pkl_name)
            torch.save(decoder, decoder_pkl_name)
            print("--------model saved-------- (best_test_loss = {})\n".format(best_test_loss))