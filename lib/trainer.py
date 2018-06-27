import os

import torch as t
import torch.nn.functional as F

from lib.save_fig import imwrite
from lib.utils import mkdir

from .model import *


def get_optimizer(lr, optimizer_path, parameters):
    optimizer = t.optim.Adam(
        parameters, lr=lr,  betas=(0.5, 0.999))
    if os.path.isfile(optimizer_path):
        optimizer.load_state_dict(t.load(optimizer_path))
    return optimizer


def save_optimizer(optimizer_path, optimizer):
    t.save(optimizer.state_dict(), optimizer_path)


class BaseTrainer:
    def __init__(self, output_dir='./output', sub_epoch=100, fix_enc=False, 
                 no_cuda=False, seed=1, lrG=5e-5, lrD=5e-4, batch_size=64, mask_loss=False):
        self.output_dir = output_dir
        self.sub_epoch = sub_epoch
        self.fix_enc = fix_enc
        self.lrG = lrG
        self.lrD = lrD
        self.batch_size = batch_size
        self.mask_loss = mask_loss
        
        # Torch Seed
        t.manual_seed(seed)

        # CUDA/CUDNN setting
        self.use_cuda = no_cuda is False and t.cuda.is_available()
        self.n_gpu = t.cuda.device_count()
        t.backends.cudnn.benchmark = self.use_cuda
        self.device = t.device("cuda" if self.use_cuda else "cpu")
        self.dataloader_args = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
    
        # GAN Loss criterion and label
        self.criterion = nn.MSELoss()

    def get_path(self, name, face_id):
        fname = name + ('' if face_id is None else face_id) + '.pth'
        return os.path.join(self.output_dir, fname)

    def get_model(self, model_name, model_class, **kwargs):
        print('build {}...'.format(model_name))
        model = model_class(**kwargs)
        model.load()
        if self.n_gpu > 1:
            model = nn.DataParallel(model)
        return model.to(self.device)

    def l1_loss(self, fake_img, real_img, mask=False):
        if mask is True:
            mask = (real_img.detach() != 0.).float()
            fake_img, real_img = fake_img * mask, real_img * mask

        return nn.L1Loss()(fake_img, real_img)

    def gan_loss(self, output, label=0):
        labels = t.full(output.shape, label, device=self.device)
        return self.criterion(output, labels)

    def edge_loss(self, fake_img, real_img):
        # Edge loss (similar with total variation loss)
        def first_order(x, axis=1):
            _, _, w, h = x.shape
            if axis == 2:
                left = x[:, :, 0:w-1, :]
                right = x[:, :, 1:w, :]
                return t.abs(left - right)
            elif axis == 3:
                upper = x[:, :, :, 0:h-1]
                lower = x[:, :, :, 1:h]
                return t.abs(upper - lower)
            else:
                return None

        edge_loss_w = t.mean(t.abs(
            first_order(fake_img, axis=2) - first_order(real_img, axis=2)))
        edge_loss_h = t.mean(t.abs(
            first_order(fake_img, axis=3) - first_order(real_img, axis=3)))
        return edge_loss_w + edge_loss_h

    def train_one_epoch(self, face_id, epoch, dataset, logger):
        self.encoder.train()
        self.decoder.train()

        for sub_idx in range(1, self.sub_epoch + 1):
            dataset.load_data(augment=True)
            dataset.warp_data(warp=True, to=self.INPUT_SIZE)
            dataloader = t.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, **self.dataloader_args)
            
            lossG_sum = 0.
            for batch_idx, (warped, target) in enumerate(dataloader):
                warped, target = warped.to(self.device), target.to(self.device)
                output = self.decoder(self.encoder(warped))
    
                # GENERATOR
                self.optG.zero_grad()
                lossG = self.l1_loss(output, target, mask=self.mask_loss)
                lossG.backward()
                self.optG.step()

                print('\rEpoch: {}(face {}; loop {}/{}) '
                        'LossG: {:.6f} [{}/{} ({:.0f}%)]'
                        .format(epoch, face_id, sub_idx, self.sub_epoch, 
                                lossG.item(), 
                                batch_idx * len(warped), 
                                len(dataloader.dataset), 
                                100. * batch_idx / len(dataloader)), end='')
                lossG_sum += lossG.item() * len(warped)
            net_epoch = self.sub_epoch * (epoch - 1) + sub_idx
            logger.scalar_summary(
                'lossG', lossG_sum / len(dataloader.dataset), net_epoch)

        this_output_dir = mkdir(os.path.join(self.output_dir, face_id))
        fname = '{}/epoch_{}.png'.format(this_output_dir, epoch * self.sub_epoch)
        img_list = [warped, output, target]
        imwrite(img_list, fname, size=8)

        dataset.clear_data()


    def train_one_epoch_masked(self, face_id, epoch, dataset, logger):
        self.encoder.train()
        self.decoder.train()

        for sub_idx in range(1, self.sub_epoch + 1):
            dataset.load_data(augment=True)
            dataset.warp_data(warp=True, to=self.INPUT_SIZE)
            dataloader = t.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, **self.dataloader_args)
            
            lossG_sum = 0.
            for batch_idx, (warped, target) in enumerate(dataloader):
                warped, target = warped.to(self.device), target.to(self.device)
                rgb, alpha = self.decoder(self.encoder(warped))
                output = alpha * rgb + (1-alpha) * warped

                # GENERATOR
                self.optG.zero_grad()
                lossG = self.l1_loss(output, target)
                lossG += self.l1_loss(rgb, target)
                lossG.backward()
                self.optG.step()

                print('\rEpoch: {}(face {}; loop {}/{}) '
                        'LossG: {:.6f} [{}/{} ({:.0f}%)]'
                        .format(epoch, face_id, sub_idx, self.sub_epoch, 
                                lossG.item(), 
                                batch_idx * len(warped), 
                                len(dataloader.dataset), 
                                100. * batch_idx / len(dataloader)), end='')
                lossG_sum += lossG.item() * len(warped)
            net_epoch = self.sub_epoch * (epoch - 1) + sub_idx
            logger.scalar_summary(
                'lossG', lossG_sum / len(dataloader.dataset), net_epoch)

        this_output_dir = mkdir(os.path.join(self.output_dir, face_id))
        fname = '{}/epoch_{}.png'.format(this_output_dir, epoch * self.sub_epoch)
        img_list = [warped, rgb, alpha, output, target]
        imwrite(img_list, fname, size=8)

        dataset.clear_data()

    def train_one_epoch_step(self, face_id, epoch, dataset, logger):
        self.encoder.train()
        self.decoder.train()

        for sub_idx in range(1, self.sub_epoch + 1):
            dataset.load_data(augment=True)
            dataset.warp_data(warp=True, to=self.INPUT_SIZE)
            dataloader = t.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, **self.dataloader_args)
            
            lossG_sum = 0.
            for batch_idx, (warped, target) in enumerate(dataloader):
                warped, target = warped.to(self.device), target.to(self.device)
                output64, output128 = self.decoder(self.encoder(warped))
                target64 = F.upsample(target, size=(64, 64), mode='bilinear')
    
                # GENERATOR
                self.optG.zero_grad()
                lossG = self.l1_loss(output128, target, mask=self.mask_loss)
                lossG += self.l1_loss(output64, target64, mask=self.mask_loss)
                lossG.backward()
                self.optG.step()

                print('\rEpoch: {}(face {}; loop {}/{}) '
                        'LossG: {:.6f} [{}/{} ({:.0f}%)]'
                        .format(epoch, face_id, sub_idx, self.sub_epoch, 
                                lossG.item(), 
                                batch_idx * len(warped), 
                                len(dataloader.dataset), 
                                100. * batch_idx / len(dataloader)), end='')
                lossG_sum += lossG.item() * len(warped)
            net_epoch = self.sub_epoch * (epoch - 1) + sub_idx
            logger.scalar_summary(
                'lossG', lossG_sum / len(dataloader.dataset), net_epoch)

        this_output_dir = mkdir(os.path.join(self.output_dir, face_id))
        fname = '{}/epoch_{}.png'.format(this_output_dir, epoch * self.sub_epoch)
        img_list = [warped, output, target]
        imwrite(img_list, fname, size=8)

        dataset.clear_data()

    def train_one_epoch_GAN(self, face_id, epoch, dataset, logger):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()

        for sub_idx in range(1, self.sub_epoch + 1):
            dataset.load_data(augment=True)
            dataset.warp_data(warp=True, to=self.INPUT_SIZE)
            dataloader = t.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, **self.dataloader_args)
            
            lossG_sum = lossD_sum = 0.
            for batch_idx, (warped, target) in enumerate(dataloader):
                warped, target = warped.to(self.device), target.to(self.device)
                output = self.decoder(self.encoder(warped))

                # DISCRIMINATOR
                self.optD.zero_grad()
                output_pos = self.discriminator(t.cat([target, warped], 1))
                output_neg = self.discriminator(t.cat([output.detach(), warped], 1))
                lossD = self.gan_loss(output_pos, label=1) + self.gan_loss(output_neg, label=0)
                lossD.backward()
                self.optD.step()

                # GENERATOR
                self.optG.zero_grad()
                output_neg = self.discriminator(t.cat([output, warped], 1))
                lossG = self.l1_loss(output, target, mask=self.mask_loss)
                lossG += 0.5 * self.gan_loss(output_neg, label=1)
                lossG.backward()
                self.optG.step()

                print('\rEpoch: {}(face {}; loop {}/{}) '
                        'LossG/D: {:.6f}/{:.6f} [{}/{} ({:.0f}%)]'
                        .format(epoch, face_id, sub_idx, self.sub_epoch, 
                                lossG.item(), lossD.item(), 
                                batch_idx * len(warped), 
                                len(dataloader.dataset), 
                                100. * batch_idx / len(dataloader)), end='')
                lossG_sum += lossG.item() * len(warped)
                lossD_sum += lossD.item() * len(warped)

            net_epoch = self.sub_epoch * (epoch - 1) + sub_idx
            logger.scalar_summary(
                'lossG', lossG_sum / len(dataloader.dataset), net_epoch)
            logger.scalar_summary(
                'lossD', lossD_sum / len(dataloader.dataset), net_epoch)

        this_output_dir = mkdir(os.path.join(self.output_dir, face_id))
        fname = '{}/epoch_{}.png'.format(this_output_dir, epoch * self.sub_epoch)
        img_list = [warped, output, target]
        imwrite(img_list, fname, size=8)

        dataset.clear_data()

    def save_model(self, model, num_epoch):
        if self.n_gpu > 1:
            model = model.module
        model.save(num_epoch)

    def convert(self, target_id, dataset, size=8):
        dataset.load_data(augment=False)
        dataset.warp_data(warp=False, to=self.INPUT_SIZE)
        dataloader = t.utils.data.DataLoader(
            dataset, batch_size=size, shuffle=False, **self.dataloader_args)

        dec_path = self.get_path('decoder', target_id)
        self.decoder = self.get_model(
            'decoder_' + target_id, self.DECODER, path=dec_path).to(self.device)

        this_output_dir = mkdir(
            os.path.join(self.output_dir, 'swap_to_{}'.format(target_id)))

        for batch_idx, (warped, target) in enumerate(dataloader):
            warped, target = warped.to(self.device), target.to(self.device)
            output = self.decoder(self.encoder(warped))

            output_file = os.path.join(this_output_dir, '{}.png'.format(batch_idx))
            img_list = [warped, output]
            imwrite(img_list, output_file, size=size)