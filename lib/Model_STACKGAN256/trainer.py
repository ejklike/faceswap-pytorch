import torch as t

from lib.save_fig import imwrite
from lib.utils import mkdir
from lib.logger import Logger
from lib.trainer import *
from lib.image_loader import DatasetMerger

from .model import *


def define_lossG(fake_img, real_img, output_neg, device=None):
        # MAE loss
        l1_loss = nn.L1Loss()(fake_img, real_img)

        # Edge loss (similar with total variation loss)
        edge_loss_w = t.mean(t.abs(
            first_order(fake_img, axis=2) - first_order(real_img, axis=2)))
        edge_loss_h = t.mean(t.abs(
            first_order(fake_img, axis=3) - first_order(real_img, axis=3)))

        # LS-GAN loss
        loss_G_fake = nn.MSELoss()(output_neg, t.ones(output_neg.shape).to(device))

        lossG = l1_loss + 0.5 * loss_G_fake + 0.01 * (edge_loss_w + edge_loss_h)
        return lossG


def define_lossD(output_pos, output_neg, device=None):
        # LS-GAN loss
        # MSE for real
        loss_D_real = nn.MSELoss()(output_pos, t.ones(output_pos.shape).to(device))
        # MSE for fake
        loss_D_fake = nn.MSELoss()(output_neg, t.zeros(output_neg.shape).to(device))

        return loss_D_real + loss_D_fake


class Trainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)
        self.fix_enc = True
        self.fix_dec = True

        # ENCODER
        encoder_args = dict(
            path=self.get_path('encoder', None),
            init_dim=256,
            code_dim=1024)
        self.encoder = get_model('encoder', S1ENC, **encoder_args).to(self.device)
        if self.fix_enc:
            print('encoder (from StageI) will not be trained!')
        if self.fix_enc:
            print('decoder (from StageI) will not be trained!')
        print('')
        

    def _train(self, face_id, epoch, dataset, logger):
        self.netG.train()
        self.netD.train()

        for sub_idx in range(1, self.sub_epoch + 1):
            merged_dataset = DatasetMerger()
            dataset.load_data(augment=True)
            dataset.warp_data(warp=True, to=64)
            merged_dataset.append(dataset)
            dataset.load_data(augment=True)
            dataset.warp_data(warp=True, to=256)
            merged_dataset.append(dataset)
            
            dataloader = t.utils.data.DataLoader(
                merged_dataset, batch_size=self.batch_size, shuffle=True, **self.dataloader_args)
            
            lossG_sum = lossD_sum = 0.
            for batch_idx, ((warp64, real64), (warp256, real256)) in enumerate(dataloader):
                warp64, real64 = warp64.to(self.device), real64.to(self.device)
                warp256, real256 = warp256.to(self.device), real256.to(self.device)
                
                fake_s1 = self.decoder(self.encoder(warp64))
                input_s2 = t.cat([fake_s1.detach(), warp64], 1)
                fake_s2 = self.netG(input_s2)

                # DISCRIMINATOR
                self.optD.zero_grad()
                output_pos = self.netD(t.cat([real256, warp256], 1))
                output_neg = self.netD(t.cat([fake_s2.detach(), warp256], 1))
                lossD = define_lossD(output_pos, output_neg, device=self.device)
                lossD.backward()
                self.optD.step()

                # GENERATOR
                self.optG.zero_grad()
                output_neg = self.netD(t.cat([fake_s2, warp256], 1))
                lossG = define_lossG(fake_s2, real256, output_neg, device=self.device)
                lossG.backward()
                self.optG.step()


                print('\rEpoch: {}(face {}; loop {}/{}) '
                        'LossG/D: {:.6f}/{:.6f} [{}/{} ({:.0f}%)]'
                        .format(epoch, face_id, sub_idx, self.sub_epoch, 
                                lossG.item(), lossD.item(), 
                                batch_idx * len(warp64), 
                                len(dataloader.dataset), 
                                100. * batch_idx / len(dataloader)), end='')
                lossG_sum += lossG.item() * len(warp64)
                lossD_sum += lossD.item() * len(warp64)

            net_epoch = self.sub_epoch * (epoch - 1) + sub_idx
            logger.scalar_summary(
                'lossG', lossG_sum / len(dataloader.dataset), net_epoch)
            logger.scalar_summary(
                'lossD', lossD_sum / len(dataloader.dataset), net_epoch)

        this_output_dir = mkdir(os.path.join(self.output_dir, face_id))
        fname = '{}/epoch_{}_s1.png'.format(this_output_dir, epoch * self.sub_epoch)
        img_list = [warp64, fake_s1, real64]
        imwrite(img_list, fname, size=8)
        fname = '{}/epoch_{}_s2.png'.format(this_output_dir, epoch * self.sub_epoch)
        img_list = [warp256, fake_s2, real256]
        imwrite(img_list, fname, size=8)

        dataset.clear_data()

    def train(self, face_id, epoch, dataset, logger):

        # paths to load/save model/optimizer while training
        dec_path = self.get_path('decoder', face_id)
        netG_path = self.get_path('netG', face_id)
        netD_path = self.get_path('netD', face_id)
        optG_path = self.get_path('optG', face_id)
        optD_path = self.get_path('optD', face_id)

        # load decoder
        self.decoder = get_model(
            'decoder_' + face_id, S1DEC, path=dec_path).to(self.device)

        # # fix parameters of stageI GAN
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        # for param in self.decoder.parameters():
        #     param.requires_grad = False

        # load s2 models
        self.netG = get_model(
            'netG_s2_' + face_id, S2GEN, path=netG_path).to(self.device)
        # load discriminator
        self.netD = get_model(
            'netD_s2_' + face_id, S2DISC, path=netD_path).to(self.device)

        # define optimizer for enc/dec
        self.optG = get_optimizer(self.lrG, optG_path, self.netG.parameters())

        # define optimizer for disc
        self.optD = get_optimizer(self.lrD, optD_path, self.netD.parameters())

        # train
        self._train(face_id, epoch, dataset, logger)

        # save checkpoints
        print('')
        self.netG.save(epoch * self.sub_epoch)
        self.netD.save(epoch * self.sub_epoch)
        save_optimizer(optG_path, self.optG)
        save_optimizer(optD_path, self.optD)
        print('')

    def convert(self, source_id, target_id, dataset, size=8):
        dataset.load_data(augment=False)
        dataset.warp_data(warp=False, to=64)
        dataloader = t.utils.data.DataLoader(
            dataset, batch_size=size, shuffle=False, **self.dataloader_args)

        dec_path = self.get_path('decoder', target_id)
        self.decoder = get_model(
            'decoder_' + target_id, S1DEC, path=dec_path).to(self.device)

        this_output_dir = mkdir(
            os.path.join(self.output_dir, 'from_{}_to_{}'.format(source_id, target_id)))

        for batch_idx, (warped, target) in enumerate(dataloader):
            warped, target = warped.to(self.device), target.to(self.device)
            output = self.decoder(self.encoder(warped))

            output_file = os.path.join(this_output_dir, '{}.png'.format(batch_idx))
            img_list = [warped, output]
            imwrite(img_list, output_file, size=size)