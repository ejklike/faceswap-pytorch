import torch as t

from lib.save_fig import imwrite
from lib.utils import mkdir
from lib.logger import Logger
from lib.trainer import *

from .model import *


def define_lossG(fake_img, real_img, device=None):
        # MAE loss
        l1_loss = nn.L1Loss()(fake_img, real_img)

        # Edge loss (similar with total variation loss)
        edge_loss_w = t.mean(t.abs(
            first_order(fake_img, axis=2) - first_order(real_img, axis=2)))
        edge_loss_h = t.mean(t.abs(
            first_order(fake_img, axis=3) - first_order(real_img, axis=3)))

        lossG = l1_loss + 0.01 * (edge_loss_w + edge_loss_h)
        return lossG


class Trainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)

        # ENCODER
        encoder_args = dict(
            path=self.get_path('encoder', None),
            init_dim=256,
            code_dim=1024)
        self.encoder = get_model('encoder', S1ENC, **encoder_args).to(self.device)
        if self.fix_enc:
            print('encoder will not be trained!')
        print('')

    def _train(self, face_id, epoch, dataset, logger):
        self.encoder.train()
        self.decoder.train()

        for sub_idx in range(1, self.sub_epoch + 1):
            dataset.load_data(augment=True)
            dataset.warp_data(warp=True, to=64)
            dataloader = t.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, **self.dataloader_args)
            
            lossG_sum = lossD_sum = 0.
            for batch_idx, (warped, target) in enumerate(dataloader):
                warped, target = warped.to(self.device), target.to(self.device)
                output = self.decoder(self.encoder(warped))
    
                # GENERATOR
                self.optG.zero_grad()
                lossG = define_lossG(output, target, device=self.device)
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

    def train(self, face_id, epoch, dataset, logger):

        # paths to load/save model/optimizer while training
        dec_path = self.get_path('decoder', face_id)
        optG_path = self.get_path('optG', face_id)

        # load decoder
        self.decoder = get_model(
            'decoder_' + face_id, S1DEC, path=dec_path).to(self.device)

        # define optimizer for enc/dec
        paramG = list(self.decoder.parameters())
        if not self.fix_enc:
            paramG += list(self.encoder.parameters())
        self.optG = get_optimizer(self.lrG, optG_path, paramG)

        # train
        self._train(face_id, epoch, dataset, logger)

        # save checkpoints
        print('')
        if not self.fix_enc:
            self.encoder.save(epoch * self.sub_epoch)
        self.decoder.save(epoch * self.sub_epoch)
        save_optimizer(optG_path, self.optG)
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

