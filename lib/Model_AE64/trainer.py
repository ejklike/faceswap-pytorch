import torch as t

from lib.save_fig import imwrite
from lib.utils import mkdir
from lib.logger import Logger
from lib.trainer import *

from .model import *


class Trainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)

        self.ENCODER = Encoder64
        self.DECODER = Decoder64
        self.INPUT_SIZE = 64
        INIT_DIM = 256
        CODE_DIM = 1024

        # ENCODER
        encoder_args = dict(
            path=self.get_path('encoder', None),
            init_dim=INIT_DIM,
            code_dim=CODE_DIM)
        self.encoder = self.get_model('encoder', self.ENCODER, **encoder_args)
        if self.fix_enc:
            print('encoder will not be trained!\n')

    def train(self, face_id, epoch, dataset, logger):

        # paths to load/save model/optimizer while training
        dec_path = self.get_path('decoder', face_id)
        optG_path = self.get_path('optG', face_id)

        # load decoder
        self.decoder = self.get_model(
            'decoder_' + face_id, self.DECODER, path=dec_path)

        # define optimizer for enc/dec
        paramG = list(self.decoder.parameters())
        if not self.fix_enc:
            paramG += list(self.encoder.parameters())
        self.optG = get_optimizer(self.lrG, optG_path, paramG)

        # train
        self.train_one_epoch(face_id, epoch, dataset, logger)

        # save checkpoints
        print('')
        if not self.fix_enc:
            self.encoder.save(epoch * self.sub_epoch)
        self.decoder.save(epoch * self.sub_epoch)
        save_optimizer(optG_path, self.optG)
        print('')