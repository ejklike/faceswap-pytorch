import os

import torch as t


def get_model(model_name, model_class, **kwargs):
    print('build {}...'.format(model_name))
    model = model_class(**kwargs)
    model.load()
    return model


def get_optimizer(lr, optimizer_path, parameters):
    optimizer = t.optim.Adam(
        parameters, lr=lr,  betas=(0.5, 0.999))
    if os.path.isfile(optimizer_path):
        optimizer.load_state_dict(t.load(optimizer_path))
    return optimizer


def save_optimizer(optimizer_path, optimizer):
    t.save(optimizer.state_dict(), optimizer_path)


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


class BaseTrainer:
    def __init__(self, output_dir='./output', sub_epoch=100, fix_enc=False, 
                 no_cuda=False, seed=1, lrG=5e-5, lrD=5e-4, batch_size=64):
        self.output_dir = output_dir
        self.sub_epoch = sub_epoch
        self.fix_enc = fix_enc
        self.lrG = lrG
        self.lrD = lrD
        self.batch_size = batch_size
        
        # Torch Seed
        t.manual_seed(seed)

        # CUDA/CUDNN setting
        use_cuda = no_cuda is False and t.cuda.is_available()
        t.backends.cudnn.benchmark = use_cuda
        self.device = t.device("cuda" if use_cuda else "cpu")
        self.dataloader_args = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    def get_path(self, name, face_id):
        fname = name + ('' if face_id is None else face_id) + '.pth'
        return os.path.join(self.output_dir, fname)
