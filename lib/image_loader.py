import os
from scandir import scandir
from random import shuffle
import numpy as np

from torch import from_numpy
from torch.utils.data.dataset import Dataset

from lib.image_augmetor import ImageProcessor
from lib.utils import get_image_paths


class ToTensor(object):
    """
    Convert ndarray image to Tensor.
    (https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
    """
    def __call__(self, images):
        def _to_tensor(image):
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            image = image.transpose((2, 0, 1)).astype(np.float32)
            return from_numpy(image)

        images = [_to_tensor(img) for img in images]
        return images


class FaceImages(Dataset):
    def __init__(self, data_dir, transform=None,
                 random_augment_args=None, random_warp_args=None):
        # assert random_augment_args is not None
        # assert random_warp_args is not None

        self.image_paths = get_image_paths(data_dir)
        self.image_processor = ImageProcessor(
            random_augment_args=random_augment_args,
            random_warp_args=random_warp_args)
        self.transform = transform

        self.distorted_imgs, self.target_imgs = None, None

    def __getitem__(self, index):
        distorted_img = self.distorted_imgs[index]
        target_img = self.target_imgs[index]
        return distorted_img, target_img

    def __len__(self):
        return len(self.image_paths)

    def _shuffle(self):
        shuffle(self.image_paths)

    def load_data(self, augment=True, warp=True, shuffle=True, to=64):
        if shuffle:
            self._shuffle()

        self.distorted_imgs, self.target_imgs = [], []
        for path in self.image_paths:
            image = self.image_processor.read_image(path)
            if augment:
                image = self.image_processor.affine_transform(image)

            distorted_img, target_img = None, None
            if warp:
                distorted_img, target_img = self.image_processor.warp(image, to=to)
            else:
                distorted_img = target_img = self.image_processor.resize(image, to=to)

            if self.transform is not None:
                distorted_img, target_img = self.transform([distorted_img, target_img])
            self.distorted_imgs.append(distorted_img)
            self.target_imgs.append(target_img)

    def clear_data(self):
        # del self.distorted_imgs, self.target_imgs
        self.distorted_imgs, self.target_imgs = None, None