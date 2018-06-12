import os
from scandir import scandir
import numpy as np

from torch import from_numpy
from torch.utils.data.dataset import Dataset

from lib.image_augmetor import ImageAugmentor
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
    random_transform_args = {
        'rotation_range': 10,
        'zoom_range': 0.05,
        'shift_range': 0.05,
        'random_flip': 0.4,
    }
    random_warp_args = {
        'coverage': 180, # 600, #160 #180 #200
        'warp_scale': 5,
    }

    def __init__(self, data_dir, transform=None):
        image_paths = get_image_paths(data_dir)
        self.image_augmentor = ImageAugmentor(
            random_transform_args=self.random_transform_args, 
            random_warp_args=self.random_warp_args)
        self.original_images = [
            self.image_augmentor.read_image(path) for path in image_paths]
        self.transform = transform

    def __getitem__(self, index):
        distorted_img = self.distorted_imgs[index]
        target_img = self.target_imgs[index]
        return distorted_img, target_img

    def __len__(self):
        return len(self.original_images)

    def shuffle(self):
        perm_indices = np.random.permutation(len(self))
        self.distorted_imgs = [self.distorted_imgs[i] for i in perm_indices]
        self.target_imgs = [self.target_imgs[i] for i in perm_indices]

    def distort_and_shuffle_images(self):
        self.distorted_imgs, self.target_imgs = [], []
        for image in self.original_images:
            distorted_img, target_img = self.image_augmentor.transform_image(image)
            if self.transform is not None:
                distorted_img, target_img = self.transform([distorted_img, target_img])
            self.distorted_imgs.append(distorted_img)
            self.target_imgs.append(target_img)
        self.shuffle()