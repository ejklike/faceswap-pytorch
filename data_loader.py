

from scandir import scandir
import numpy as np

from torch import from_numpy
from torch.utils.data.dataset import Dataset

from lib.image_loader import ImageLoader

def get_image_paths(directory):
    image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    dir_scanned = list(scandir(directory))

    image_paths = []
    for x in dir_scanned:
        if any([x.name.lower().endswith(ext) for ext in image_extensions]):
            image_paths.append(x.path)
    return image_paths


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

# class FaceImages(Dataset):
#     random_transform_args = {
#         'rotation_range': 10,
#         'zoom_range': 0.05,
#         'shift_range': 0.05,
#         'random_flip': 0.4,
#     }
#     random_warp_args = {
#         'coverage': 160,
#         'warp_scale': 5
#     }

#     def __init__(self, data_dir, transform=None):
#         self.image_paths = get_image_paths(data_dir)
#         self.image_loader = ImageLoader(
#             random_transform_args=self.random_transform_args, 
#             random_warp_args=self.random_warp_args)
#         self.transform = transform
        
#     def __getitem__(self, index):
#         path = self.image_paths[index]
#         distorted_img, target_img = self.image_loader.read_image(path)

#         if self.transform:
#             distorted_img, target_img = self.transform([distorted_img, target_img])

#         return distorted_img, target_img

#     def __len__(self):
#         return len(self.image_paths)

# class FaceImages(Dataset):
#     random_transform_args = {
#         'rotation_range': 10,
#         'zoom_range': 0.05,
#         'shift_range': 0.05,
#         'random_flip': 0.4,
#     }
#     random_warp_args = {
#         'coverage': 160,
#         'warp_scale': 3,
#     }

#     def __init__(self, data_dir, transform=None):
#         self.image_paths = get_image_paths(data_dir)
#         self.image_loader = ImageLoader(
#             random_transform_args=self.random_transform_args, 
#             random_warp_args=self.random_warp_args)
#         self.transform = transform

#         self.images = [self.image_loader.read_image(path) for path in self.image_paths]
        
#     def __getitem__(self, index):
#         index = np.random.randint(0, len(self.images))
#         image = self.images[index]
#         distorted_img, target_img = self.image_loader.transform_image(image)
#         if self.transform is not None:
#             distorted_img, target_img = self.transform([distorted_img, target_img])
#         return distorted_img, target_img

#     def __len__(self):
#         return len(self.image_paths)


class FaceImages(Dataset):
    random_transform_args = {
        'rotation_range': 10,
        'zoom_range': 0.05,
        'shift_range': 0.05,
        'random_flip': 0.4,
    }
    random_warp_args = {
        'coverage': 160,
        'warp_scale': 3,
    }

    def __init__(self, data_dir, transform=None):
        self.image_paths = get_image_paths(data_dir)
        image_loader = ImageLoader(
            random_transform_args=self.random_transform_args, 
            random_warp_args=self.random_warp_args)

        images = [image_loader.read_image(path) for path in self.image_paths]
        self.distorted_imgs, self.target_imgs = [], []
        for image in images:        
            distorted_img, target_img = image_loader.transform_image(image)
            if transform is not None:
                distorted_img, target_img = transform([distorted_img, target_img])
            self.distorted_imgs.append(distorted_img)
            self.target_imgs.append(target_img)
        
    def __getitem__(self, index):
        index = np.random.randint(0, len(self.image_paths))
        distorted_img = self.distorted_imgs[index]
        target_img = self.target_imgs[index]
        return distorted_img, target_img

    def __len__(self):
        return len(self.image_paths)