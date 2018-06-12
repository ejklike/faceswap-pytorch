import cv2
import numpy as np

from lib.landmark_aligner.umeyama import umeyama

# TODO: warp_scale, coverage ==> main args from argparse


def get_random_affine_transformation(width, height, 
        rotation_range, zoom_range, shift_range):
    """
    get a random affine transformation matrix

    input:
        rotation_range: angle (degree, 0 ~ 180)
        zoom_range: increase/decrease (0 ~ 1) 
        shift_range: (0 ~ inf)

    output:
        a 2x3 matrix for affine transformation
    """
    # rotation and scale
    center = (width // 2, height // 2)
    angle = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    mat = cv2.getRotationMatrix2D(center, angle, scale)

    # translation
    tx = np.random.uniform(-shift_range, shift_range) * width
    ty = np.random.uniform(-shift_range, shift_range) * height
    mat[:, 2] += (tx, ty)

    return mat

def random_transform(image, rotation_range,
                     zoom_range, shift_range, random_flip):
    """
    rotate, scale, and translate an image in random
    """
    h, w, _ = image.shape

    # affine transform
    size = (w, h)
    mat = get_random_affine_transformation(
        w, h, rotation_range, zoom_range, shift_range)
    result = cv2.warpAffine(
        image, mat, size, borderMode=cv2.BORDER_REPLICATE)

    # random flip
    if np.random.random() < random_flip:
        result = result[:, ::-1]
    return result

def get_randomly_warped_grid(coverage=180, warp_scale=5, image_size=256, n_grid=5):
    """
    return (n_grid, n_grid) points of warped points in the coverage region
    ---
    input
        coverage: length (pixel) of image to crop
        warp_scale: warping range of grid points
    output

    """
    # 5 x 5 grid points in the coverage area
    center = image_size //2
    range_ = np.linspace(center - coverage//2, 
                         center + coverage//2, 
                         n_grid)
    mapx = np.broadcast_to(range_, (n_grid, n_grid))
    mapy = mapx.T

    # warp grid points randomly
    random_noise = np.random.normal(size=(n_grid, n_grid), scale=warp_scale)
    mapx = mapx + random_noise
    mapy = mapy + random_noise

    return mapx, mapy

def random_warp(image, mapx, mapy, magnify_factor=1):
    """
    get pair of random warped images from aligned face image

    input

        magnify_factor: for (64, 64) * magnify_factor image
    output
        warped image of size (64, 64) * magnify_factor
    """
    assert image.shape == (256,256,3)

    # densify grid points (5x5 -> 64x64)
    # (side values are removed since their values are 
    #  almost same and thus make duplicate neighboring pixels)
    interp_mapx = cv2.resize(mapx, (80*x, 80*x))[8*x:72*x, 8*x:72*x].astype('float32')
    interp_mapy = cv2.resize(mapy, (80*x, 80*x))[8*x:72*x, 8*x:72*x].astype('float32')

    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

    src_points = np.stack([mapx.ravel(), mapy.ravel() ], axis=-1)
    dst_points = np.mgrid[0:65*x:16*x, 0:65*x:16*x].T.reshape(-1,2)
    mat = umeyama(src_points, dst_points, True)[0:2]

    target_image = cv2.warpAffine(image, mat, (64*x, 64*x))

    return warped_image, target_image

def normalize(img):
    """
    scale from (0, 255) to (0, 1)
    """
    return img / 255.0

class ImageAugmentor(object):
    """
    input_image_size = (256, 256)
    output_image_size = (64, 64), (128, 128)
    """
    def __init__(self, random_transform_args, random_warp_args):
        self.random_transform_args = random_transform_args
        self.random_warp_args = random_warp_args

    def read_image(self, image_path):
        try:
            image = normalize(cv2.imread(image_path))
        except TypeError:
            raise Exception("Error while reading image", image_path)
        return image

    def transform_image(self, image):
        assert image.shape == (256, 256, 3)

        # np.random.seed()
        transformed_img = random_transform(image, **self.random_transform_args)

        mapx, mapy = get_randomly_warped_grid(**self.random_warp_args)
        warped_64, target_64 = random_warp(transformed_img, mapx, mapy, magnify_factor=1)
        warped_128, target_128 = random_warp(transformed_img, mapx, mapy, magnify_factor=2)

        return (warped_img, target_img