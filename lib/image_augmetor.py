import cv2
import numpy as np

from lib.landmark_aligner.umeyama import umeyama


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

def random_augment(image, rotation_range,
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

def _get_warp_xy(coverage, warp_scale):
    size = 256
    center = size //2

    # 5 x 5 grid points in the coverage area
    grid_size = (5, 5)
    range_ = np.linspace(center - coverage//2, 
                         center + coverage//2, 
                         grid_size[0])
    mapx = np.broadcast_to(range_, grid_size)
    mapy = mapx.T

    # warp points randomly
    mapx = mapx + np.random.normal(size=grid_size, scale=warp_scale)
    mapy = mapy + np.random.normal(size=grid_size, scale=warp_scale)

    return mapx, mapy


def random_warp(image, mapx, mapy, to=64):
    """
    get pair of random warped images from aligned face image

    input
        coverage: length (pixel) of image to crop
        warp_scale: warping range of grid points

    output

    """
    x = to // 64

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


class ImageProcessor(object):
    """
    input_image_size = (256, 256)
    output_image_size = (64, 64)
    """
    def __init__(self, random_augment_args, random_warp_args):
        self.random_augment_args = random_augment_args
        self.random_warp_args = random_warp_args

    def read_image(self, image_path):
        try:
            image = normalize(cv2.imread(image_path))
        except TypeError:
            raise Exception("Error while reading image", image_path)
        return image

    def affine_transform(self, image):
        assert image.shape == (256, 256, 3)
        return random_augment(image, **self.random_augment_args)

    def warp(self, image, to=64):
        assert image.shape == (256, 256, 3)
        assert to in [64, 128, 256]
        mapx, mapy = _get_warp_xy(**self.random_warp_args)
        warped_img, target_img = random_warp(image, mapx, mapy, to=to)
        return warped_img, target_img

    def resize(self, image, to=64):
        assert image.shape == (256, 256, 3)
        image = cv2.resize(image, (to, to), interpolation=cv2.INTER_LINEAR).astype('float32')
        return image