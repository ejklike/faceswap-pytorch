import os
from scandir import scandir
import cv2
import numpy as np

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory

def get_face_ids(input_dir='./data'):
    return sorted(os.listdir(input_dir))

def get_image_paths(directory):
    image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    dir_scanned = list(scandir(directory))

    image_paths = []
    for x in dir_scanned:
        if any([x.name.lower().endswith(ext) for ext in image_extensions]):
            image_paths.append(x.path)
    return image_paths

def extract_aligned_face(image, align_mat, size=256, padding=48, upper_padding=False):
    """
    get aligned face
    ---
    face_width(=height) = size - 2 * padding
    """
    matrix = align_mat * (size - 2 * padding)
    matrix[:, 2] += padding # 4 corners are padded
    if upper_padding:
        matrix[1, 2] += padding / 2
    facial_image = cv2.warpAffine(image, matrix, (size, size))
    return facial_image

# def torch_to_np(tensor, permute_channel=True):
#     """
#     - input
#         tensor: (1, 3, w, h) or (3, w, h)
#     - output
#         img: (w, h, 3)
#     """
#     *_, w, h = tensor.shape
#     tensor = tensor.data.cpu().numpy()
#     img = np.reshape(img, (3, w, h)).transpose((1, 2, 0))
#     if permute_channel:
#         img = img[:, :, ::-1] # channel permutation
#     return img