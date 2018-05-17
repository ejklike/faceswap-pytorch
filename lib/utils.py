import os

import numpy as np

def mkdir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory

def get_face_ids(input_dir='./data'):
    return sorted(os.listdir(input_dir))

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