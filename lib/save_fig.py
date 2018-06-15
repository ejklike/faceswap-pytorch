import cv2
import numpy as np


def imwrite(img_list, fname, size=64, n=8):
    def _torch_to_img(tensor):
        return tensor.data.cpu().numpy().transpose((0,2,3,1))

    # (batch_size, 3, w, h) ==> (n, w, h, 3)
    img_list = [_torch_to_img(img_batch)[:n] for img_batch in img_list]
    # TODO: resize to given size
    # img_list = [ for img in img_list]
    figure = np.stack(img_list, axis=0)
    # print(figure.shape)
    
    # figure = figure.reshape((len(img_list), ) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip(figure * 255, 0, 255).astype('uint8')
    cv2.imwrite(fname, figure)


def stack_images(images):
    def get_transpose_axes(n):
        if n % 2 == 0:
            y_axes = list(range(1, n - 1, 2))
            x_axes = list(range(0, n - 1, 2))
        else:
            y_axes = list(range(0, n - 1, 2))
            x_axes = list(range(1, n - 1, 2))
        return y_axes, x_axes, [n - 1]

    images_shape = np.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [np.prod(images_shape[x]) for x in new_axes]
    return np.transpose(
        images,
        axes=np.concatenate(new_axes)
        ).reshape(new_shape)
