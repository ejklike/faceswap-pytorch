import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def save_fig(output_dir, epoch, input_data, output_data, target_data, size=8):
    # initialize figure
    scale = 20
    fig, axis = plt.subplots(3, size, figsize=(5 * scale, 2 * scale))

    input_data = input_data.data.cpu().numpy()
    output_data = output_data.data.cpu().numpy()
    target_data = target_data.data.cpu().numpy()

    def change_axis_and_colorspace(img):
        img = np.reshape(img, (3, 64, 64)).transpose((1, 2, 0))
        img = img[:, :, ::-1] # channel permutation
        return img

    for i in range(size):
        axis[0][i].imshow(change_axis_and_colorspace(input_data[i]))
        axis[0][i].set_xticks(()); axis[0][i].set_yticks(())

        axis[1][i].clear()
        axis[1][i].imshow(change_axis_and_colorspace(output_data[i]))
        axis[1][i].set_xticks(()); axis[1][i].set_yticks(())

        axis[2][i].clear()
        axis[2][i].imshow(change_axis_and_colorspace(target_data[i]))
        axis[2][i].set_xticks(()); axis[1][i].set_yticks(())
    
    axis[0][0].set_ylabel('input_data')
    axis[1][0].set_ylabel('output_data')
    axis[2][0].set_ylabel('target_data')

    fig.tight_layout()
    plt.savefig('{}/epoch_{:00000}.png'.format(output_dir, epoch))
    plt.close()


def save_fig(output_dir, epoch, input_data, mask_data, output_data, target_data, size=8):
    # initialize figure
    scale = 20
    fig, axis = plt.subplots(4, size, figsize=(5 * scale, 2 * scale))

    input_data = input_data.data.cpu().numpy()
    mask_data = mask_data.data.cpu().numpy()
    output_data = output_data.data.cpu().numpy()
    target_data = target_data.data.cpu().numpy()

    def change_axis_and_colorspace(img):
        img = np.reshape(img, (3, 64, 64)).transpose((1, 2, 0))
        img = img[:, :, ::-1] # channel permutation
        return img

    for i in range(size):
        axis[0][i].imshow(change_axis_and_colorspace(input_data[i]))
        axis[0][i].set_xticks(()); axis[0][i].set_yticks(())

        axis[1][i].clear()
        axis[1][i].imshow(change_axis_and_colorspace(mask_data[i]))
        axis[1][i].set_xticks(()); axis[1][i].set_yticks(())

        axis[2][i].clear()
        axis[2][i].imshow(change_axis_and_colorspace(output_data[i]))
        axis[2][i].set_xticks(()); axis[1][i].set_yticks(())

        axis[3][i].clear()
        axis[3][i].imshow(change_axis_and_colorspace(target_data[i]))
        axis[3][i].set_xticks(()); axis[1][i].set_yticks(())
    
    axis[0][0].set_ylabel('input_data')
    axis[1][0].set_ylabel('mask_data')
    axis[2][0].set_ylabel('output_data')
    axis[3][0].set_ylabel('target_data')

    fig.tight_layout()
    plt.savefig('{}/epoch_{:00000}.png'.format(output_dir, epoch))
    plt.close()