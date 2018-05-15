import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def save_fig(output_dir, epoch, img_list, size=8):
    # initialize figure
    scale = 20
    img_count = len(img_list) # num_column
    fig, axis = plt.subplots(img_count, size, figsize=(5 * scale, 2 * scale))

    img_list = [img_batch.data.cpu().numpy() for img_batch in img_list]

    def change_axis_and_colorspace(img):
        img = np.reshape(img, (3, 64, 64)).transpose((1, 2, 0))
        img = img[:, :, ::-1] # channel permutation
        return img

    for i in range(size):
        for j, img_data in enumerate(img_list):
            axis[j][i].clear()
            axis[j][i].imshow(change_axis_and_colorspace(img_data[i]))
            axis[j][i].set_xticks(()); axis[j][i].set_yticks(())

    fig.tight_layout()
    plt.savefig('{}/epoch_{:00000}.png'.format(output_dir, epoch))
    plt.close()