import argparse
import cv2
import os

from pathlib import Path
import torch

from models import *
from lib.directory_processor import DirectoryProcessor
from lib.converter import Convert
from lib.utils import mkdir


class ConvertProcessor(DirectoryProcessor):

    def __init__(self, input_dir, output_dir):
        super(ConvertProcessor, self).__init__(input_dir, output_dir)

    def convert(self, encoder, decoder, **converter_args):
        converter = Convert(encoder, decoder, **converter_args)

        for filename, image, landmarks_and_matrices in self.read_images():
            try:
                for landmark, align_mat in landmarks_and_matrices:
                    image = converter.patch_image(image, landmark, align_mat, size=64)

                output_file = os.path.join(
                    self.output_dir, Path(filename).name)
                cv2.imwrite(str(output_file), image)
            except Exception as e:
                print('Failed to convert image: {}. Reason: {}'.format(filename, e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert a source image to a new one with the face swapped')

    parser.add_argument('-i', '--input-dir',
                        dest="input_dir",
                        default="./input/",
                        help="Input directory. A directory containing the files \
                        you wish to process. Defaults to './input/'")

    parser.add_argument('-n', '--model-name',
                        dest="model_name",
                        default="model_name",
                        help="Mode name. Defaults to 'model_name'")

    parser.add_argument('-tg', '--target',
                        type=str,
                        dest="target",
                        default='A',
                        help="Select target.")

    parser.add_argument('--init-dim', 
                        type=int, 
                        dest='init_dim', 
                        default=32,
                        help="the number of initial channel (default: 32)")
    
    parser.add_argument('--code-dim', 
                        type=int, 
                        dest='code_dim', 
                        default=1024,
                        help="the number of channel in encoded tensor (default: 1024)")

    parser.add_argument('-S', '--seamless',
                        action="store_true",
                        dest="seamless_clone",
                        default=False,
                        help="Use cv2's seamless clone. (Masked converter only)")

    parser.add_argument('-M', '--mask-type',
                        type=str.lower, #lowercase this, because its just a string later on.
                        dest="mask_type",
                        choices=["rect", "facehull", "facehullandrect"],
                        default="facehullandrect",
                        help="Mask to use to replace faces. (Masked converter only)")

    parser.add_argument('--no-cuda', 
                        action='store_true', 
                        default=False,
                        help='disables CUDA training')

    args = parser.parse_args()

    # CUDA/CUDNN setting
    torch.backends.cudnn.benchmark = True
    use_cuda = args.no_cuda is False and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # DATALOADER
    dataset = MultiResFaceImages(
        data_dir=os.path.join(args.input_dir), 
        transform=transforms.Compose([ToTensor()]))


    Trainer = PluginLoader.get_trainer(args.model_name)
    trainer = Trainer(
        output_dir=args.output_dir, 
        no_cuda=args.no_cuda,
        batch_size=1)

    processor = ConvertProcessor(args.input_dir, output_dir)
    converter_args = dict(
        seamless_clone=args.seamless_clone,
        mask_type=args.mask_type)
    processor.convert(encoder, decoder, **converter_args)
    processor.finalize()