import argparse
import cv2
import os

from pathlib import Path
import torch
from torchvision import transforms

from lib.directory_processor import DirectoryProcessor
from lib.loader import PluginLoader
from lib.image_loader import MultiResFaceImages, FaceImages, ToTensor
from lib.utils import mkdir
from lib.save_fig import imwrite


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert a source image to a new one with the face swapped')

    parser.add_argument('-i', '--input-dir',
                        dest="input_dir",
                        default="./data/256_70_up/",
                        help="Input directory. A directory containing the files \
                        you wish to process. Defaults to './data/input/'")

    parser.add_argument('-m', '--model-name', 
                        dest='model_name', 
                        choices=PluginLoader.get_available_models(),
                        default=PluginLoader.get_default_model(),
                        help="select a model to train")

    parser.add_argument('-o', '--output-dir', 
                        dest='output_dir', 
                        default='output_dir',
                        help="output dir name (which will become output dir name)")

    parser.add_argument('-tg', '--target',
                        type=str,
                        default='7',
                        help="Select target.")

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
        data_dir=args.input_dir, 
        transform=transforms.Compose([ToTensor()]))


    Trainer = PluginLoader.get_trainer(args.model_name)
    trainer = Trainer(
        output_dir=args.output_dir, 
        no_cuda=args.no_cuda,
        batch_size=1)
    trainer.convert(args.target, dataset)