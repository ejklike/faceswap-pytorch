import argparse
import cv2
import os
import glob

from pathlib import Path
from tqdm import tqdm
import torch

from models import *
from lib.converter import Convert
from lib.face_alignment import FaceAlignment
from lib.utils import mkdir

# from lib.utils import get_target_paths, get_image_paths, get_folder

class DetectedFace(object):
    def __init__(self, face, landmark):
        self.x = face.left()
        self.y = face.top()
        self.w = face.right() - face.left()
        self.h = face.bottom() - face.top()
        self.landmark = landmark


class ConvertProcessor(object):

    images_found = 0
    num_faces_detected = 0

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        print("Input Directory: {}".format(self.input_dir))
        print("Output Directory: {}".format(self.output_dir))

        self.input_image_fnames = self.read_directory(self.input_dir)
        self.images_found = len(self.input_image_fnames)

        self.face_detector = FaceAlignment()

    def read_directory(self, directory):
        types = ('*.jpg', '*.png')
        images_list = []
        for files in types:
            images_list.extend(glob.glob(os.path.join(directory, files)))
        return images_list

    def prepare_images(self):
        for filename in tqdm(self.input_image_fnames):
            image = cv2.imread(filename)
            faces_and_landmarks = self.get_faces_and_landmarks(filename)
            # ###
            # simage = image.copy()
            # for d in faces_and_landmarks:
            #     cv2.rectangle(simage, (d.x, d.y), (d.x+d.w, d.y+d.h), (0,255,0) , 3)
            #     for x in d.landmark[17:]:
            #         cv2.circle(simage, (x[0], x[1]), 2, (0,0,255), -1)
            #     cv2.imwrite('./detected{}.jpg'.format(filename[-6:]), simage) ###
            # ###
            yield filename, image, faces_and_landmarks

    def get_faces_and_landmarks(self, iamge):
        for face, landmark in self.face_detector.get_landmarks(iamge):
            yield DetectedFace(face, landmark)
            self.num_faces_detected += 1

    def convert(self, encoder, decoder, **converter_args):
        converter = Convert(encoder, decoder, **converter_args)

        try:
            for filename, image, faces in self.prepare_images():
                for face in faces:
                    image = converter.patch_image(image, face, size=64)

                output_file = os.path.join(
                    self.output_dir, Path(filename).name)
                cv2.imwrite(str(output_file), image)
        except Exception as e:
            print('Failed to convert image: {}. Reason: {}'.format(filename, e))

    def finalize(self):
        print('-------------------------')
        print('Images found:        {}'.format(self.images_found))
        print('Faces detected:      {}'.format(self.num_faces_detected))
        print('-------------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a source image to a new one with the face swapped')

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

    # MODEL/OUTPUT DIR
    input_dir = args.input_dir
    output_dir = mkdir(os.path.join('./output', args.model_name, 'convert'))
    model_dir = mkdir(os.path.join('./output', args.model_name))

    # ENCODER
    encoder_args = dict(
        path=os.path.join(model_dir, 'encoder.pth'),
        init_dim=args.init_dim,
        code_dim=args.code_dim)
    encoder = get_model('encoder', FaceEncoder, device=device, **encoder_args).eval()
    print('')

    # DECODER
    decoder_args = dict(
        path=os.path.join(model_dir, 'decoder{}.pth'.format(args.target)))
    decoder = get_model('decoder_' + args.target, FaceDecoder, device=device, **decoder_args).eval()
    print('')

    processor = ConvertProcessor(args.input_dir, output_dir)
    converter_args = dict(
        seamless_clone=args.seamless_clone,
        mask_type=args.mask_type)
    processor.convert(encoder, decoder, **converter_args)
    processor.finalize()