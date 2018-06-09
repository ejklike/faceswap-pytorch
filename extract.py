import argparse
import cv2
import os
from pathlib import Path

from lib.directory_processor import DirectoryProcessor
from lib.utils import mkdir, extract_aligned_face


class ExtractProcessor(DirectoryProcessor):

    def __init__(self, input_dir, output_dir):
        super(ExtractProcessor, self).__init__(input_dir, output_dir)

    def extract_and_save_facial_images(self, size, padding, upper_padding=False,
                                       debug_landmarks=False):
        for filename, image, landmarks_and_matrices in self.read_images():
            try:
                processed = False

                for idx, (landmark, align_mat) in enumerate(landmarks_and_matrices):
                    # Draws landmarks for debug
                    if debug_landmarks is True:
                        for x, y in landmark:
                            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

                    facial_image = extract_aligned_face(
                        image, align_mat, size=size, padding=padding, upper_padding=upper_padding)
                    fname = os.path.join(
                        self.output_dir,
                        '{}_{}{}'.format(
                            Path(filename).stem, idx, Path(filename).suffix))
                    cv2.imwrite(fname, facial_image)

                    processed = True

                if processed is False:
                    fname = os.path.join(
                        mkdir(os.path.join(self.output_dir, 'no_face')),
                        Path(filename).name)
                    cv2.imwrite(fname, image)
                    continue

            except Exception as e:
                print('Failed to extract facial image: {}. Reason: {}'.format(filename, e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract facial images')

    parser.add_argument('-i', '--input-dir',
                        dest="input_dir",
                        default="./input/",
                        help="Input directory. A directory containing the files \
                        you wish to process. Defaults to './input/'")

    parser.add_argument('-o', '--output-dir',
                        dest="output_dir",
                        default="./output/",
                        help="Output directory. A destination to save processed \
                        images. Defaults to './output/'")

    parser.add_argument('-s', '--size', 
                        type=int, 
                        default=256,
                        help="Output image size (default: 256)")

    parser.add_argument('-p', '--padding', 
                        type=int, 
                        default=0,
                        help="Padding around the facial region (default: 0)")

    parser.add_argument('-up', '--upper-padding',
                        action="store_true",
                        default=False,
                        help="Apply additional padding to the upper facial region.")

    parser.add_argument('-dl', '--debug-landmarks',
                        action="store_true",
                        dest="debug_landmarks",
                        default=False,
                        help="Draw landmarks for debug.")

    args = parser.parse_args()
    processor = ExtractProcessor(
        args.input_dir, args.output_dir)
    processor_args = dict(
        size=args.size,
        padding=args.padding,
        upper_padding=args.upper_padding,
        debug_landmarks=args.debug_landmarks)
    processor.extract_and_save_facial_images(**processor_args)
    processor.finalize()