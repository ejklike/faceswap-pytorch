import cv2
from glob import glob
import os

from tqdm import tqdm

from lib.landmark_extractor import LandmarkExtractor
from lib.landmark_aligner import get_align_mat
from lib.utils import mkdir


class DirectoryProcessor(object):
    """
    read image, and get facial landmarks
    """

    images_found = 0
    num_faces_detected = 0

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = mkdir(output_dir)

        print("Input Directory: {}".format(self.input_dir))
        print("Output Directory: {}".format(self.output_dir))

        assert os.path.exists(self.input_dir)

        self.input_image_fnames = self._read_directory(self.input_dir)
        self.images_found = len(self.input_image_fnames)

        self.landmark_extractor = LandmarkExtractor()

    def _read_directory(self, directory):
        types = ('*.jpg', '*.png')
        image_fname_list = []
        for files in types:
            image_fname_list.extend(glob(os.path.join(directory, files)))
        return image_fname_list

    def _get_points_and_mapping(self, fname, jawline=False):
        """
        points: facial landmarks
        mapping: from landmark points to mean facial landmarks in [0, 1] x [0, 1]
        """
        for facial_landmark in self.landmark_extractor.get_landmarks(fname):
            yield facial_landmark, get_align_mat(facial_landmark, jawline=jawline)
            self.num_faces_detected += 1

    def read_images(self, jawline=False):
        for filename in tqdm(self.input_image_fnames):
            image = cv2.imread(filename)
            landmarks_and_matrices = self._get_points_and_mapping(filename, jawline=jawline)
            yield filename, image, landmarks_and_matrices

    def finalize(self):
        print('-------------------------')
        print('Images found:        {}'.format(self.images_found))
        print('Faces detected:      {}'.format(self.num_faces_detected))
        print('-------------------------')