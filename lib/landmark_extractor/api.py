# https://github.com/1adrianb/face-alignment/blob/master/face_alignment/api.py

import os
import glob
import dlib
import torch
import torch.nn as nn
from skimage import io
try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from lib.landmark_extractor.models import FAN, ResNetDepth
from lib.landmark_extractor.utils import *


# class FacialLandmark(object):
#     def __init__(self, face, landmark):
#         self.x = face.left()
#         self.y = face.top()
#         self.w = face.right() - face.left()
#         self.h = face.bottom() - face.top()
#         self.landmark = landmark


class LandmarkExtractor(object):
    """Initialize the face alignment pipeline
    Args:
        enable_cuda (bool, optional): If True, all the computations will be done on a CUDA-enabled GPU (recommended).
        enable_cudnn (bool, optional): If True, cudnn library will be used in the benchmark mode
        use_cnn_face_detector (bool, optional): If True, dlib's CNN based face detector is used even if CUDA
                                                is disabled.
    Example:
        >>> FaceAlignment()
    """

    def __init__(self, enable_cuda=True, enable_cudnn=True, use_cnn_face_detector=False):
        self.enable_cuda = enable_cuda
        self.use_cnn_face_detector = use_cnn_face_detector

        if enable_cudnn and self.enable_cuda:
            torch.backends.cudnn.benchmark = True

        # Initialise the face detector
        if self.use_cnn_face_detector:
            path_to_detector = os.path.join(
                base_path, "mmod_human_face_detector.dat")
            if not os.path.isfile(path_to_detector):
                print("Downloading the face detection CNN. Please wait...")

                request_file.urlretrieve(
                    "https://www.adrianbulat.com/downloads/dlib/mmod_human_face_detector.dat",
                    os.path.join(path_to_detector))

            self.face_detector = dlib.cnn_face_detection_model_v1(
                path_to_detector)

        else:
            self.face_detector = dlib.get_frontal_face_detector()

        # Initialise the face alignemnt networks
        self.face_alignemnt_net = FAN(4)
        network_name = '2DFAN-4.pth.tar'
        fan_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), network_name)

        if not os.path.isfile(fan_path):
            print("Downloading the Face Alignment Network(FAN). Please wait...")

            request_file.urlretrieve(
                "https://www.adrianbulat.com/downloads/python-fan/" +
                network_name, os.path.join(fan_path))

        fan_weights = torch.load(
            fan_path,
            map_location=lambda storage,
            loc: storage)

        self.face_alignemnt_net.load_state_dict(fan_weights)

        if self.enable_cuda:
            self.face_alignemnt_net.cuda()
        self.face_alignemnt_net.eval()

    def _detect_faces(self, image):
        """Run the dlib face detector over an image
        Args:
            image (``ndarray`` object or string): either the path to the image or an image previosly opened
            on which face detection will be performed.
        Returns:
            Returns a list of detected faces
        """
        return self.face_detector(image, 1)

    def get_landmarks(self, input_image):
        with torch.no_grad():
            if isinstance(input_image, str):
                try:
                    image = io.imread(input_image)
                except IOError:
                    print("error opening file :: ", input_image)
                    return None
            else:
                image = input_image

            detected_faces = self._detect_faces(image)
            if len(detected_faces) > 0:
                landmarks = []
                for i, d in enumerate(detected_faces):
                    if self.use_cnn_face_detector:
                        d = d.rect

                    center = torch.FloatTensor(
                        [d.right() - (d.right() - d.left()) / 2.0,
                         d.bottom() - (d.bottom() - d.top()) / 2.0])
                    center[1] = center[1] - (d.bottom() - d.top()) * 0.12
                    scale = (d.right() - d.left() +
                             d.bottom() - d.top()) / 195.0

                    inp = crop(image, center, scale)
                    inp = torch.from_numpy(inp.transpose(
                        (2, 0, 1))).float().div(255.0).unsqueeze_(0)

                    if self.enable_cuda:
                        inp = inp.cuda()

                    out = self.face_alignemnt_net(inp)[-1].data.cpu()

                    pts, pts_img = get_preds_fromhm(out, center, scale)
                    pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)

                    landmarks.append(pts_img.numpy())
            else:
                print("Warning: No faces were detected.")
                return None

            return landmarks
            # return [FacialLandmark(face, landmark) 
                    # for face, landmark in zip(detected_faces, landmarks)]

    # def process_folder(self, path, all_faces=False):
    #     types = ('*.jpg', '*.png')
    #     images_list = []
    #     for files in types:
    #         images_list.extend(glob.glob(os.path.join(path, files)))

    #     predictions = []
    #     for image_name in images_list:
    #         predictions.append((
    #             image_name, self.get_landmarks(image_name, all_faces)))

    #     return predictions