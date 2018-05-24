# Based on: https://gist.github.com/anonymous/d3815aba83a8f79779451262599b0955 
#           found on https://www.reddit.com/r/deepfakes/

import cv2
import numpy as np

import torch

from lib.aligner import get_align_mat
from lib.save_fig import imwrite

class Convert():
    def __init__(self, encoder, decoder, seamless_clone=False, mask_type="facehullandrect", **kwargs):
        self.encoder = encoder
        self.decoder = decoder
        self.seamless_clone = seamless_clone
        self.mask_type = mask_type.lower() # Choose in 'FaceHullAndRect','FaceHull','Rect'

    def patch_image(self, image, face_detected, size):
        """
        image: full image (containing at least one face)
        face_detected: containing landmark info
        size: model input size
        """
        # find affine transformation matrix to map 
        # given landmarks to aligned landmarks in (size, size) image
        mat = np.array(get_align_mat(face_detected)).reshape(2, 3) * size

        # get output image from faceswap model (64x64 image)
        new_face = self.get_new_face(image, mat, size)

        image_size = (image.shape[1], image.shape[0])
        image_mask = self.get_image_mask(image, new_face, face_detected, mat, image_size)

        return self.apply_new_face(image, new_face, image_mask, mat, image_size, size)

    def get_new_face(self, image, mat, size):
        face = cv2.warpAffine(image, mat, (size, size))
        normalized_tensor = torch.from_numpy(face.transpose(
                        (2, 0, 1))).float().div(255.0).unsqueeze_(0).cuda()
        new_face = self.decoder(self.encoder(normalized_tensor)) # it returns rgb, mask
        
        # ###
        # print('save_image')
        # nor_face_fig = torch.cat([normalized_tensor, normalized_tensor], dim=0)
        # new_face_fig = torch.cat([new_face, new_face], dim=0)
        # imwrite([nor_face_fig, new_face_fig], 'tmp.png', size=2)
        # ###
        # print('done saving')

        new_face = new_face.data.cpu().numpy()
        new_face = np.reshape(new_face, (3, 64, 64)).transpose((1, 2, 0))
        new_face = np.clip(new_face * 255, 0, 255).astype(image.dtype)
        
        return new_face

    def apply_new_face(self, image, new_face, image_mask, mat, image_size, size):
        base_image = np.copy(image)
        new_image = np.copy(image)

        cv2.warpAffine(new_face, mat, image_size, new_image, cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT)

        if self.seamless_clone:
            unitMask = np.clip(image_mask * 255, 0, 255).astype(np.uint8) # * 365??? * 255?
            maxregion = np.argwhere(unitMask==255)

            if maxregion.size > 0:
                miny, minx = maxregion.min(axis=0)[:2]
                maxy, maxx = maxregion.max(axis=0)[:2]
                lenx = maxx - minx
                leny = maxy - miny
                masky = int(minx + (lenx//2))
                maskx = int(miny + (leny//2))
                outimage = cv2.seamlessClone(
                    new_image.astype(np.uint8), base_image.astype(np.uint8),
                    unitMask, (masky, maskx), cv2.NORMAL_CLONE)
                return outimage

        foreground = cv2.multiply(image_mask, new_image.astype(float))
        background = cv2.multiply(1.0 - image_mask, base_image.astype(float))
        outimage = cv2.add(foreground, background)
        return outimage


    def get_image_mask(self, image, new_face, face_detected, mat, image_size):
        face_mask = np.zeros(image.shape, dtype=float)
        if 'rect' in self.mask_type:
            face_src = np.ones(new_face.shape, dtype=float)
            cv2.warpAffine(face_src, mat, image_size, face_mask, 
                cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT)

        hull_mask = np.zeros(image.shape,dtype=float)
        if 'hull' in self.mask_type:
            hull = cv2.convexHull(
                np.array(face_detected.landmark).reshape((-1,2)).astype(int)).flatten().reshape((-1,2))
            cv2.fillConvexPoly(
                hull_mask, hull, (1,1,1))

        if self.mask_type == 'rect':
            image_mask = face_mask
        elif self.mask_type == 'facehull':
            image_mask = hull_mask
        else:
            image_mask = ((face_mask*hull_mask))

        return image_mask