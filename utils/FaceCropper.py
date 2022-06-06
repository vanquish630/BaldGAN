import os

import io
import numpy as np
from PIL import Image
import cv2
import IPython.display
# from natsort import natsorted
import matplotlib.pyplot as plt
import face_alignment
from skimage.transform import estimate_transform, warp, resize, rescale
import dlib
from .inverter import StyleGANInverter
from utils import segment
from  models.invert_model.helper import build_generator
import scipy.ndimage


class FaceCropper(object):
    def __init__(self, image_size , model_dir = "pretrained_models"):
        self.landmarkDetector68 = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        self.image_size = image_size
        self.model_dir = model_dir
        self.landmark_model_name = 'shape_predictor_68_face_landmarks.dat'
        self.mmod_model_name = 'mmod_human_face_detector.dat'

    def get_landmarks_68(self, image):
        landmark = self.landmarkDetector68.get_landmarks(image)

        if landmark is None:
            return None

        landmark = np.array(landmark)
        landmark.resize((68, 2))

        return landmark

    def faceDetector(self, image, use_mmod = False):

        self.use_mmod = use_mmod

        image = np.array(image)
        if self.use_mmod:
            mmod_face_detector_model_path = os.path.join(self.model_dir, self.mmod_model_name)
            self.face_detector = dlib.cnn_face_detection_model_v1(mmod_face_detector_model_path)
        else:
            self.face_detector = dlib.get_frontal_face_detector()


        bboxes = self.face_detector(image, 2)

        return bboxes

    def detect_face_simple(self , image):
        image = np.array(image)

        lmk68 = self.get_landmarks_68(np.array(image))

        left = np.min(lmk68[:, 0])
        right = np.max(lmk68[:, 0])
        bottom = np.max(lmk68[:, 1])
        top = np.min(lmk68[:, 1]) - (np.max(lmk68[:, 1]) - np.min(lmk68[:, 1]))

        old_size = (right - left + bottom - top) / 2 * 1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * 1.7)

        tl = [max(int(center[0] - size / 2), 0), max(int(center[1] - size / 2), 0)]
        tr = [min(int(center[0] + size / 2), image.shape[1]), max(int(center[1] - size / 2), 0)]

        bl = [max(int(center[0] - size / 2), 0), min(int(center[1] + size / 2), image.shape[0])]
        br = [max(int(center[0] + size / 2), image.shape[1]), min(int(center[1] + size / 2), image.shape[0])]

        start_y, end_y, start_x, end_x = tl[1], bl[1], tl[0], tr[0]

        src_pts = np.array([tl, bl, tr, br])

        cropped_image = image[start_y: end_y, start_x: end_x]

        return cropped_image, [start_y, end_y, start_x, end_x], lmk68

    def detect_face_multiple(self,image, bbox):

        left = bbox.left()
        right = bbox.right()
        bottom = bbox.bottom()
        top = bbox.top() - 20

        old_size = (right - left + bottom - top) / 2 * 1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * 2)

        tl = [max(int(center[0] - size / 2), 0), max(int(center[1] - size / 2), 0)]
        tr = [min(int(center[0] + size / 2), image.shape[1]), max(int(center[1] - size / 2), 0)]

        bl = [max(int(center[0] - size / 2), 0), min(int(center[1] + size / 2), image.shape[0])]
        br = [max(int(center[0] + size / 2), image.shape[1]), min(int(center[1] + size / 2), image.shape[0])]

        start_y, end_y, start_x, end_x = tl[1], bl[1], tl[0], tr[0]

        src_pts = np.array([tl, bl, tr, br])

        cropped_image = np.array(image)[start_y: end_y, start_x: end_x]

        return cropped_image , [start_y, end_y, start_x, end_x]

    def refineCrop(self , image , bbox_course):
        crop_refine, bbox_refine = segment.returnfacebbox(np.array(image), msk_type='full',
                                                                          getbbox=True, margin=0.1)
        final_bbox = [bbox_course[0] + bbox_refine[0],
                      bbox_course[0] + bbox_refine[1],
                      bbox_course[2] + bbox_refine[2],
                      bbox_course[2] + bbox_refine[3],
                      ]
        return crop_refine, final_bbox






